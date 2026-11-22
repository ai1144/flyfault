"""FlyGym v2.1.1's physical turning controller with held descending commands."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from hashlib import sha256
from math import cos, sin
from pathlib import Path

import mujoco
import numpy as np
from flygym import Simulation, assets_dir
from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    HybridControllerObservation,
    HybridTurningController,
    LocomotionAction,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
)
from numpy.typing import ArrayLike, NDArray

FLY_NAME = "flyfault"
CAMERA = "flyfault/replay"
MAX_DESCENDING_DRIVE = 1.2
SPAWN_HEIGHT_MM = 0.8
WARMUP_S = 0.05


@dataclass(frozen=True)
class Pose:
    """Thorax frame in world coordinates; upright is the dorsal-axis vertical component."""

    xyz_mm: NDArray[np.float64]
    body_to_world: NDArray[np.float64]
    heading_rad: float
    upright: float


@cache
def bundled_asset_provenance() -> dict:
    """Hash the installed NeuroMechFly inputs once per Python process."""
    model_assets = Path(assets_dir) / "model" / "neuromechfly"
    inputs = sorted(model_assets.glob("*.yaml"))
    inputs += sorted(model_assets.glob("meshes/*/*.stl"))
    return {
        "origin": "NeuroMechFly meshes and YAML supplied by the FlyGym distribution",
        "mesh_resolution": "SIMPLIFIED_MAX2000FACES",
        "files_sha256": {
            str(path.relative_to(model_assets)): sha256(path.read_bytes()).hexdigest()
            for path in inputs
        },
    }


def compose_arena(
    start_xy_mm: tuple[float, float],
    start_yaw_rad: float,
    source_xy_mm: tuple[float, float],
    source_radius_mm: float,
):
    """Official flat-ground walking world with visual-only odor-source sites."""
    fly = make_locomotion_fly(name=FLY_NAME, add_adhesion=True, colorize=True)
    camera = fly.add_tracking_camera(
        name="replay",
        pos_offset=(-0.35, -5.7, 4.0),
        rotation=Rotation3D("xyaxes", (1, 0, 0, 0, 0.57, 0.82)),
        fovy=37.0,
    )
    world = FlatGroundWorld()
    world.mjcf_root.material("grid").reflectance = 0.03
    checker = world.mjcf_root.texture("checker")
    checker.rgb1 = (0.27, 0.30, 0.32)
    checker.rgb2 = (0.32, 0.35, 0.37)
    world.mjcf_root.worldbody.add_site(
        name="odor_source",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=(*source_xy_mm, 0.012),
        size=(0.38, 0.012, 0.0),
        rgba=(1.0, 0.62, 0.24, 1.0),
    )
    world.mjcf_root.worldbody.add_site(
        name="odor_target_radius",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=(*source_xy_mm, 0.003),
        size=(source_radius_mm, 0.003, 0.0),
        rgba=(0.99, 0.61, 0.22, 0.20),
    )
    world.add_fly(
        fly,
        (*start_xy_mm, SPAWN_HEIGHT_MM),
        Rotation3D("quat", (cos(start_yaw_rad / 2), 0.0, 0.0, sin(start_yaw_rad / 2))),
        bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
        add_ground_contact_sensors=False,
    )
    return fly, world, camera


class FlyPhysics:
    """One seeded NeuroMechFly and the official two-channel motor adapter."""

    def __init__(
        self,
        seed: int,
        *,
        start_xy_mm: tuple[float, float] = (0.0, 0.0),
        start_yaw_rad: float = 0.0,
        source_xy_mm: tuple[float, float] = (18.0, 8.0),
        source_radius_mm: float = 3.0,
        fps: int = 25,
        playback_speed: float = 0.25,
        physics_dt_s: float | None = None,
    ) -> None:
        if fps <= 0 or not np.isfinite(playback_speed) or playback_speed <= 0:
            raise ValueError("replay frame rate and playback speed must be positive")
        self.fly, world, camera = compose_arena(
            start_xy_mm, start_yaw_rad, source_xy_mm, source_radius_mm
        )
        self.sim = Simulation(world, timestep=physics_dt_s)
        self.camera_name = camera.name
        self.dt = float(self.sim.timestep)
        interval_steps = playback_speed / (fps * self.dt)
        if interval_steps < 1 or not np.isclose(
            interval_steps, round(interval_steps), rtol=0.0, atol=1e-7
        ):
            raise ValueError("playback_speed/fps must span an integer number of physics steps")

        dofs = self.fly.get_actuated_jointdofs_order("position")
        preprogrammed = PreprogrammedSteps()
        self.controller = HybridTurningController(
            timestep=self.dt,
            preprogrammed_steps=preprogrammed,
            output_dof_order=dofs,
        )
        self._thorax_name = self.fly.bodyseg_to_mjcfbody[BodySegment("c_thorax")].name
        self._pose_data = mujoco.MjData(self.sim.mj_model)
        self.sim.reset()
        self.controller.reset(seed=seed)
        neutral_action = LocomotionAction(
            joint_angles=preprogrammed.default_pose_by_dof_order(dofs),
            adhesion_onoff=np.ones(6, dtype=bool),
        )
        apply_locomotion_action(self.sim, self.fly.name, neutral_action)
        self.sim.warmup(WARMUP_S)
        self._time_origin = self.sim.time
        self.recorder = self.sim.set_trajectory_recorder(
            playback_speed=playback_speed, output_fps=fps
        )
        self.sim.record_as_needed()

    @property
    def time_s(self) -> float:
        """Episode-relative time, excluding initial settling."""
        return float(self.sim.time - self._time_origin)

    def pose(self) -> Pose:
        # mj_step can leave derived transforms at the previous evaluation time.
        # Recompute against current qpos on a separate MjData without altering dynamics.
        self._pose_data.qpos[:] = self.sim.mj_data.qpos
        mujoco.mj_kinematics(self.sim.mj_model, self._pose_data)
        thorax = self._pose_data.body(self._thorax_name)
        xyz = thorax.xpos.copy()
        rotation = thorax.xmat.reshape(3, 3).copy()
        heading = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        return Pose(xyz, rotation, heading, float(rotation[2, 2]))

    def step(self, command: ArrayLike, n_steps: int) -> None:
        """Hold bounded left and right CPG amplitudes across MuJoCo steps."""
        drive = np.asarray(command, dtype=float)
        if drive.shape != (2,) or not np.isfinite(drive).all():
            raise ValueError("descending command must be a finite [left, right] pair")
        if not isinstance(n_steps, (int, np.integer)) or n_steps < 0:
            raise ValueError("n_steps must be a nonnegative integer")
        drive = np.clip(drive, 0.0, MAX_DESCENDING_DRIVE)
        initial_time = self.sim.time
        for _ in range(n_steps):
            observation = HybridControllerObservation.from_sim(self.sim, self.fly.name)
            action = self.controller.step(drive, observation)
            apply_locomotion_action(self.sim, self.fly.name, action)
            self.sim.step()
            self.sim.record_as_needed()
        data = self.sim.mj_data
        if (
            abs(self.sim.time - initial_time - n_steps * self.dt) > self.dt / 4
            or not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
        ):
            raise RuntimeError("MuJoCo clock reset or the physical state became nonfinite")

    def save_replay(self, run_dir: Path) -> dict:
        """Persist official FlyGym frames and their exact compatible compiled model."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        trajectory = self.recorder.trajectory
        trajectory.time -= self._time_origin
        tmp_traj = run_dir / ".trajectory.tmp.npz"
        trajectory.save(tmp_traj)
        tmp_traj.replace(run_dir / "trajectory.npz")
        tmp_model = run_dir / ".model.tmp.mjb"
        mujoco.mj_saveModel(self.sim.mj_model, str(tmp_model), None)
        model_sha = sha256(tmp_model.read_bytes()).hexdigest()
        tmp_model.replace(run_dir / "model.mjb")
        return {
            "physical_model": "NeuroMechFly legs-only / FlatGroundWorld",
            "controller": "flygym_demo.complex_terrain.HybridTurningController",
            "camera": self.camera_name,
            "assets": bundled_asset_provenance(),
            "model_sha256": model_sha,
            "qpos_size": self.sim.mj_model.nq,
            "frame_count": trajectory.n_frames,
            "fps": trajectory.output_fps,
            "playback_speed": trajectory.playback_speed,
            "warmup_s": self._time_origin,
            "first_frame_s": float(trajectory.time[0]),
            "last_frame_s": float(trajectory.time[-1]),
            "episode_end_s": self.time_s,
        }


def calibrate_locomotion(seed: int = 42, duration_s: float = 0.4) -> dict:
    """Measure actual physical responses to three fixed, non-neural commands."""
    results = {}
    for name, command in {"right": (1.2, 0.4), "straight": (0.8, 0.8), "left": (0.4, 1.2)}.items():
        physics = FlyPhysics(seed)
        initial = physics.pose()
        physics.step(command, round(duration_s / physics.dt))
        final = physics.pose()
        delta = final.xyz_mm - initial.xyz_mm
        yaw_change = final.heading_rad - initial.heading_rad
        results[name] = {
            "command_left_right": command,
            "sim_duration_s": physics.time_s,
            "displacement_xyz_mm": delta.tolist(),
            "heading_change_deg": float(np.rad2deg(np.arctan2(sin(yaw_change), cos(yaw_change)))),
            "final_upright": final.upright,
        }
    return {"seed": seed, "requested_duration_s": duration_s, "trials": results}
