import mujoco
import numpy as np
from flygym.recording import RecordedTrajectory
from pytest import approx, raises

from flyfault.locomotion import FlyPhysics, calibrate_locomotion


def test_official_hybrid_controller_turns_left_and_right():
    measured = calibrate_locomotion()["trials"]
    assert measured["left"]["heading_change_deg"] > 15
    assert measured["right"]["heading_change_deg"] < -15
    assert abs(measured["straight"]["heading_change_deg"]) < 15
    for trial in measured.values():
        assert trial["displacement_xyz_mm"][0] > 1
        assert trial["final_upright"] > 0.9


def test_replay_keeps_the_physics_clock_and_actual_pose(tmp_path):
    physics = FlyPhysics(42)
    initial = physics.pose()
    with raises(ValueError, match="descending"):
        physics.step((np.nan, 1.0), 1)
    physics.step((0.4, 1.2), 200)
    assert physics.time_s == approx(0.02, abs=1e-12)
    metadata = physics.save_replay(tmp_path)
    saved = RecordedTrajectory.from_file(tmp_path / "trajectory.npz")
    model = mujoco.MjModel.from_binary_path(str(tmp_path / "model.mjb"))
    assert saved.time.tolist() == approx([0, 0.01, 0.02], abs=1e-12)
    assert metadata["last_frame_s"] == approx(physics.time_s)
    assert saved.qpos[-1] == approx(physics.sim.mj_data.qpos)
    assert saved.qpos.shape == (3, model.nq)
    data = mujoco.MjData(model)
    data.qpos[:] = saved.qpos[0]
    mujoco.mj_kinematics(model, data)
    assert data.body("flyfault/c_thorax").xpos == approx(initial.xyz_mm)
