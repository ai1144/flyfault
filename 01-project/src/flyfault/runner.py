"""Causal closed loop, persisted episodes, and resumable paired batches."""

from __future__ import annotations

import csv
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import CONDITIONS, STIMULUS_SIDES, Arena, Protocol, grid_steps
from .metrics import behavior_summary
from .storage import (
    artifact_lock,
    atomic,
    fingerprint,
    load_json,
    now,
    runtime,
    save_json,
    save_signals,
    science_digest,
)

UNITS = {
    "positions": "mm", "time": "s", "heading": "rad", "heading_metrics": "deg",
    "rate": "Hz", "current": "pA", "voltage": "mV", "odor": "normalized concentration",
    "descending_command": "dimensionless",
}


def episode_key(protocol: Protocol, sha: str, condition: str, seed: int, side: str) -> str:
    digest = fingerprint({
        "config_sha256": sha, "science_sha256": science_digest(),
        "packages": runtime()["packages"], "python": runtime()["python"],
        "condition": condition, "seed": seed, "stimulus_side": side,
    })
    return f"{protocol.name}-{condition}-{seed}-{side}-{digest[:12]}"


def _archive_attempt(directory: Path, manifest: dict) -> None:
    attempt = manifest["started_at"].replace(":", "-")
    archive = directory / "previous_attempts" / attempt
    archive.mkdir(parents=True)
    for item in directory.iterdir():
        if item.name not in {".lock", "previous_attempts"}:
            item.rename(archive / item.name)


def _terminal(pose, arena) -> str | None:
    if not np.isfinite([*pose.xyz_mm, pose.heading_rad, pose.upright]).all():
        raise FloatingPointError("Non-finite physical pose")
    if pose.upright <= 0:
        return "flipped"
    if np.abs(pose.xyz_mm[:2]).max() >= arena.limit_mm:
        return "out_of_bounds"
    if np.linalg.norm(pose.xyz_mm[:2] - arena.source_xy_mm) <= arena.source_radius_mm:
        return "source_reached"
    return None


def _simulate(
    directory: Path, protocol: Protocol, arena: Arena, condition: str, seed: int, manifest: dict,
) -> None:
    from .locomotion import FlyPhysics
    from .network import Microcircuit
    from .readout import DescendingReadout
    from .sensing import ANTENNAL_OFFSETS_MM, GaussianOdorField

    timing = protocol.timing
    physics = FlyPhysics(
        seed, start_xy_mm=arena.start_xy_mm, start_yaw_rad=arena.start_yaw_rad,
        source_xy_mm=arena.source_xy_mm, source_radius_mm=arena.source_radius_mm,
        fps=protocol.media.fps, playback_speed=protocol.media.playback_speed,
    )
    field = GaussianOdorField(arena.source_xy_mm, arena.sigma_mm)
    network = Microcircuit(seed, condition, timing.neural_dt_s, timing.decision_s,
                           protocol.circuit.lesion_fraction)
    readout = DescendingReadout(timing.decision_s)
    steps_per_decision = grid_steps(timing.decision_s, physics.dt, "decision_s")
    decisions = grid_steps(timing.duration_s, timing.decision_s, "duration_s")
    manifest["timesteps_s"] = {
        "physics": physics.dt, "neural": timing.neural_dt_s, "decision": timing.decision_s,
    }
    manifest["network"] = network.metadata()
    manifest["intervention"] = manifest["network"]["intervention"]
    manifest["readout"] = readout.metadata()
    manifest["sensory_model"] = {
        "type": "normalized planar Gaussian sampled by body-fixed virtual antennae",
        "field_center_mm": arena.source_xy_mm, "sigma_mm": arena.sigma_mm,
        "antennae_thorax_offsets_mm_left_right": ANTENNAL_OFFSETS_MM,
        "controller_observations": ["left local concentration", "right local concentration"],
    }

    pose = physics.pose()
    poses = [pose]
    pose_times = [physics.time_s]
    held = np.asarray(readout.neutral, dtype=float)
    odours, commands, applied, rates, currents = [], [], [], [], []
    neural_costs, physics_costs = [], []
    trial_start = perf_counter()
    terminal = "timeout"
    for index in range(decisions):
        # Both systems advance the same interval using only information at its start.
        sensory = field.read(pose.xyz_mm, pose.body_to_world)
        tic = perf_counter()
        response = network.update(sensory)
        command = readout.update(response.rates_hz, frozen=condition == "no_neural_control")
        neural_costs.append(perf_counter() - tic)
        tic = perf_counter()
        physics.step(held, steps_per_decision)
        physics_costs.append(perf_counter() - tic)
        pose = physics.pose()
        expected_time = (index + 1) * timing.decision_s
        if not np.isclose(physics.time_s, expected_time, atol=1e-9, rtol=0):
            raise RuntimeError(f"Physical clock drifted at decision {index}")
        poses.append(pose)
        pose_times.append(physics.time_s)
        odours.append(sensory)
        applied.append(held.copy())
        commands.append(np.asarray(command).copy())
        rates.append(np.asarray(response.rates_hz).copy())
        currents.append(response.rescue_current_pa)
        held = command
        reason = _terminal(pose, arena)
        if reason:
            terminal = reason
            break

    wall_s = perf_counter() - trial_start
    manifest["cpu_benchmark"] = {
        "loop_wall_s": wall_s,
        "simulated_s_per_wall_s": physics.time_s / wall_s,
        "decision_updates": len(commands),
        "mean_neural_update_ms": float(np.mean(neural_costs) * 1000),
        "p95_neural_update_ms": float(np.percentile(neural_costs, 95) * 1000),
        "mean_physics_per_decision_ms": float(np.mean(physics_costs) * 1000),
    }
    spike_time, spike_neuron = network.spikes()
    state_time, voltage_mv = network.state_samples()
    time_s = np.asarray(pose_times)
    xyz = np.array([sample.xyz_mm for sample in poses])
    heading = np.array([sample.heading_rad for sample in poses])
    save_signals(
        directory / "signals.npz", pose_time_s=time_s, xyz_mm=xyz, heading_rad=heading,
        upright=np.array([sample.upright for sample in poses]), decision_time_s=time_s[:-1],
        readout_time_s=time_s[1:], odor=np.array(odours), command=np.array(commands),
        applied_command=np.array(applied), rates_hz=np.array(rates),
        rescue_current_pa=np.array(currents), spike_time_s=spike_time, spike_neuron=spike_neuron,
        state_time_s=state_time, state_voltage_mv=voltage_mv,
    )
    summary = behavior_summary(
        time_s, xyz[:, :2], heading, arena.source_xy_mm, arena.source_radius_mm, terminal,
    )
    summary.update(
        condition=condition, seed=seed, stimulus_side=manifest["stimulus_side"],
        mean_rate_hz=np.mean(rates, axis=0).tolist(),
        total_spikes=len(spike_time),
    )
    manifest["physical_model"] = physics.save_replay(directory)
    save_json(directory / "replay.json", manifest["physical_model"])
    manifest["artifacts"] = {
        "signals": "signals.npz", "summary": "summary.json", "trajectory": "trajectory.npz",
        "model": "model.mjb", "replay": "replay.json",
    }
    save_json(directory / "summary.json", summary)


def run_episode(
    root: Path, protocol: Protocol, sha: str, condition: str, seed: int, side: str,
) -> Path:
    if condition not in CONDITIONS or side not in STIMULUS_SIDES or not 0 <= seed < 2**32:
        raise ValueError("Unknown condition/side or invalid unsigned 32-bit seed")
    directory = root / episode_key(protocol, sha, condition, seed, side)
    # The protocol's canonical-side validation applies to its input, not a mirrored episode.
    arena = protocol.arena.for_side(side)
    config = asdict(protocol)
    config["arena"] = asdict(arena)
    with artifact_lock(directory):
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            previous = load_json(manifest_path)
            if previous["status"] == "completed":
                missing = [p for p in previous["artifacts"].values() if not (directory / p).exists()]
                if not missing:
                    return directory
            _archive_attempt(directory, previous)
        manifest = {
            **runtime(), "kind": "episode", "run_id": directory.name, "status": "running",
            "condition": condition, "seed": seed, "stimulus_side": side, "config": config,
            "config_sha256": sha, "effective_config_sha256": fingerprint(config),
            "science_sha256": science_digest(), "started_at": now(), "command": sys.argv,
            "units": UNITS, "artifacts": {},
        }
        save_json(manifest_path, manifest)
        try:
            _simulate(directory, protocol, arena, condition, seed, manifest)
        except BaseException as error:
            manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        else:
            manifest.update(status="completed")
        finally:
            manifest["finished_at"] = now()
            save_json(manifest_path, manifest)
    return directory


def run_batch(root: Path, protocol: Protocol, sha: str) -> Path:
    trials = [(condition, seed, side) for seed in protocol.seeds
              for side in protocol.stimulus_sides for condition in protocol.conditions]
    keys = [episode_key(protocol, sha, *trial) for trial in trials]
    directory = root / f"{protocol.name}-batch-{fingerprint(keys)[:12]}"
    with artifact_lock(directory):
        path = directory / "manifest.json"
        previous = load_json(path) if path.exists() else {}
        manifest = {
            **runtime(), "kind": "batch", "run_id": directory.name, "status": "running",
            "config": asdict(protocol), "config_sha256": sha, "science_sha256": science_digest(),
            "started_at": previous.get("started_at", now()), "updated_at": now(),
            "command": sys.argv, "units": UNITS, "planned_episodes": [f"episodes/{k}" for k in keys],
            "episodes": [], "artifacts": previous.get("artifacts", {}),
        }
        save_json(path, manifest)
        rows = []
        try:
            for index, trial in enumerate(trials, start=1):
                print(f"[{index}/{len(trials)}] {trial[0]} seed={trial[1]} side={trial[2]}", flush=True)
                episode = run_episode(directory / "episodes", protocol, sha, *trial)
                manifest["episodes"].append(episode.relative_to(directory).as_posix())
                rows.append(load_json(episode / "summary.json"))
                with atomic(directory / "summary.csv") as file:
                    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                manifest["artifacts"]["summary"] = "summary.csv"
                manifest["updated_at"] = now()
                save_json(path, manifest)
        except BaseException as error:
            manifest.update(status="incomplete", error=f"{type(error).__name__}: {error}")
            raise
        else:
            manifest.update(status="completed")
        finally:
            manifest["updated_at"] = now()
            save_json(path, manifest)
    return directory
