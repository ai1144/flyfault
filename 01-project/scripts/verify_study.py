"""Audit one completed FLY//FAULT demo and decode its comparison replay."""

import argparse
import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np

from flyfault.rendering import validate_video
from flyfault.storage import load_json
from flyfault.viz_data import paired_improvements


def audit_episode(directory: Path) -> tuple[dict, dict, np.ndarray]:
    manifest = load_json(directory / "manifest.json")
    summary = load_json(directory / "summary.json")
    assert manifest["status"] == "completed", directory
    assert all((directory / path).is_file() for path in manifest["artifacts"].values()), directory
    assert sha256((directory / "model.mjb").read_bytes()).hexdigest() == (
        manifest["physical_model"]["model_sha256"]
    ), directory
    with np.load(directory / "signals.npz") as data, np.load(directory / "trajectory.npz") as replay:
        assert all(np.isfinite(data[key]).all() for key in data.files), directory
        assert np.isfinite(replay["qpos"]).all() and np.isfinite(replay["time"]).all(), directory
        np.testing.assert_allclose(data["applied_command"][1:], data["command"][:-1])
        np.testing.assert_allclose(data["readout_time_s"], data["pose_time_s"][1:])
        np.testing.assert_allclose(data["decision_time_s"], data["pose_time_s"][:-1])
        np.testing.assert_allclose(data["applied_command"][0], manifest["readout"]["neutral_drive"])
        dt = manifest["timesteps_s"]["decision"]
        step = manifest["timesteps_s"]["neural"]
        windows = np.rint(data["spike_time_s"] / step).astype(int) // round(dt / step)
        rates = np.zeros_like(data["rates_hz"])
        for index, group in enumerate(manifest["network"]["populations"]):
            start, stop = group["start"], group["stop"]
            mask = (data["spike_neuron"] >= start) & (data["spike_neuron"] < stop)
            counts = np.bincount(windows[mask], minlength=len(rates))
            rates[:, index] = counts / ((stop - start) * dt)
        np.testing.assert_allclose(data["rates_hz"], rates)
        assert summary["total_spikes"] == len(windows), directory
        intervention = manifest["intervention"]
        assert not np.isin(data["spike_neuron"], intervention["affected_neuron_indices"]).any()
        stimulation = intervention["stimulation"]
        if stimulation:
            left, right = data["odor"].T
            expected = np.minimum(
                stimulation["max_current_pa"], stimulation["level_gain_pa"] * left
                + stimulation["positive_left_contrast_gain_pa"] * np.maximum(left - right, 0),
            )
            np.testing.assert_allclose(data["rescue_current_pa"], expected)
        else:
            np.testing.assert_allclose(data["rescue_current_pa"], 0)
        if intervention["descending_frozen"]:
            np.testing.assert_allclose(data["applied_command"], manifest["readout"]["neutral_drive"])
        assert summary["path_length_mm"] > 0, directory
        assert np.linalg.norm(replay["qpos"][-1] - replay["qpos"][0]) > 0, directory
        assert abs(replay["time"][0]) < 1e-9 and replay["time"][-1] <= summary["duration_s"] + 1e-9
        start_pose = data["xyz_mm"][0].copy()
    return manifest, summary, start_pose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    args = parser.parse_args()
    batch = load_json(args.batch_dir / "manifest.json")
    assert batch["kind"] == "batch" and batch["status"] == "completed"
    assert batch["episodes"] == batch["planned_episodes"]
    pairs = defaultdict(dict)
    summaries = []
    for episode in batch["episodes"]:
        manifest, summary, start_pose = audit_episode(args.batch_dir / episode)
        assert manifest["science_sha256"] == batch["science_sha256"]
        assert manifest["config_sha256"] == batch["config_sha256"]
        pairs[manifest["seed"], manifest["stimulus_side"]][manifest["condition"]] = manifest, start_pose
        summaries.append(summary)
    for arms in pairs.values():
        intact, pose = arms["intact"]
        for manifest, start_pose in arms.values():
            np.testing.assert_allclose(start_pose, pose, atol=1e-9, rtol=0)
            assert manifest["config"]["arena"] == intact["config"]["arena"]
        lesion = arms["lesion_left"][0]["intervention"]["affected_neuron_indices"]
        assert lesion == arms["rescue"][0]["intervention"]["affected_neuron_indices"]
        assert len(lesion) == len(arms["sham_random"][0]["intervention"]["affected_neuron_indices"])
    video = validate_video(args.batch_dir / batch["artifacts"]["comparison_mp4"])
    selected = pairs[batch["config"]["seeds"][0], batch["config"]["stimulus_sides"][0]]
    frame_count = max(selected[arm][0]["physical_model"]["frame_count"] for arm in (
        "intact", "lesion_left", "rescue",
    ))
    assert video["frame_count"] == frame_count
    assert video["width"] == 1920 and video["height"] == 1080
    assert video["first_to_last_mean_abs_pixel_delta"] > 1
    outcomes = {}
    for condition in batch["config"]["conditions"]:
        rows = [row for row in summaries if row["condition"] == condition]
        outcomes[condition] = {
            "n": len(rows),
            "reached": sum(row["reached"] for row in rows),
            "flipped": sum(row["flipped"] for row in rows),
            "out_of_bounds": sum(row["out_of_bounds"] for row in rows),
            "mean_distance_improvement_mm": float(np.mean([r["distance_improvement_mm"] for r in rows])),
            "mean_path_length_mm": float(np.mean([r["path_length_mm"] for r in rows])),
        }
    print(json.dumps({
        "run": batch["run_id"], "status": "passed", "episodes_audited": len(summaries),
        "spikes_audited": sum(row["total_spikes"] for row in summaries), "video": video,
        "outcomes": outcomes, "paired_vs_intact": paired_improvements(summaries),
        "paired_rescue_vs_left_lesion": paired_improvements(summaries, "lesion_left")["rescue"],
    }, indent=2))


if __name__ == "__main__":
    main()
