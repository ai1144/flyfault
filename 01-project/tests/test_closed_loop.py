from pathlib import Path

import numpy as np

from flyfault.config import load_protocol
from flyfault.runner import run_episode
from flyfault.storage import load_json


def test_paired_closed_loop_causality_and_reproduction(tmp_path):
    config, digest = load_protocol(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    left = run_episode(tmp_path / "first", config, digest, "intact", 42, "left")
    again = run_episode(tmp_path / "second", config, digest, "intact", 42, "left")
    right = run_episode(tmp_path / "first", config, digest, "intact", 42, "right")
    baseline = run_episode(tmp_path / "first", config, digest, "no_neural_control", 42, "left")
    with (np.load(left / "signals.npz") as a, np.load(again / "signals.npz") as b,
          np.load(right / "signals.npz") as mirrored, np.load(baseline / "signals.npz") as frozen):
        assert a["odor"][0, 0] > a["odor"][0, 1]
        assert mirrored["odor"][0, 0] < mirrored["odor"][0, 1]
        assert a["spike_time_s"].size
        for name in a.files:
            assert np.isfinite(a[name]).all()
            np.testing.assert_allclose(a[name], b[name], atol=1e-9, rtol=0)
        np.testing.assert_allclose(a["applied_command"][1:], a["command"][:-1])
        np.testing.assert_allclose(a["readout_time_s"][:-1], a["decision_time_s"][1:])
        np.testing.assert_allclose(frozen["applied_command"], 0.82)
        assert np.linalg.norm(a["xyz_mm"][-1, :2] - a["xyz_mm"][0, :2]) > 0.1
        assert np.linalg.norm(a["xyz_mm"][-1, :2] - frozen["xyz_mm"][-1, :2]) > 0.001
        with np.load(left / "trajectory.npz") as trajectory:
            assert np.isclose(trajectory["time"][0], 0)
            assert np.isclose(trajectory["time"][-1], a["pose_time_s"][-1])
    manifest = load_json(left / "manifest.json")
    assert manifest["status"] == "completed"
    assert manifest["config_sha256"] == digest
    assert run_episode(tmp_path / "first", config, digest, "intact", 42, "left") == left
