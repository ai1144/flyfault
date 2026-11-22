# Recorded pilot data

`demo/` contains all 24 episodes from the original paired pilot. Numeric signal archives and summary files are byte-identical to the source experiment. The manifests retain model parameters, seeds, intervention targets, package versions, source hashes, and units. Personal filesystem paths, interpreter locations, shell commands, and execution timestamps were removed from the published manifests.

The dataset includes the original comparison video, recorded paths, spikes, sensory readings, neural rates, stimulation currents, and motor commands. The compiled MuJoCo models, full physics replay trajectories, and raw execution logs stay in the local archive. Rebuild those with `sh scripts/run_demo.sh` followed by the render command.

After setup, inspect the included experiment without running physics.

```bash
.venv/bin/python -m flyfault dashboard --results examples/demo
```

Regenerate publication figures directly from the saved numeric traces.

```bash
.venv/bin/python -m flyfault render --run-dir examples/demo --plots-only
```

The self-contained HTML figures in `demo/figures/` can be downloaded and opened in a browser. GitHub displays the HTML source in its file viewer.

`calibration/` contains measured input-response data and a summary of the independent motor command calibration. `demo/study_audit.json` preserves the audit from the complete local archive. Repeating its full-model audit requires regenerating the omitted replay files.
