# Executed local study

The study ran on a MacBook Air M1 (`MacBookAir10,1`, 8 GiB, macOS 26.5) with native ARM64 CPython 3.12.13. The science used FlyGym 2.1.1, MuJoCo 3.16.0, Brian2 2.9.0 and NumPy 2.2.6. NumPy 2.5.4 and 2.4.6 both failed to import Brian2 at `ndarray.ptp`. The lock pins the passing version.

## Calibration and smoke

- The fixed drives `[1.2, 0.4]`, `[0.8, 0.8]`, and `[0.4, 1.2]` produced heading changes of −47.39°, +7.97°, and +53.57°, respectively, over 0.4 s at seed 42. The isolated gait has a small positive bias at equal drive in this trial. The 320×240 prototype decoded 41 real replay frames. [Motor calibration](../examples/calibration/motor_calibration.json)
- In a stand-alone odor clamp, switching concentration contrast from −0.06 to +0.06 switched the mean premotor difference from −11.39 to +10.89 Hz and the mean right-minus-left drive from −0.405 to +0.405. This probe used three seeds. [Input response](../examples/calibration/input_response.png)
- The 0.24 s intact smoke trial produced 1,092 spikes and 2.20 mm net XY displacement. Every logged rate was reconstructed from the spike timestamps. Physics held the previous neural command each window, with neutral drive in the first window. The final-code smoke matched the earlier smoke signals within an absolute tolerance of 10⁻⁹. The complete smoke archive is retained locally.

## Paired demo

The 24 episodes paired all six conditions across seeds 42 and 43 and both source sides. The target radius was 3 mm and the timeout was 2 s. Positive progress means the final thorax position was closer to the source. Each row below summarizes four trials. There were no flips or arena exits.

| Condition | Arrivals | Mean progress (mm) | Mean path length (mm) | Mean signed yaw change (°) |
| --- | ---: | ---: | ---: | ---: |
| Intact | 0/4 | +12.26 | 28.07 | +2.5 |
| Left lesion | 0/4 | −7.16 | 27.52 | −204.1 |
| Right lesion | 0/4 | −7.48 | 27.56 | +208.8 |
| Random sham | 0/4 | +12.25 | 28.06 | +2.8 |
| Rescue attempt | 2/4 | +14.70 | 24.30 | −2.0 |
| Constant descending drive | 0/4 | +9.35 | 28.19 | +3.4 |

Rescue improved progress relative to the paired left lesion by 21.59 mm for seed 42 and 22.14 mm for seed 43, after averaging the two sides within each seed. The arrivals were both on right-source trials at 1.48 and 1.52 s. The lesion arms showed opposite yaw biases despite similar total path lengths. There are two independent seeds, so these are descriptive comparisons without confidence intervals. The replay uses seed 42 with the left source as specified by config order, and all three displayed trials time out. [Batch manifest](../examples/demo/manifest.json) · [Metrics CSV](../examples/demo/summary.csv)

The batch took about 6 min on this machine. Across the 24 episodes, the mean per-episode update times were 51 ms for Brian2 plus readout and 101 ms for the held-command physics interval. That corresponds to roughly 0.13 simulated seconds per wall-clock second for the loop.

## Included artifacts

- [Three-way replay MP4](../examples/demo/media/flyfault_comparison.mp4) · [GIF preview](../media/preview.gif)
- [Interactive perturbation atlas](../examples/demo/figures/perturbation_atlas.html) · [Static atlas](../examples/demo/figures/perturbation_atlas.png)
- [Interactive neural dynamics](../examples/demo/figures/neural_dynamics.html) · [Neural raster](../examples/demo/figures/neural_raster.png)
- [Behavior paths](../examples/demo/figures/behavior_paths.png) · [Perturbation summary](../examples/demo/figures/perturbation_summary.png)

## Verification from the original run

The commands below ran from the project directory. `MPLCONFIGDIR` and `XDG_CACHE_HOME` pointed into `.tooling` during sandboxed checks. Native CGL checks required the logged-in desktop graphics context, and the server smoke used a loopback socket.

| Command | Result |
| --- | --- |
| `UV_OFFLINE=1 ./scripts/bootstrap.sh` | Exit 0 with the installed, locked environment |
| `.venv/bin/ruff check .` | Exit 0 |
| `.venv/bin/pytest -q` | Exit 0, 20 tests passed. Brian2 emitted upstream Pyparsing deprecation warnings |
| `.venv/bin/python -m flyfault doctor` | Exit 0, verified ARM64 Python and NumPy, `mjpython`, bundled assets and a nonuniform 320×240 RGB frame |
| `.venv/bin/python -m flyfault smoke --config configs/smoke.yaml` | Exit 0 |
| `.venv/bin/python -m flyfault run --config configs/demo.yaml --condition intact --seed 42 --stimulus-side right` | Exit 0 |
| `.venv/bin/python -m flyfault render --run-dir results/smoke-intact-42-left-1d13e4196580` | Exit 0 |
| `.venv/bin/python -m flyfault render --run-dir results/demo-intact-42-right-606c5a49eba5` | Exit 0 |
| `.venv/bin/python -m flyfault batch --config configs/demo.yaml` | Exit 0. A second invocation reused the same 24 signal files |
| `.venv/bin/python -m flyfault render --run-dir results/demo-batch-d67381774ab6` | Exit 0, built the figures and comparison from saved runs |
| `.venv/bin/python scripts/verify_study.py results/demo-batch-d67381774ab6` | Exit 0. Audited 24 episodes and 260,333 spikes, reconstructed rates and stimulation, and decoded all 201 frames at 1920×1080, 25 fps and 8.04 s |
| `MPLCONFIGDIR=/tmp/flyfault-neural-mpl XDG_CACHE_HOME=/tmp/flyfault-neural-cache PYTHONPATH=src .venv/bin/python scripts/neural_probe.py --output results/calibration/neural` | Exit 0, 420 recorded windows |
| `.venv/bin/python -m flyfault dashboard --results results --port 8514` | Served locally. A health request returned `ok`. The app also loaded real trial data in Streamlit AppTest |
| `.venv/bin/python -m flyfault dashboard --results results --port 8515` | Final launcher served locally, returned `ok` from `/_stcore/health`, and shut down cleanly |

The [original full-archive audit](../examples/demo/study_audit.json) is included with the example data. Raw verification logs are retained locally because they include machine-specific paths. The full eight-seed benchmark and an interactive passive viewer are outside this verification. For a larger follow-up, vary source geometry and compare predeclared outcomes across independent seeds before testing a FlyWire- or FlyVis-derived circuit.

The only setup downloads were the native `uv` wheel and Python package/build dependencies from PyPI and the `uv`-managed CPython archive from Astral's Python standalone releases. The NeuroMechFly geometry and gait templates came from the FlyGym wheel. The public repository includes the reviewed source and a compact export of the measured pilot. The original machine-specific run archive remains local.

## Publication checks

The publication copy passed all 20 tests, Ruff, and the native offscreen render check again. The dashboard loaded all 24 included episodes, both replay videos, and its three inspection views. Every published signal archive and per-trial summary matched the corresponding original file byte for byte.

Gitleaks and targeted checks inspected the selected files for credentials and personal paths. The review also checked image and video metadata and the numeric contents of the NPZ archives. The original prompt, local environments, caches, credentials, compiled physics models, and raw execution logs are excluded from the repository. Published manifests omit personal filesystem paths and interpreter locations.
