# Reproducibility

Each `manifest.json` captures the Python build, full package versions, platform, executable path, command, original and effective configuration SHA-256, scientific source fingerprint, timing, units, and execution status. Episode manifests also record intervention identities, CPG model and bundled mesh hashes, benchmark timing, and artifact paths. `model.mjb` stores the exact model used to record each `trajectory.npz`; replay checks its hash and compiled state size. Retain the same MuJoCo version when reopening binary models. Portable `signals.npz` and `summary.json` support analytics without the binary model.

The seed controls gait phase initialization, neural initialization and noise, and the intervention mask through separate random streams. Matching seed and source side across arms produces paired starting conditions. The left lesion and rescue have identical silenced cells. Replicates span distinct seeds. Numeric matching is checked on this machine, while solver and floating point differences can affect trajectories across platforms.

An identical batch command revisits the content-addressed directory and reuses finished episodes. Partial or failed attempts are retained in each episode's `previous_attempts` before retrying. Preserve the original config, lockfile and scientific sources to reuse an existing ID. A changed scientific module or package version creates a new study directory.

The simplified NeuroMechFly meshes and preprogrammed locomotion steps come from the FlyGym wheel. The setup requires network access for Python and package wheels on a fresh machine. The executed model uses the bundled assets. A different composition that requests full-size assets may need first-run downloads from the simulator's asset host.

The dashboard serves persisted runs on the loopback interface. Replay and figures read completed run data. Native offscreen CGL requires a usable connection to the macOS graphics session. Run the render from the logged-in desktop, and use `--plots-only` for data-only plotting in a restricted terminal. MuJoCo's passive interactive viewer on macOS requires `.venv/bin/mjpython` for applications using `mujoco.viewer.launch_passive`. FLY//FAULT provides post-hoc replay as its visual entry point.

## Published example

The included `examples/demo` export removes personal paths, interpreter locations, shell commands, and execution timestamps from manifests. Numeric signals and trial summaries match the original run files exactly. Compiled physics models, replay trajectories, and raw logs remain in the local archive. Use the included sample for the dashboard and `render --plots-only`. Run a new local protocol to produce a complete physics archive for replay and `scripts/verify_study.py`.
