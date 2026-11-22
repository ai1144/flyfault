# Methods

## Closed-loop timing

| Process | Step | Input and output |
| --- | --- | --- |
| MuJoCo physics and FlyGym gait | 0.1 ms | Held left and right leg drive → joint position targets and adhesion → physical motion |
| Brian2 LIF integration | 0.5 ms | Held local concentrations → sensory, inhibitory and premotor spikes |
| Sensor and descending update | 20 ms | Pose at the window start → odor sample, followed by a spike-rate readout at the window end |
| Trajectory recorder | 10 ms | Official FlyGym state samples, 25 fps at 0.25× playback |

At decision time `t`, the odor sampler reads the current thorax pose. The neural circuit integrates those two held inputs over `[t, t + 20 ms)`. The physics integrates the same interval using the previously available command. At `t + 20 ms` the new readout becomes available to the physics. The first physics interval uses `[0.82, 0.82]`. The 50 ms physical settling period is excluded from episode time. Replay and neural monitors use the same episode-relative origin.

## Odor and circuit

The planar concentration is `exp(-||p - source||² / (2 σ²))`. Coordinates are in mm. Two virtual antennae sit at thorax-local `(0.9, +0.35, 0.20)` and `(0.9, −0.35, 0.20)` mm. The environment rotates and translates them with the thorax and samples the field at their projected XY positions. The circuit receives two scalar concentrations. The deterministic smooth field omits airflow, turbulence, and receptor kinetics.

The 100 neurons comprise 12 sensory, 20 premotor and 8 inhibitory cells per hemisphere, plus 20 reserve cells. A 10 ms membrane time constant and 100 MΩ resistance integrate drive, synaptic current, and optional stimulation. Membrane rest and reset are −65 mV, threshold is −50 mV, and refractory time is 2 ms. Synaptic current decays with a 5 ms time constant. Sensory neurons excite ipsilateral premotor and inhibitory pools. Inhibitory neurons suppress contralateral premotor cells. The reserve pool weakly excites both premotor pools. Wiring is deterministic and regular, with seeded current variability and initial voltages. The circuit is phenomenological and connectome-inspired.

Spikes per decision interval are normalized by the original population size and window duration. A 60 ms exponential filter smooths the left-minus-right premotor rate. The normalized contrast goes through `tanh(contrast / 18 Hz)` and sets an opponent offset of up to 0.38 around neutral drive. Isolated motor calibration maps stronger right-leg drive to positive yaw and stronger left-leg drive to negative yaw. MuJoCo integrates the gait controller's joint and adhesion actuation.

## Interventions and outcomes

| Arm | Manipulation from 0 s |
| --- | --- |
| Intact | Full circuit with the paired sensory field and seeds |
| Left lesion | Spike generation disabled in a seeded subset of left premotor neurons |
| Right lesion | Homologous manipulation of the right premotor population |
| Random sham | Equal cell count silenced in the reserve pool, which supplies weak bilateral excitation |
| Rescue | Same left lesion as the paired lesion arm. Online contrast-dependent current enters surviving left premotor neurons |
| No neural control | Fixed bilateral leg drive. The circuit is logged for comparison |

Lesions are configured as a fraction of the 20-cell target. The demo uses 50%. The rescue amplitude per spared left premotor neuron is `min(150, 140 × L + 1200 × max(L − R, 0))` pA, where `L` and `R` are the current local concentrations. Stimulation details and exact indices are in each episode manifest. `rescue_current_pa` stores the applied current per target cell, held for each neural window. Zero-current windows are preserved.

Reaching, inversion, and arena exit are evaluated at 20 ms decision boundaries. The signed change in heading uses the unwrapped thorax yaw with positive turns toward world +Y from +X. Distance improvement is initial minus final XY source distance. Path length is the sum of XY intersample displacements. Path efficiency is signed improvement divided by path length. Mean absolute heading error is the average absolute wrapped bearing error at sampled poses. The latency is time to first observed entry into the target radius. Timeouts preserve missing latency. Condition effects pair each treatment to the intact episode with the same seed and stimulus side. Mirrored sides are repeated within seeds. The source is reflected across the horizontal world line through the spawn point. In the shipped protocols the spawn yaw is zero, so positive source Y is the fly’s initial left side. The plots suppress uncertainty when independent seeds are too few.

## Scope

The model tests causal dependencies within a defined simulation and a hand-wired circuit. It cannot establish neuronal identity, synapse-level anatomy, odor plume navigation performance, or correspondence with lesion phenotypes in living flies. A larger study can vary source geometry and sensor noise, blind selection of perturbation targets, preregister power and uncertainty, and then test a separately versioned FlyWire- or FlyVis-derived circuit.
