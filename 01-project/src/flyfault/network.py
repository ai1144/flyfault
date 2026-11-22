"""Explicit Brian2 microcircuit with local odor current as its only input."""

from dataclasses import dataclass
from typing import Final

import numpy as np
from brian2 import (
    Clock,
    Network,
    NeuronGroup,
    SpikeMonitor,
    StateMonitor,
    Synapses,
    ms,
    mV,
    pA,
    second,
)
from brian2.codegen.runtime.numpy_rt import NumpyCodeObject

from .config import grid_steps
from .interventions import select_intervention

POPULATION_RANGES: Final = {
    "sensory_left": range(0, 12),
    "sensory_right": range(12, 24),
    "premotor_left": range(24, 44),
    "premotor_right": range(44, 64),
    "inhibitory_left": range(64, 72),
    "inhibitory_right": range(72, 80),
    "reserve": range(80, 100),
}
POPULATIONS: Final = tuple(POPULATION_RANGES)
NEURON_COUNT: Final = POPULATION_RANGES["reserve"].stop
STATE_NEURON_INDICES: Final = tuple(group.start for group in POPULATION_RANGES.values())
_POPULATION_SIZES = np.array([len(group) for group in POPULATION_RANGES.values()])
_SENSORY_GAIN_PA: Final = 240.0
_NOISE_SD_PA: Final = 6.0
_BIAS_PA: Final = (115.0, 115.0, 151.0, 151.0, 141.0, 141.0, 175.0)


@dataclass(frozen=True)
class Projection:
    source: str
    target: str
    fan_in: int
    weight_pa: float

    def metadata(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "kind": "excitatory" if self.weight_pa > 0 else "inhibitory",
            "synapses": self.fan_in * len(POPULATION_RANGES[self.target]),
            "weight_pa": self.weight_pa,
        }


PROJECTIONS: Final = (
    Projection("sensory_left", "premotor_left", 6, 21.0),
    Projection("sensory_right", "premotor_right", 6, 21.0),
    Projection("sensory_left", "inhibitory_left", 6, 24.0),
    Projection("sensory_right", "inhibitory_right", 6, 24.0),
    Projection("inhibitory_left", "premotor_right", 4, -18.0),
    Projection("inhibitory_right", "premotor_left", 4, -18.0),
    Projection("reserve", "premotor_left", 4, 2.0),
    Projection("reserve", "premotor_right", 4, 2.0),
)

_EQUATIONS = """
dv/dt = ((-65*mV - v) + (100*Mohm)*(I_drive + I_syn + I_rescue)) / (10*ms) : volt (unless refractory)
dI_syn/dt = -I_syn / (5*ms) : amp
I_drive : amp
I_rescue : amp
enabled : boolean (constant)
"""


@dataclass(frozen=True)
class NeuralStep:
    rates_hz: np.ndarray
    rescue_current_pa: float


class Microcircuit:
    """A 100-neuron, phenomenological LIF network recreated per episode."""

    def __init__(
        self,
        seed: int,
        condition: str,
        dt_s: float,
        decision_s: float,
        lesion_fraction: float = 0.5,
    ):
        if not np.isfinite([dt_s, decision_s]).all() or not 0 < dt_s <= decision_s:
            raise ValueError("expected 0 < neural dt <= decision period")
        grid_steps(decision_s, dt_s, "decision_s")

        initial_rng, intervention_rng, self._noise_rng = (
            np.random.default_rng(stream) for stream in np.random.SeedSequence(seed).spawn(3)
        )
        self.dt_s = dt_s
        self.decision_s = decision_s
        self.intervention = select_intervention(
            condition, lesion_fraction, POPULATION_RANGES, intervention_rng
        )
        self._bias_pa = np.repeat(_BIAS_PA, _POPULATION_SIZES)
        self._bias_pa = self._bias_pa + self._bilateral_values(initial_rng, -10.0, 10.0)
        self._counts = np.zeros(NEURON_COUNT, dtype=np.int64)

        clock = Clock(dt=dt_s * second)
        self.neurons = NeuronGroup(
            NEURON_COUNT,
            _EQUATIONS,
            threshold="enabled and v > -50*mV",
            reset="v = -65*mV",
            refractory=2 * ms,
            method="exact",
            clock=clock,
            codeobj_class=NumpyCodeObject,
        )
        self.neurons.v = self._bilateral_values(initial_rng, -65.0, -56.0) * mV
        self.neurons.enabled = True
        if self.intervention.silenced:
            self.neurons.enabled[list(self.intervention.silenced)] = False

        synapses = Synapses(
            self.neurons,
            self.neurons,
            model="w : amp (constant)",
            on_pre="I_syn_post += w",
            clock=clock,
            codeobj_class=NumpyCodeObject,
        )
        # Regular, homologous wiring isolates seeded current variability from side bias.
        weights = []
        sources = []
        targets = []
        for projection in PROJECTIONS:
            source = POPULATION_RANGES[projection.source]
            target = POPULATION_RANGES[projection.target]
            local_target = np.repeat(np.arange(len(target)), projection.fan_in)
            offsets = np.tile(np.arange(projection.fan_in), len(target))
            sources.extend(source.start + (local_target + offsets) % len(source))
            targets.extend(target.start + local_target)
            weights.extend([projection.weight_pa] * local_target.size)
        synapses.connect(i=np.array(sources), j=np.array(targets))
        synapses.w = np.array(weights) * pA

        self.spike_monitor = SpikeMonitor(self.neurons, codeobj_class=NumpyCodeObject)
        self.state_monitor = StateMonitor(
            self.neurons,
            "v",
            record=STATE_NEURON_INDICES,
            dt=decision_s * second,
            codeobj_class=NumpyCodeObject,
        )
        self.network = Network(self.neurons, synapses, self.spike_monitor, self.state_monitor)

    @staticmethod
    def _bilateral_values(rng: np.random.Generator, low: float, high: float) -> np.ndarray:
        draws = [rng.uniform(low, high, int(size)) for size in _POPULATION_SIZES[[0, 2, 4, 6]]]
        return np.concatenate(
            (draws[0], draws[0], draws[1], draws[1], draws[2], draws[2], draws[3])
        )

    def update(self, odor: np.ndarray) -> NeuralStep:
        """Advance one window from its start odor and read only spikes from that window."""
        odor = np.asarray(odor, dtype=float)
        if odor.shape != (2,) or not np.isfinite(odor).all() or np.any((odor < 0) | (odor > 1)):
            raise ValueError("expected two finite odor concentrations in [0, 1]")
        drive = self._bias_pa + self._noise_rng.normal(0, _NOISE_SD_PA, NEURON_COUNT)
        for side, concentration in zip(("sensory_left", "sensory_right"), odor, strict=True):
            drive[POPULATION_RANGES[side]] += _SENSORY_GAIN_PA * concentration
        rescue_pa = self.intervention.rescue_current_pa(odor)
        self.neurons.I_drive = drive * pA
        if self.intervention.stimulated:
            self.neurons.I_rescue[list(self.intervention.stimulated)] = rescue_pa * pA

        self.network.run(self.decision_s * second, namespace={})
        counts = self.spike_monitor.count[:].astype(np.int64)
        population_counts = np.add.reduceat(counts - self._counts, STATE_NEURON_INDICES)
        self._counts = counts
        # Divide by original population size so silencing cannot be hidden by normalization.
        rates_hz = population_counts / (_POPULATION_SIZES * self.decision_s)
        return NeuralStep(rates_hz, rescue_pa)

    def spikes(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.spike_monitor.t[:] / second), np.asarray(self.spike_monitor.i[:])

    def state_samples(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.state_monitor.t[:] / second), np.asarray(self.state_monitor.v / mV).T

    def metadata(self) -> dict:
        return {
            "model": "phenomenological, connectome-inspired leaky integrate-and-fire microcircuit",
            "backend": "Brian2 NumPy runtime",
            "populations": [
                {"name": name, "start": group.start, "stop": group.stop}
                for name, group in POPULATION_RANGES.items()
            ],
            "edges": [projection.metadata() for projection in PROJECTIONS],
            "state_neuron_indices": list(STATE_NEURON_INDICES),
            "dt_s": self.dt_s,
            "decision_s": self.decision_s,
            "parameters": {
                "rest_mv": -65.0,
                "threshold_mv": -50.0,
                "reset_mv": -65.0,
                "membrane_tau_ms": 10.0,
                "membrane_resistance_mohm": 100.0,
                "synaptic_tau_ms": 5.0,
                "refractory_ms": 2.0,
                "bias_current_pa_by_population": dict(zip(POPULATIONS, _BIAS_PA, strict=True)),
                "sensory_gain_pa_per_concentration": _SENSORY_GAIN_PA,
                "paired_uniform_bias_jitter_half_width_pa": 10.0,
                "per_neuron_per_window_current_noise_sd_pa": _NOISE_SD_PA,
            },
            "intervention": self.intervention.metadata(),
        }
