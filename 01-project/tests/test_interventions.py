import numpy as np
from brian2 import pA

from flyfault.network import POPULATION_RANGES, POPULATIONS, Microcircuit


def test_matched_lesions_silence_real_spikes_and_rescue_targets_only_survivors():
    circuits = {
        condition: Microcircuit(9, condition, 0.001, 0.02)
        for condition in ("lesion_left", "lesion_right", "sham_random", "rescue")
    }
    left = circuits["lesion_left"].intervention
    right = circuits["lesion_right"].intervention
    sham = circuits["sham_random"].intervention
    rescue = circuits["rescue"].intervention
    assert left.silenced == rescue.silenced
    assert (
        tuple(i + len(POPULATION_RANGES["premotor_left"]) for i in left.silenced) == right.silenced
    )
    assert len(left.silenced) == len(sham.silenced) == 10
    assert set(sham.silenced).issubset(POPULATION_RANGES["reserve"])
    assert set(rescue.stimulated) == set(POPULATION_RANGES["premotor_left"]) - set(rescue.silenced)

    recent_rates = {}
    for condition, circuit in circuits.items():
        windows = [circuit.update([0.6, 0.5]) for _ in range(8)]
        recent_rates[condition] = np.mean([window.rates_hz for window in windows[3:]], axis=0)
        _, neurons = circuit.spikes()
        assert not np.isin(neurons, circuit.intervention.silenced).any()
        enabled = sorted(
            set(POPULATION_RANGES[circuit.intervention.affected_population])
            - set(circuit.intervention.silenced)
        )
        assert np.isin(neurons, enabled).any()
        applied_pa = np.asarray(circuit.neurons.I_rescue[:] / pA)
        assert np.flatnonzero(applied_pa).tolist() == list(circuit.intervention.stimulated)
        np.testing.assert_allclose(applied_pa[applied_pa > 0], windows[-1].rescue_current_pa)
    premotor_left = POPULATIONS.index("premotor_left")
    assert recent_rates["rescue"][premotor_left] > recent_rates["lesion_left"][premotor_left]


def test_full_lesion_records_empty_rescue_target_and_zero_current():
    circuit = Microcircuit(9, "rescue", 0.001, 0.02, lesion_fraction=1.0)
    result = circuit.update([1, 0])
    assert result.rescue_current_pa == 0
    assert circuit.metadata()["intervention"]["stimulation"]["target_neuron_indices"] == []
