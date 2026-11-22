import numpy as np
import pytest

from flyfault.network import POPULATION_RANGES, POPULATIONS, Microcircuit
from flyfault.readout import DescendingReadout


@pytest.mark.parametrize("side,other", [("left", "right"), ("right", "left")])
def test_local_odor_evokes_lateralized_spikes_and_commands(side, other):
    circuit = Microcircuit(7, "intact", 0.001, 0.02)
    readout = DescendingReadout(0.02)
    odor = [0.8, 0.1] if side == "left" else [0.1, 0.8]
    windows = []
    commands = []
    for _ in range(10):
        window = circuit.update(odor)
        windows.append(window.rates_hz)
        commands.append(readout.update(window.rates_hz))
    rates = dict(zip(POPULATIONS, np.mean(windows[3:], axis=0), strict=True))
    assert rates[f"sensory_{side}"] > rates[f"sensory_{other}"] + 30
    assert rates[f"premotor_{side}"] > rates[f"premotor_{other}"] + 15
    left_drive, right_drive = np.mean(commands[3:], axis=0)
    assert (right_drive > left_drive) == (side == "left")


def test_rates_match_real_spikes_and_state_monitor_is_sparse():
    circuit = Microcircuit(22, "intact", 0.001, 0.02)
    first = circuit.update([0.5, 0.4])
    second = circuit.update([0.3, 0.5])
    time_s, neuron = circuit.spikes()
    for index, window in enumerate((first, second)):
        in_window = (time_s >= index * 0.02) & (time_s < (index + 1) * 0.02)
        for group, rate in zip(POPULATION_RANGES.values(), window.rates_hz, strict=True):
            count = np.count_nonzero(in_window & (neuron >= group.start) & (neuron < group.stop))
            assert rate * len(group) * 0.02 == pytest.approx(count)
    sample_time, voltage_mv = circuit.state_samples()
    np.testing.assert_allclose(sample_time, [0, 0.02])
    assert voltage_mv.shape == (len(sample_time), len(POPULATIONS))
    assert np.isfinite(voltage_mv).all()


@pytest.mark.parametrize("odor", [[0.2], [0.2, np.nan], [-0.2, 0.2], [0.2, 1.1]])
def test_invalid_odor_does_not_advance_network(odor):
    circuit = Microcircuit(7, "intact", 0.001, 0.02)
    with pytest.raises(ValueError, match="odor concentrations"):
        circuit.update(odor)
    assert circuit.network.t_ == 0
