import numpy as np

from flyfault.network import Microcircuit


def test_recreated_network_is_reproducible_with_interleaved_updates():
    first = Microcircuit(13, "rescue", 0.001, 0.02)
    second = Microcircuit(13, "rescue", 0.001, 0.02)
    other = Microcircuit(14, "rescue", 0.001, 0.02)
    for i in range(8):
        odor = [0.3 + 0.01 * i, 0.38]
        expected = first.update(odor)
        other.update(odor)
        actual = second.update(odor)
        np.testing.assert_array_equal(expected.rates_hz, actual.rates_hz)
        assert expected.rescue_current_pa == actual.rescue_current_pa
    assert first.metadata() == second.metadata()
    for expected, actual in zip(first.spikes(), second.spikes(), strict=True):
        np.testing.assert_array_equal(expected, actual)
    for expected, actual in zip(first.state_samples(), second.state_samples(), strict=True):
        np.testing.assert_allclose(expected, actual)
    assert not np.array_equal(first.spikes()[0], other.spikes()[0])
