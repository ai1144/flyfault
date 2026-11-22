import numpy as np
import pytest

from flyfault.network import POPULATIONS
from flyfault.readout import DescendingReadout


def test_readout_smooths_premotor_rate_contrast_and_bounds_commands():
    readout = DescendingReadout(0.02)
    equal_rates = np.full(len(POPULATIONS), 20.0)
    np.testing.assert_array_equal(readout.update(equal_rates), readout.neutral)
    left_rates = equal_rates.copy()
    left_rates[POPULATIONS.index("premotor_left")] = 50
    first = readout.update(left_rates)
    second = readout.update(left_rates)
    assert readout.neutral[1] < first[1] < second[1] < 1.2
    assert 0.44 < second[0] < first[0] < readout.neutral[0]
    left_rates[POPULATIONS.index("premotor_left")] = 500
    saturated = readout.update(left_rates)
    np.testing.assert_allclose(saturated, [0.44, 1.2], rtol=1e-6)
    np.testing.assert_array_equal(readout.update(left_rates, frozen=True), readout.neutral)
    np.testing.assert_array_equal(readout.update(equal_rates), readout.neutral)


def test_nonfinite_rates_are_rejected():
    rates = np.zeros(len(POPULATIONS))
    rates[0] = np.inf
    with pytest.raises(ValueError, match="rate"):
        DescendingReadout(0.02).update(rates)
