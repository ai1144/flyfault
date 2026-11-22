"""Convert causal premotor firing rates to two bounded FlyGym commands."""

import math

import numpy as np

from .network import POPULATIONS

_LEFT = POPULATIONS.index("premotor_left")
_RIGHT = POPULATIONS.index("premotor_right")


class DescendingReadout:
    def __init__(
        self,
        decision_s: float,
        *,
        neutral_drive: float = 0.82,
        steering_range: float = 0.38,
        smoothing_tau_s: float = 0.06,
        rate_scale_hz: float = 18.0,
    ):
        parameters = (decision_s, neutral_drive, steering_range, smoothing_tau_s, rate_scale_hz)
        if not np.isfinite(parameters).all() or min(parameters) <= 0:
            raise ValueError("readout parameters must be positive and finite")
        if steering_range > neutral_drive or neutral_drive + steering_range > 1.3:
            raise ValueError("commands must stay in [0, 1.3]")
        self.neutral_drive = neutral_drive
        self.steering_range = steering_range
        self.smoothing_tau_s = smoothing_tau_s
        self.rate_scale_hz = rate_scale_hz
        self._alpha = -math.expm1(-decision_s / smoothing_tau_s)
        self._contrast_hz = 0.0

    @property
    def neutral(self) -> np.ndarray:
        return np.full(2, self.neutral_drive)

    def update(self, rates_hz: np.ndarray, *, frozen: bool = False) -> np.ndarray:
        rates = np.asarray(rates_hz, dtype=float)
        if rates.shape != (len(POPULATIONS),) or not np.isfinite(rates).all() or (rates < 0).any():
            raise ValueError("expected one nonnegative finite rate per neural population")
        if frozen:
            self._contrast_hz = 0.0
            return self.neutral
        contrast_hz = rates[_LEFT] - rates[_RIGHT]
        self._contrast_hz += self._alpha * (contrast_hz - self._contrast_hz)
        offset = self.steering_range * math.tanh(self._contrast_hz / self.rate_scale_hz)
        # FlyGym signal order is left legs, right legs. Stronger right drive turns left.
        return np.clip(
            self.neutral + np.array([-offset, offset]),
            self.neutral_drive - self.steering_range,
            self.neutral_drive + self.steering_range,
        )

    def metadata(self) -> dict:
        return {
            "command_order": ["left_leg_drive", "right_leg_drive"],
            "neutral_drive": self.neutral_drive,
            "steering_range": self.steering_range,
            "smoothing_tau_s": self.smoothing_tau_s,
            "rate_scale_hz": self.rate_scale_hz,
            "left_positive_contrast": "decreases left drive and increases right drive",
        }
