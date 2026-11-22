"""Local planar odor measurements at two thorax-fixed virtual antennae."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

# NeuroMechFly rest axes: +X forward, +Y left, +Z dorsal.
# These points are explicit model approximations in millimeters from the thorax.
ANTENNAL_OFFSETS_MM = ((0.9, 0.35, 0.20), (0.9, -0.35, 0.20))


@dataclass(frozen=True)
class GaussianOdorField:
    """Unit-peak, static, isotropic 2D concentration field."""

    source_xy_mm: tuple[float, float]
    sigma_mm: float

    def __post_init__(self) -> None:
        source = np.asarray(self.source_xy_mm, dtype=float)
        if source.shape != (2,) or not np.isfinite(source).all():
            raise ValueError("source_xy_mm must be a finite XY pair")
        if not np.isfinite(self.sigma_mm) or self.sigma_mm <= 0:
            raise ValueError("sigma_mm must be finite and positive")

    def concentration(self, xy_mm: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the dimensionless field for points ending in an XY axis."""
        delta = np.asarray(xy_mm, dtype=float) - self.source_xy_mm
        squared_distance_mm2 = np.sum(delta**2, axis=-1)
        return np.exp(-squared_distance_mm2 / (2 * self.sigma_mm**2))

    def read(self, xyz_mm: ArrayLike, body_to_world: ArrayLike) -> NDArray[np.float64]:
        """Return concentrations in [left, right] order from the current body pose."""
        center = np.asarray(xyz_mm, dtype=float)
        rotation = np.asarray(body_to_world, dtype=float)
        if center.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError("expected a thorax XYZ and a 3×3 body-to-world rotation")
        points = center + np.asarray(ANTENNAL_OFFSETS_MM) @ rotation.T
        return self.concentration(points[:, :2])
