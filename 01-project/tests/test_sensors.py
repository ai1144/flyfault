import numpy as np
from pytest import approx, raises

from flyfault.sensing import GaussianOdorField


def test_sensor_sides_follow_body_rotation():
    field = GaussianOdorField((4.0, 3.0), 3.0)
    xyz = (0.0, 0.0, 1.0)
    forward = field.read(xyz, np.eye(3))
    turned_around = field.read(xyz, np.diag((-1, -1, 1)))
    assert forward[0] > forward[1]
    assert turned_around[0] < turned_around[1]
    mirrored_field = GaussianOdorField((4.0, -3.0), 3.0)
    assert mirrored_field.read(xyz, np.eye(3)) == approx(forward[::-1])
    assert field.concentration((4, 3)) == approx(1.0)


def test_field_rejects_nonphysical_sigma():
    with raises(ValueError, match="sigma"):
        GaussianOdorField((4.0, 3.0), 0)
