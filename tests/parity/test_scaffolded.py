"""Smoke coverage for ops that graduated from scaffolded templates."""
from __future__ import annotations

import numpy as np
import pytest

from lucid._C import engine as E


@pytest.mark.parametrize("device", [E.Device.CPU, E.Device.GPU])
def test_cube_root_forward(device):
    x = np.linspace(-8.0, 8.0, num=20, dtype=np.float32).reshape(4, 5)
    t = E.TensorImpl(x, device, False)
    out = E.cube_root(t)
    got = np.asarray(out.data_as_python()).reshape(out.shape)
    np.testing.assert_allclose(got, np.cbrt(x), rtol=1e-5, atol=1e-5)
