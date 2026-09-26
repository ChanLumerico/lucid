"""Two constructor divergences found while porting Self-Forcing.

``nn.Identity`` raised on any argument, so it could not stand in where a
layer is built from arguments (``norm_layer(dim)``).  ``RMSNorm`` defaulted
``eps`` to a fixed ``1e-8``, which rounds to zero in float16: a
half-precision export of an all-zero row divided by an exact zero.  The
default now follows the input's dtype, as the reference framework's does.
"""

import numpy as np

import lucid
import lucid.nn as nn
import lucid.nn.functional as F


def test_identity_accepts_and_ignores_arguments() -> None:
    layer = nn.Identity(64, eps=1e-5, inplace=True)
    x = lucid.randn(2, 64)
    assert layer(x) is x or (layer(x) - x).abs().max().item() == 0.0
    assert list(layer.parameters()) == []


def test_the_default_eps_follows_the_dtype() -> None:
    for dtype in (lucid.float16, lucid.float32):
        x = lucid.zeros(2, 4, dtype=dtype)
        assert F.rms_norm(x, (4,)).to(lucid.float32).tolist() == [[0.0] * 4] * 2
    a = lucid.randn(3, 5)
    arr = a.numpy()
    want = arr / np.sqrt((arr**2).mean(-1, keepdims=True) + np.finfo(np.float32).eps)
    np.testing.assert_allclose(nn.RMSNorm(5)(a).numpy(), want, rtol=1e-6)


def test_an_explicit_eps_is_still_used() -> None:
    a = lucid.randn(3, 5)
    arr = a.numpy()
    want = arr / np.sqrt((arr**2).mean(-1, keepdims=True) + 0.5)
    np.testing.assert_allclose(nn.RMSNorm(5, eps=0.5)(a).numpy(), want, rtol=1e-6)
