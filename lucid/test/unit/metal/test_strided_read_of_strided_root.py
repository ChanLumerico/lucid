"""A strided Metal view reads through a root whose MLX value is itself strided.

A Metal view family keeps one MLX value, its root, and reads a strided
member with ``mlx::core::as_strided`` over the flattened root.  MLX refuses
an ``as_strided`` input that is not row-contiguous, and flattening a strided
root shares its buffer, strides and all, whenever its axes merge without a
copy.  Mobile MQA's ``o.reshape(b, heads, n, d).permute(0, 2, 1, 3)
.reshape(...)`` over an attention output hit that, and
``mobilenet_v4_hybrid_medium`` / ``_large`` failed on Metal with "AsStrided
must be used with row contiguous arrays only" (LCD-17).  The root is now laid
out before the window is taken.
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.models.vision.mobilenet_v4._model import _MultiQueryAttention
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="metal unavailable")


def _np(t: lucid.Tensor) -> np.ndarray:
    return np.asarray(t.to("cpu").tolist(), dtype=np.float32)


def test_permuted_attention_output_reshapes_on_metal() -> None:
    b, heads, n, d = 1, 4, 49, 64
    rng = np.random.default_rng(0)
    q, k, v = (
        rng.standard_normal(s).astype(np.float32)
        for s in ((b, heads * n, d), (b, 16, d), (b, 16, d))
    )

    def run(device: str) -> lucid.Tensor:
        qt, kt, vt = (lucid.tensor(a, device=device) for a in (q, k, v))
        o = F.scaled_dot_product_attention(qt, kt, vt, scale=d**-0.5)
        o = o.reshape(b, heads, n, d).permute(0, 2, 1, 3)
        return o.reshape(b, 7, 7, heads * d).permute(0, 3, 1, 2).contiguous()

    np.testing.assert_allclose(_np(run("metal")), _np(run("cpu")), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("kv_stride", [1, 2])
def test_mobile_mqa_forward_matches_cpu(kv_stride: int) -> None:
    lucid.manual_seed(0)
    attn = _MultiQueryAttention(32, 32, 4, 16, kv_stride).eval()
    x = lucid.randn(1, 32, 8, 8)
    ref = _np(attn(x))
    out = _np(attn.to("metal")(x.to("metal")))
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
