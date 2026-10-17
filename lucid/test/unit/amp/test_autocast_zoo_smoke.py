"""Zoo code paths that mix dtypes run under autocast (LCD-500).

The per-device policy table changed what dtype each op hands the next, and
two zoo paths met a combination they had not seen: deformable attention
samples float16 values at float32 locations (``grid_sample`` is WIDEST under
Metal autocast), and ZFNet's local response norm met its own float32 ``sum``
with a float16 operand.  Small configs keep these quick.
"""

import pytest

import lucid
import lucid.models as models
from lucid.amp import autocast
from lucid.models._utils._detection import multi_scale_deformable_attention
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="Metal unavailable")


def test_deformable_attention_with_half_values_and_float32_locations() -> None:
    lucid.manual_seed(0)
    shapes = [(4, 4), (2, 2)]
    value = lucid.randn(1, 20, 2, 8, device="metal").to(lucid.float16)
    locations = lucid.rand(1, 3, 2, 2, 4, 2, device="metal")
    weights = lucid.rand(1, 3, 2, 2, 4, device="metal")
    with autocast(device_type="metal", dtype=lucid.float16):
        out = multi_scale_deformable_attention(value, shapes, locations, weights)
    assert out.shape == (1, 3, 16)
    # grid_sample's row is WIDEST on Metal: float16 values with float32
    # locations sample in float32, as the reference's do.
    assert out.dtype == lucid.float32
    want = multi_scale_deformable_attention(value.float(), shapes, locations, weights)
    assert float((out - want).abs().max().item()) < 1e-2


@pytest.mark.xfail(strict=True, reason="LCD-500 A1: local_response_norm via _autocast_entry")
def test_zfnet_forward_under_metal_autocast() -> None:
    lucid.manual_seed(0)
    model = models.create_model("zfnet_cls", num_classes=10).to("metal").eval()
    x = lucid.randn(1, 3, 64, 64, device="metal")
    with autocast(device_type="metal", dtype=lucid.float16):
        logits = model(x).logits
    assert logits.shape == (1, 10)
    assert bool(lucid.isfinite(logits).all().item())
