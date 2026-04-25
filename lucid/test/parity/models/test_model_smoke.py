import numpy as np

import pytest

import lucid

import lucid.models as models

pytestmark = pytest.mark.smoke


def _img_input(C=3, H=32, W=32, batch=1):
    return lucid.tensor(
        np.random.default_rng(0).standard_normal((batch, C, H, W)).astype(np.float64)
    )


def _check_forward(model, x, *, expected_ndim: int = 2):
    out = model(x)
    arr = np.asarray(out.data if hasattr(out, "data") else out)
    assert (
        arr.ndim == expected_ndim
    ), f"output ndim {arr.ndim} != expected {expected_ndim}"
    assert np.isfinite(arr).all(), "output contains NaN/Inf"
    assert arr.shape[0] == x.shape[0], "batch dim changed"


def test_smoke_alexnet():
    m = models.alexnet(num_classes=10)
    _check_forward(m, _img_input(H=224, W=224))


def test_smoke_resnet18():
    m = models.resnet_18(num_classes=10)
    _check_forward(m, _img_input(H=64, W=64))


def test_smoke_resnet50():
    m = models.resnet_50(num_classes=10)
    _check_forward(m, _img_input(H=64, W=64))


def test_smoke_convnext_tiny():
    m = models.convnext_tiny(num_classes=10)
    _check_forward(m, _img_input(H=64, W=64))


def test_smoke_vit_tiny():
    if not hasattr(models, "vit_tiny"):
        pytest.skip("vit_tiny not available")
    m = models.vit_tiny(num_classes=10, image_size=64)
    _check_forward(m, _img_input(H=64, W=64))


def test_smoke_unet():
    factory = None
    for cand in ("unet_base", "build_unet", "UNet"):
        f = getattr(models, cand, None)
        if callable(f) and (not isinstance(f, type(models))):
            factory = f
            break
    if factory is None:
        pytest.skip("UNet factory not exposed at top-level lucid.models")
    m = factory(in_channels=3, num_classes=2)
    out = m(_img_input(H=64, W=64))
    arr = np.asarray(out.data if hasattr(out, "data") else out)
    assert np.isfinite(arr).all()
    assert arr.shape[0] == 1


def test_smoke_mobilenet_v2():
    if not hasattr(models, "mobilenet_v2"):
        pytest.skip("mobilenet_v2 not available")
    m = models.mobilenet_v2(num_classes=10)
    _check_forward(m, _img_input(H=64, W=64))


def test_smoke_efficientnet_b0():
    if not hasattr(models, "efficientnet_b0"):
        pytest.skip("efficientnet_b0 not available")
    m = models.efficientnet_b0(num_classes=10)
    _check_forward(m, _img_input(H=64, W=64))
