"""Model-zoo forward parity across devices.

Found 2026-07-26 by sweeping every vision family on Metal: ``crossvit`` raised
``bad_variant_access``.  Its bicubic-resize helper built the sampling
coordinates with ``lucid.arange(...)`` and no ``device=``, then used them to
index the (Metal) feature map — the same CPU-index-into-GPU-data pattern as the
``pdist`` and transforms bugs.

After the fix the sweep reports **0 families that work on the CPU but fail on
Metal**.  These tests keep a representative slice of that sweep in CI, cheaply:
config overrides shrink the models so the check costs a forward pass, not a
real network.
"""

import numpy as np
import pytest

import lucid
import lucid.models as M

DEVICES = ["cpu", "metal"]


def _output_tensor(out):
    if hasattr(out, "shape"):
        return out
    for attr in ("logits", "last_hidden_state"):
        if hasattr(out, attr):
            return getattr(out, attr)
    raise AssertionError(f"no tensor in model output: {type(out)!r}")


def _forward(name, device, size=224, seed=0, **overrides):
    lucid.manual_seed(0)
    model = M.create_model(name, **overrides).to(device).eval()
    data = np.random.default_rng(seed).standard_normal((1, 3, size, size))
    x = lucid.tensor(data.astype(np.float32), device=device)
    with lucid.no_grad():
        return _output_tensor(model(x)).numpy()


@pytest.mark.parametrize("device", DEVICES)
def test_crossvit_runs_on_device(device):
    """The regression: bicubic-resize coords were CPU-only."""
    out = _forward("crossvit_15", device)
    assert out.ndim == 2
    assert not np.isnan(out).any()


def test_crossvit_matches_across_devices():
    cpu = _forward("crossvit_15", "cpu")
    metal = _forward("crossvit_15", "metal")
    assert cpu.shape == metal.shape
    assert np.abs(cpu - metal).max() < 1e-4


# A slice across architecture styles: plain CNN, residual, depthwise,
# windowed attention, plain ViT.  Kept small so the sweep is cheap.
_REPRESENTATIVE = [
    ("resnet_18", 224),
    ("mobilenet_v2", 224),
    ("convnext_tiny", 224),
    ("swin_tiny", 224),
    ("vit_base_16", 224),
]


@pytest.mark.parametrize("name,size", _REPRESENTATIVE)
def test_representative_families_match_across_devices(name, size):
    cpu = _forward(name, "cpu", size=size)
    metal = _forward(name, "metal", size=size)
    assert cpu.shape == metal.shape, name
    assert not np.isnan(metal).any(), name
    assert np.abs(cpu - metal).max() < 1e-3, name


@pytest.mark.parametrize("name,size", _REPRESENTATIVE)
@pytest.mark.parametrize("device", DEVICES)
def test_representative_families_train_one_step(name, size, device):
    """Forward + backward + step — the gradient path must stay on-device too."""
    lucid.manual_seed(0)
    model = M.create_model(name).to(device)
    model.train()
    optimizer = lucid.optim.SGD(model.parameters(), lr=1e-4)
    data = np.random.default_rng(1).standard_normal((1, 3, size, size))
    x = lucid.tensor(data.astype(np.float32), device=device)
    out = _output_tensor(model(x))
    loss = (out * out).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, f"{name}: no parameter received a gradient"
    for g in grads:
        assert str(g.device) == f"device('{device}')", name
