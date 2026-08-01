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


# ─────────────────────────────────────────────────────────────────────────────
# Continuous flows
#
# The sweep above drives every model through one ``model(x)`` call, which the
# flow families do not fit: Flow Matching's forward takes ``(sample, t)``, and
# what actually needs checking is not a forward pass but a *solve* — an ODE
# integrated on-device, with a divergence taken by autograd inside it.  That is
# the pattern most likely to leak a CPU tensor into GPU data, so it gets its
# own checks rather than being left out of the sweep.
# ─────────────────────────────────────────────────────────────────────────────


def _paired(name, **overrides):
    """The same model on both devices, weight-for-weight."""
    lucid.manual_seed(0)
    cpu = M.create_model(name, **overrides).eval()
    metal = M.create_model(name, **overrides).eval()
    metal.load_state_dict(cpu.state_dict())
    return cpu, metal.to("metal")


def _agree(cpu_out, metal_out, tol, what):
    # Guard the instrument: without this, a model that quietly stayed on
    # the CPU would make every comparison below trivially true, and the
    # check would keep passing while testing nothing.
    assert str(cpu_out.device) == "device('cpu')", what
    assert str(metal_out.device) == "device('metal')", what
    a = cpu_out.numpy()
    b = metal_out.to("cpu").numpy()
    assert a.shape == b.shape, what
    assert not np.isnan(b).any(), what
    scale = max(float(np.abs(a).max()), 1e-8)
    assert (
        np.abs(a - b).max() / scale < tol
    ), f"{what}: rel {np.abs(a - b).max() / scale}"


def test_neural_ode_matches_across_devices():
    """Encode, decode and log-density must agree wherever they are solved."""
    cpu, metal = _paired(
        "neural_ode",
        sample_size=(1, 2),
        in_channels=1,
        out_channels=1,
        hidden_dim=16,
        rtol=1e-6,
        atol=1e-8,
    )
    x = lucid.rand((4, 1, 1, 2))
    x_gpu = x.to("metal")

    z_cpu, det_cpu = cpu.encode(x)
    z_metal, det_metal = metal.encode(x_gpu)
    _agree(z_cpu, z_metal, 1e-4, "neural_ode encode")
    _agree(det_cpu, det_metal, 1e-4, "neural_ode log_det")
    _agree(cpu.decode(z_cpu), metal.decode(z_metal), 1e-4, "neural_ode decode")
    # Exact trace at two dimensions, so this one is deterministic.
    _agree(cpu.log_prob(x), metal.log_prob(x_gpu), 1e-4, "neural_ode log_prob")


def test_flow_matching_matches_across_devices():
    """Field, path, target and a fixed-noise solve.

    ``log_prob`` is left out on purpose: above two dimensions the trace is
    a Hutchinson estimate, and the two devices draw their own probes — the
    disagreement would be Monte-Carlo variance, not a device bug.
    """
    cpu, metal = _paired(
        "flow_matching_cifar",
        sample_size=8,
        base_channels=16,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(4,),
        resnet_groups=8,
    )
    x1 = lucid.randn((2, 3, 8, 8))
    x0 = lucid.randn((2, 3, 8, 8))
    t = lucid.tensor([0.3, 0.7])
    g1, g0, gt = x1.to("metal"), x0.to("metal"), t.to("metal")

    _agree(
        cpu(x1, lucid.tensor(0.4)).sample,
        metal(g1, lucid.tensor(0.4, device="metal")).sample,
        1e-4,
        "flow_matching velocity",
    )
    _agree(
        cpu.path_sample(x1, x0, t), metal.path_sample(g1, g0, gt), 1e-5, "path_sample"
    )
    _agree(
        cpu.conditional_target(x1, x0, t),
        metal.conditional_target(g1, g0, gt),
        1e-5,
        "conditional_target",
    )
    _agree(
        cpu.sample(noise=x0, steps=8),
        metal.sample(noise=g0, steps=8),
        1e-4,
        "flow_matching sample",
    )


@pytest.mark.parametrize("device", DEVICES)
def test_flow_matching_trains_one_step_on_device(device):
    """The objective is simulation-free, so no solve should appear here."""
    lucid.manual_seed(0)
    model = M.create_model(
        "flow_matching_cifar_gen",
        sample_size=8,
        base_channels=16,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(4,),
        resnet_groups=8,
    ).to(device)
    model.train()
    optimizer = lucid.optim.SGD(model.parameters(), lr=1e-4)
    out = model(lucid.randn((2, 3, 8, 8), device=device))
    optimizer.zero_grad()
    out.loss.backward()
    optimizer.step()

    assert model.nfe == 0
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "flow_matching: no parameter received a gradient"
    for g in grads:
        assert str(g.device) == f"device('{device}')"


def test_rectified_flow_matches_across_devices():
    """Path, target, a fixed-noise solve, and the straightness measure.

    ``straightness`` is included because it is the only quantity here that
    both solves *and* reduces over the whole batch, so a stray CPU tensor
    inside the loop would show up as a disagreement rather than as a
    crash.
    """
    cpu, metal = _paired(
        "rectified_flow_cifar",
        sample_size=8,
        base_channels=16,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(),
        resnet_groups=8,
        init_scale=1.0,
    )
    x1 = lucid.randn((2, 3, 8, 8))
    x0 = lucid.randn((2, 3, 8, 8))
    t = lucid.tensor([0.3, 0.7])
    g1, g0, gt = x1.to("metal"), x0.to("metal"), t.to("metal")

    _agree(cpu.path_sample(x1, x0, t), metal.path_sample(g1, g0, gt), 1e-6, "path")
    _agree(
        cpu.conditional_target(x1, x0, t),
        metal.conditional_target(g1, g0, gt),
        1e-6,
        "target",
    )
    _agree(
        cpu.sample(noise=x0, steps=8), metal.sample(noise=g0, steps=8), 1e-4, "sample"
    )
    _agree(cpu.one_step(x0), metal.one_step(g0), 1e-4, "one_step")

    s_cpu = float(cpu.straightness(x0, steps=8))
    s_metal = float(metal.straightness(g0, steps=8))
    assert abs(s_cpu - s_metal) / max(abs(s_cpu), 1e-8) < 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_rectified_flow_reflow_round_trips_on_device(device):
    """Generate couplings, then train on them — both halves must stay put.

    This is the one training path in the flow families that contains a
    solve, so it is where a device leak would be easiest to miss.
    """
    lucid.manual_seed(0)
    model = M.create_model(
        "rectified_flow_cifar_gen",
        sample_size=8,
        base_channels=16,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(),
        resnet_groups=8,
        init_scale=1.0,
    ).to(device)

    model.eval()
    z0, z1 = model.reflow_pairs(n_samples=2, device=device, steps=4)
    assert str(z0.device) == f"device('{device}')"
    assert str(z1.device) == f"device('{device}')"

    model.train()
    optimizer = lucid.optim.SGD(model.parameters(), lr=1e-4)
    out = model(z1, noise=z0)
    optimizer.zero_grad()
    out.loss.backward()
    optimizer.step()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "rectified_flow: no parameter received a gradient"
    for g in grads:
        assert str(g.device) == f"device('{device}')"


@pytest.mark.parametrize("device", DEVICES)
def test_rectified_flow_trains_one_step_on_device(device):
    """The objective is simulation-free, so no solve should appear here."""
    lucid.manual_seed(0)
    model = M.create_model(
        "rectified_flow_cifar_gen",
        sample_size=8,
        base_channels=16,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(),
        resnet_groups=8,
    ).to(device)
    model.train()
    optimizer = lucid.optim.SGD(model.parameters(), lr=1e-4)
    out = model(lucid.randn((2, 3, 8, 8), device=device))
    optimizer.zero_grad()
    out.loss.backward()
    optimizer.step()

    assert model.nfe == 0
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "rectified_flow: no parameter received a gradient"
    for g in grads:
        assert str(g.device) == f"device('{device}')"


def test_rectified_flow_high_resolution_path_matches_across_devices():
    """Filtered resampling and both pyramids — buffers must travel with ``.to()``."""
    cpu, metal = _paired(
        "rectified_flow_afhq_cat",
        sample_size=16,
        base_channels=8,
        channel_mult=(1, 2, 2),
        num_res_blocks=1,
        attention_resolutions=(4,),
        init_scale=1.0,
    )
    x = lucid.randn((2, 3, 16, 16))
    t = lucid.tensor(0.4)
    _agree(
        cpu(x, t).sample,
        metal(x.to("metal"), t.to("metal")).sample,
        1e-3,
        "fir + pyramid field",
    )


@pytest.mark.parametrize("device", DEVICES)
def test_neural_ode_trains_one_step_on_device(device):
    """Here a solve *is* the forward pass, and its gradients must stay put."""
    lucid.manual_seed(0)
    model = M.create_model(
        "neural_ode_gen",
        sample_size=(1, 2),
        in_channels=1,
        out_channels=1,
        hidden_dim=16,
        rtol=1e-4,
        atol=1e-4,
    ).to(device)
    model.train()
    optimizer = lucid.optim.SGD(model.parameters(), lr=1e-4)
    out = model(lucid.rand((2, 1, 1, 2), device=device))
    optimizer.zero_grad()
    out.loss.backward()
    optimizer.step()

    assert model.nfe > 0
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "neural_ode: no parameter received a gradient"
    for g in grads:
        assert str(g.device) == f"device('{device}')"
