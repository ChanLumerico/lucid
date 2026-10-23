"""LCD-411: the denoising families keep the zoo's output contract end to end.

DiT and MeanFlow's networks once returned a bare tensor, and Stable
Diffusion's task model once aliased ``generate`` and returned pixels — so
a caller following "every forward returns a ``ModelOutput``" got a tensor,
and a training loop written against the image-generation task got images
and no loss.  Each family is driven here through its public factory, at a
size a unit test can afford (config overrides, never an invented factory):
one optimizer step that has to move the weights, and generation on every
device.  The last section counts host syncs per sampling loop on Metal,
which must not grow with the number of steps.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.models as M
from lucid._C import engine as _C_engine
from lucid.models.generative.ddpm import DDPMConfig, DDPMForImageGeneration
from lucid.test._fixtures.devices import metal_available

DEVICES = ["cpu", "metal"] if metal_available() else ["cpu"]

_TRANSFORMER = dict(sample_size=8, hidden_size=32, depth=1, num_heads=4, num_classes=4)
_SD = dict(
    sample_size=32,
    vae_block_out_channels=(8, 8),
    vae_layers_per_block=1,
    unet_block_out_channels=(8, 16),
    unet_layers_per_block=1,
    norm_num_groups=2,
    cross_attention_dim=16,
    attention_head_dim=4,
    context_length=8,
    downsample_factor=2,
)


def _build(name: str, device: str) -> lucid.nn.Module:
    overrides = _SD if name.startswith("stable_diffusion") else _TRANSFORMER
    return M.create_model(name, **overrides).to(device)


def _labels(device: str) -> lucid.Tensor:
    return lucid.tensor([0, 1], dtype=lucid.int64).to(device)


# ── the denoising networks return the shared output ────────────────────────


@pytest.mark.parametrize("device", DEVICES)
def test_dit_network_returns_a_diffusion_output(device: str) -> None:
    model = _build("dit_small_2", device).eval()
    out = model(
        lucid.randn((2, 4, 8, 8), device=device),
        lucid.tensor([10.0, 500.0]).to(device),
        _labels(device),
    )
    assert isinstance(out, M.DiffusionModelOutput)
    assert out.sample.shape == (2, 8, 8, 8) and out.loss is None
    assert out[0] is out.sample


@pytest.mark.parametrize("device", DEVICES)
def test_mean_flow_network_returns_a_diffusion_output(device: str) -> None:
    model = _build("mean_flow_base_2", device).eval()
    out = model(
        lucid.randn((2, 4, 8, 8), device=device),
        lucid.tensor([0.0, 0.2]).to(device),
        lucid.tensor([1.0, 0.7]).to(device),
        _labels(device),
    )
    assert isinstance(out, M.DiffusionModelOutput)
    assert out.sample.shape == (2, 4, 8, 8) and out.loss is None


@pytest.mark.parametrize("device", DEVICES)
def test_tvm_network_returns_a_diffusion_output(device: str) -> None:
    model = _build("tvm_imagenet256", device).eval()
    out = model(
        lucid.randn((2, 4, 8, 8), device=device),
        lucid.tensor([1.0, 0.7]).to(device),
        lucid.tensor([0.0, 0.2]).to(device),
        _labels(device),
    )
    assert isinstance(out, M.DiffusionModelOutput)
    assert out.sample.shape == (2, 4, 8, 8) and out.loss is None


# ── one training step from the public factory ──────────────────────────────


def _dit_step(model: lucid.nn.Module, device: str) -> M.ModelOutput:
    return model(lucid.randn((2, 4, 8, 8), device=device), _labels(device))


def _sd_step(model: lucid.nn.Module, device: str) -> M.ModelOutput:
    return model(
        lucid.randn((2, 3, 32, 32), device=device),
        lucid.randn((2, 8, 16), device=device),
    )


_TRAIN: dict[str, Callable[[lucid.nn.Module, str], M.ModelOutput]] = {
    "dit_small_2_gen": _dit_step,
    "mean_flow_base_2_gen": _dit_step,
    "stable_diffusion_gen": _sd_step,
    "tvm_imagenet256_gen": _dit_step,
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(_TRAIN))
def test_a_training_step_moves_the_weights(name: str, device: str) -> None:
    lucid.manual_seed(0)
    model = _build(name, device).train()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = lucid.optim.SGD(model.parameters(), lr=0.1)

    out = _TRAIN[name](model, device)
    assert isinstance(out, M.ModelOutput)
    loss = out["loss"]
    assert loss.shape == () and bool(loss.isfinite().item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    moved = [
        n
        for n, p in model.named_parameters()
        if float((p.detach() - before[n]).abs().max().item()) > 0.0
    ]
    assert moved, f"{name}: no parameter changed after an optimizer step"


# ── generation from the public factory ─────────────────────────────────────


def _generate(name: str, model: lucid.nn.Module, device: str) -> M.GenerationOutput:
    if name == "stable_diffusion_gen":
        context = lucid.randn((2, 8, 16), device=device)
        return model.generate(context, context, num_inference_steps=2)
    if name == "dit_small_2_gen":
        return model.generate(2, labels=_labels(device), steps=2)
    return model.generate(2, labels=_labels(device))


_SAMPLE_SHAPES = {
    "dit_small_2_gen": (2, 4, 8, 8),
    "mean_flow_base_2_gen": (2, 4, 8, 8),
    "stable_diffusion_gen": (2, 3, 32, 32),
    "tvm_imagenet256_gen": (2, 4, 8, 8),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(_SAMPLE_SHAPES))
def test_generation_returns_finite_samples_on_the_device(
    name: str, device: str
) -> None:
    lucid.manual_seed(0)
    model = _build(name, device).eval()
    out = _generate(name, model, device)
    assert isinstance(out, M.GenerationOutput)
    assert out.samples.shape == _SAMPLE_SHAPES[name]
    assert str(out.samples.device) == f"device('{device}')"
    assert bool(out.samples.isfinite().all().item())


# ── host syncs per sampling loop (Metal) ───────────────────────────────────


def _syncs(run: Callable[[], M.GenerationOutput]) -> int:
    run()  # first call compiles kernels and fills caches
    before = _C_engine.host_sync_count()
    run().samples.eval()
    return _C_engine.host_sync_count() - before


def _ddpm() -> DDPMForImageGeneration:
    config = DDPMConfig(
        sample_size=8,
        base_channels=8,
        channel_mult=(1,),
        num_res_blocks=1,
        resnet_groups=4,
        num_train_timesteps=20,
        attention_resolutions=(),
    )
    return DDPMForImageGeneration(config).to("metal").eval()


def _mixin_loop(steps: int) -> Callable[[], M.GenerationOutput]:
    model = _ddpm()
    scheduler = model.make_scheduler()
    return lambda: model.generate(scheduler, n_samples=1, num_inference_steps=steps)


def _sd_loop(steps: int) -> Callable[[], M.GenerationOutput]:
    model = _build("stable_diffusion_gen", "metal").eval()
    context = lucid.randn((1, 8, 16), device="metal")
    return lambda: model.generate(context, context, num_inference_steps=steps)


def _labelled_loop(name: str) -> Callable[[int], Callable[[], M.GenerationOutput]]:
    def make(steps: int) -> Callable[[], M.GenerationOutput]:
        model = _build(name, "metal").eval()
        return lambda: model.generate(1, labels=_labels("metal")[:1], steps=steps)

    return make


# dit, mean_flow and tvm embed their class labels once per denoising step;
# nn.Embedding's index check no longer reads the indices back (LCD-283).
_LOOPS = [
    pytest.param(_mixin_loop, id="diffusion_mixin"),
    pytest.param(_sd_loop, id="stable_diffusion"),
    pytest.param(_labelled_loop("dit_small_2_gen"), id="dit"),
    pytest.param(_labelled_loop("mean_flow_base_2_gen"), id="mean_flow"),
    pytest.param(_labelled_loop("tvm_imagenet256_gen"), id="tvm"),
]


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
@pytest.mark.parametrize("loop", _LOOPS)
def test_a_sampling_loop_syncs_a_bounded_number_of_times(
    loop: Callable[[int], Callable[[], M.GenerationOutput]],
) -> None:
    """Timesteps live on the schedule's device (LCD-383); a loop that read
    one back per step would sync once per step."""
    short, long = _syncs(loop(2)), _syncs(loop(10))
    assert long == short <= 1, f"{short} syncs at 2 steps, {long} at 10"


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_the_sync_count_sees_a_read() -> None:
    """Guards the check above: a read on Metal is counted."""
    before = _C_engine.host_sync_count()
    lucid.ones(3, device="metal").sum().item()
    assert _C_engine.host_sync_count() > before
