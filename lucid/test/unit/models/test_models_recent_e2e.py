"""Every family added this cycle, exercised end to end through its factories.

A family can pass all of its own unit tests and still be unusable.  Those
tests reach inside — they build a config directly, call a sub-module,
assert on a shape — and none of that touches the path a caller takes:
factory in, training step and inference out.

That gap is not hypothetical.  DIAMOND's CS:GO upsampler loaded 51M
parameters from the released checkpoint and had no public method that
would invoke it: the module was constructed, assigned, and called from
nowhere.  The config described it, the docs rendered it, the weights
filled it, and it could not be run.  The unit tests, the zoo validator
and the docs build were all green.

So this file does the dumb thing on purpose.  For each family it takes
the *factory*, runs one training step and checks the gradients land, then
runs the inference path and checks the output has the right shape and is
finite.  No internals, no cleverness.

The scope is the sixteen families added since 2026-07-27 — the flows, the
diffusion models, the world models, CLIP and Stable Diffusion.  The older
vision families are left out deliberately: they are far simpler and have
been through many more rounds of verification than these have.

Every model here is built at a size that fits in a unit test, through the
same config overrides a caller would use.  The point is the wiring, not
the arithmetic, and the whole file runs in about a second.
"""

import pytest

import lucid
import lucid.models as models
import lucid.nn as nn


def _grads(module: nn.Module) -> tuple[int, int]:
    """Count how many of a module's parameters received a gradient.

    Parameters
    ----------
    module : nn.Module
        Module to inspect after a backward pass.

    Returns
    -------
    tuple of (int, int)
        Parameters holding a gradient, and parameters in total.
    """
    params = list(module.parameters())
    return sum(1 for p in params if p.grad is not None), len(params)


def _assert_trained(module: nn.Module) -> None:
    """Assert a backward pass reached every parameter of ``module``."""
    got, total = _grads(module)
    assert got == total, f"{total - got} of {total} parameters got no gradient"


def _assert_finite(x: lucid.Tensor, shape: tuple[int, ...]) -> None:
    """Assert a tensor has the expected shape and holds no NaN or infinity."""
    assert x.shape == shape, f"expected {shape}, got {x.shape}"
    assert bool(x.isfinite().all().item()), "output is not finite"


# ---------------------------------------------------------------------------
# Normalising flows — the loss is a log-likelihood, inference is a sample
# ---------------------------------------------------------------------------


class TestNICE:
    CONFIG = dict(
        sample_size=8,
        in_channels=1,
        input_dim=64,
        num_coupling_layers=2,
        num_hidden_layers=2,
        hidden_dim=16,
    )

    def test_a_training_step_reaches_every_parameter(self) -> None:
        model = models.nice_mnist(**self.CONFIG)  # type: ignore[arg-type]
        out = model(lucid.randn((2, 64)))
        (-out.log_prob.mean()).backward()
        _assert_trained(model)

    def test_generation_returns_a_finite_sample(self) -> None:
        model = models.nice_mnist_gen(**self.CONFIG).eval()  # type: ignore[arg-type]
        with lucid.no_grad():
            out = model.generate(2)
        _assert_finite(out.samples, (2, 64))


class TestRealNVP:
    CONFIG = dict(sample_size=8, num_scales=2, residual_blocks=1, base_dim=8)

    def test_a_training_step_reaches_every_parameter(self) -> None:
        model = models.realnvp_cifar(**self.CONFIG)  # type: ignore[arg-type]
        out = model(lucid.randn((2, 3, 8, 8)))
        (-out.log_prob.mean()).backward()
        _assert_trained(model)

    def test_generation_returns_a_finite_sample(self) -> None:
        model = models.realnvp_cifar_gen(**self.CONFIG).eval()  # type: ignore[arg-type]
        with lucid.no_grad():
            out = model.generate(2)
        _assert_finite(out.samples, (2, 3, 8, 8))


class TestNeuralODE:
    """A continuous-time flow, so the solver is part of the path under test."""

    CONFIG = dict(
        sample_size=8,
        in_channels=1,
        out_channels=1,
        hidden_dim=8,
        num_blocks=1,
        solver="euler",
        use_adjoint=False,
    )

    def test_a_training_step_reaches_every_parameter(self) -> None:
        model = models.neural_ode(**self.CONFIG)  # type: ignore[arg-type]
        out = model(lucid.randn((2, 1, 8, 8)))
        (-out.log_prob.mean()).backward()
        _assert_trained(model)

    def test_generation_returns_a_finite_sample(self) -> None:
        model = models.neural_ode_gen(**self.CONFIG).eval()  # type: ignore[arg-type]
        with lucid.no_grad():
            out = model.generate(2)
        _assert_finite(out.samples, (2, 1, 8, 8))


# ---------------------------------------------------------------------------
# Diffusion and flow-matching — the loss is a regression onto a target
# ---------------------------------------------------------------------------


class TestFlowMatching:
    CONFIG = dict(
        sample_size=8,
        base_channels=8,
        num_res_blocks=1,
        resnet_groups=2,
        num_head_channels=8,
        solver="euler",
    )

    def test_a_training_step_reaches_every_parameter(self) -> None:
        model = models.flow_matching_cifar(**self.CONFIG)  # type: ignore[arg-type]
        out = model(lucid.randn((2, 3, 8, 8)), lucid.rand((2,)))
        out.sample.square().mean().backward()
        _assert_trained(model)

    def test_generation_returns_a_finite_sample(self) -> None:
        model = models.flow_matching_cifar_gen(**self.CONFIG).eval()  # type: ignore[arg-type]
        with lucid.no_grad():
            out = model.generate(2, steps=2)
        _assert_finite(out.samples, (2, 3, 8, 8))


class TestRectifiedFlow:
    CONFIG = dict(
        sample_size=8,
        base_channels=8,
        num_res_blocks=1,
        resnet_groups=2,
        solver="euler",
    )

    def test_a_training_step_reaches_every_parameter(self) -> None:
        model = models.rectified_flow_cifar(**self.CONFIG)  # type: ignore[arg-type]
        out = model(lucid.randn((2, 3, 8, 8)), lucid.rand((2,)))
        out.sample.square().mean().backward()
        _assert_trained(model)

    def test_generation_returns_a_finite_sample(self) -> None:
        model = models.rectified_flow_cifar_gen(**self.CONFIG).eval()  # type: ignore[arg-type]
        with lucid.no_grad():
            out = model.generate(2, steps=2)
        _assert_finite(out.samples, (2, 3, 8, 8))


class TestScoreSDE:
    CONFIG = dict(
        sample_size=8,
        base_channels=8,
        channel_mult=(1, 2),
        num_res_blocks=1,
        resnet_groups=2,
        num_heads=1,
    )

    def test_a_training_step_reaches_every_parameter(self) -> None:
        model = models.score_sde_vp(**self.CONFIG)  # type: ignore[arg-type]
        out = model(lucid.randn((2, 3, 8, 8)), lucid.rand((2,)))
        out.score.square().mean().backward()
        _assert_trained(model)

    def test_generation_returns_a_finite_sample(self) -> None:
        model = models.score_sde_vp_gen(**self.CONFIG).eval()  # type: ignore[arg-type]
        with lucid.no_grad():
            out = model.generate(2, steps=2)
        _assert_finite(out.samples, (2, 3, 8, 8))


class TestMeanFlow:
    CONFIG = dict(sample_size=8, hidden_size=32, depth=1, num_heads=4, num_classes=4)

    def test_a_training_step_reaches_every_parameter(self) -> None:
        model = models.mean_flow_base_2_gen(**self.CONFIG)  # type: ignore[arg-type]
        labels = lucid.tensor([0, 1], dtype=lucid.int64)
        model(lucid.randn((2, 4, 8, 8)), labels).loss.backward()
        _assert_trained(model)

    def test_generation_returns_a_finite_sample(self) -> None:
        model = models.mean_flow_base_2_gen(**self.CONFIG).eval()  # type: ignore[arg-type]
        with lucid.no_grad():
            out = model.generate(2, labels=lucid.tensor([0, 1], dtype=lucid.int64))
        _assert_finite(out.samples, (2, 4, 8, 8))


class TestDiT:
    CONFIG = dict(sample_size=8, hidden_size=32, depth=1, num_heads=4, num_classes=4)

    def test_a_training_step_reaches_every_parameter(self) -> None:
        model = models.dit_small_2_gen(**self.CONFIG)  # type: ignore[arg-type]
        labels = lucid.tensor([0, 1], dtype=lucid.int64)
        model(lucid.randn((2, 4, 8, 8)), labels).loss.backward()
        _assert_trained(model)

    def test_generation_returns_a_finite_sample(self) -> None:
        model = models.dit_small_2_gen(**self.CONFIG).eval()  # type: ignore[arg-type]
        with lucid.no_grad():
            out = model.generate(
                2, labels=lucid.tensor([0, 1], dtype=lucid.int64), steps=2
            )
        _assert_finite(out.samples, (2, 4, 8, 8))


# ---------------------------------------------------------------------------
# Autoencoders and multimodal
# ---------------------------------------------------------------------------


class TestVQVAE:
    CONFIG = dict(
        sample_size=8,
        num_embeddings=16,
        embedding_dim=8,
        hidden_channels=8,
        residual_hidden_channels=8,
        num_downsample_layers=1,
        num_residual_layers=1,
    )

    def test_a_training_step_reaches_every_parameter(self) -> None:
        """The bare model is the architecture; the wrapper owns the loss."""
        model = models.vqvae_gen(**self.CONFIG)  # type: ignore[arg-type]
        model(lucid.randn((2, 3, 8, 8))).loss.backward()
        _assert_trained(model)

    def test_reconstruction_runs(self) -> None:
        model = models.vqvae_gen(**self.CONFIG).eval()  # type: ignore[arg-type]
        with lucid.no_grad():
            out = model(lucid.randn((2, 3, 8, 8)))
        _assert_finite(out.sample, (2, 3, 8, 8))

    def test_generation_returns_a_finite_sample(self) -> None:
        model = models.vqvae_gen(**self.CONFIG).eval()  # type: ignore[arg-type]
        with lucid.no_grad():
            out = model.generate(2)
        _assert_finite(out.samples, (2, 3, 8, 8))


class TestCLIP:
    CONFIG = dict(
        embed_dim=16,
        image_size=16,
        patch_size=8,
        vision_layers=1,
        vision_width=16,
        vision_heads=2,
        context_length=8,
        vocab_size=32,
        text_width=16,
        text_heads=2,
        text_layers=1,
    )

    def test_a_contrastive_step_reaches_every_parameter(self) -> None:
        model = models.clip_vit_base_32(**self.CONFIG)  # type: ignore[arg-type]
        pixels = lucid.randn((2, 3, 16, 16))
        ids = lucid.zeros((2, 8), dtype=lucid.int64)
        model(pixels, ids, return_loss=True).loss.backward()
        _assert_trained(model)

    def test_zero_shot_classification_runs(self) -> None:
        """Two images against three prompts, so the logits are (2, 3)."""
        model = models.clip_vit_base_32_zero_shot(**self.CONFIG).eval()  # type: ignore[arg-type]
        with lucid.no_grad():
            out = model(
                lucid.randn((2, 3, 16, 16)),
                lucid.zeros((3, 8), dtype=lucid.int64),
            )
        _assert_finite(out.logits, (2, 3))


class TestStableDiffusion:
    CONFIG = dict(
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

    def test_a_training_step_reaches_the_unet(self) -> None:
        """Only the U-Net is trained; the autoencoder stays frozen."""
        model = models.stable_diffusion(**self.CONFIG)  # type: ignore[arg-type]
        out = model(
            lucid.randn((1, 3, 32, 32)), lucid.randn((1, 8, 16)), return_loss=True
        )
        out.loss.backward()
        _assert_trained(model.unet)

    def test_generation_returns_a_finite_sample(self) -> None:
        model = models.stable_diffusion_gen(**self.CONFIG).eval()  # type: ignore[arg-type]
        with lucid.no_grad():
            sample = model.generate(lucid.randn((1, 8, 16)), num_inference_steps=2)
        _assert_finite(sample, (1, 3, 32, 32))


# ---------------------------------------------------------------------------
# World models — a latent model, a reward head and an actor-critic, each
# trained by its own loss, so "did it train" is per-group rather than global
# ---------------------------------------------------------------------------


_DREAMER_SHARED = dict(
    stoch_size=4,
    deter_size=16,
    hidden_size=16,
    cnn_depth=4,
    actor_hidden=16,
    value_hidden=16,
    reward_hidden=16,
    actor_layers=1,
    value_layers=1,
    horizon=2,
)


class _DreamerCase:
    """Shared body for the three Dreamer generations.

    Each subclass names its factories and its own config; the sequence
    shape (2 batch, 4 steps, 64x64 RGB) is the same for all three because
    the released configurations all encode 64x64 frames.
    """

    BASE: str
    WORLD: str
    ATTR: str
    CONFIG: dict[str, object]
    # Parameters that are updated by an exponential moving average rather
    # than by gradient descent, and so are expected to carry no gradient.
    EMA_ONLY: tuple[str, ...] = ()

    def _batch(self) -> tuple[lucid.Tensor, ...]:
        return (
            lucid.randn((2, 4, 3, 64, 64)),
            lucid.randn((2, 4, 1)),
            lucid.randn((2, 4)),
            lucid.ones((2, 4)),
        )

    def test_a_training_step_reaches_every_trainable_parameter(self) -> None:
        model = getattr(models, self.WORLD)(**self.CONFIG)
        out = model(*self._batch())
        # Dreamer routes each loss to its own parameter group; the wrapper
        # owns that split, so this is the only correct way to train it.
        model.backward(out)
        inner = getattr(model, self.ATTR)
        missing = [n for n, p in inner.named_parameters() if p.grad is None]
        unexpected = [
            n for n in missing if not (self.EMA_ONLY and n.startswith(self.EMA_ONLY))
        ]
        assert not unexpected, f"no gradient: {unexpected}"
        assert bool(out.loss.isfinite().item())

    def test_the_slow_target_is_the_only_thing_left_untrained(self) -> None:
        """A generation with an EMA critic must have exactly that ungradiented."""
        if not self.EMA_ONLY:
            pytest.skip("this generation has no slow target")
        model = getattr(models, self.WORLD)(**self.CONFIG)
        model.backward(model(*self._batch()))
        inner = getattr(model, self.ATTR)
        missing = {n for n, p in inner.named_parameters() if p.grad is None}
        assert missing, "the slow target should not receive a gradient"
        assert all(n.startswith(self.EMA_ONLY) for n in missing)

    def test_acting_from_an_observed_state_runs(self) -> None:
        model = getattr(models, self.BASE)(**self.CONFIG).eval()
        observations, actions, _, _ = self._batch()
        with lucid.no_grad():
            posterior, _ = model.observe(observations, actions)
            action = model.act(posterior)
        _assert_finite(action, (2, 4, 1))


class TestDreamer(_DreamerCase):
    BASE, WORLD, ATTR = "dreamer", "dreamer_world_model", "dreamer"
    CONFIG = dict(_DREAMER_SHARED, reward_layers=1)


class TestDreamerV2(_DreamerCase):
    BASE, WORLD, ATTR = "dreamer_v2", "dreamer_v2_world_model", "dreamer_v2"
    CONFIG = dict(_DREAMER_SHARED, discrete=4, reward_layers=1, pcont_layers=1)
    EMA_ONLY = ("target_value_head.",)


class TestDreamerV3(_DreamerCase):
    BASE, WORLD, ATTR = "dreamer_v3_12m", "dreamer_v3_12m_world_model", "dreamer_v3"
    CONFIG = dict(_DREAMER_SHARED, discrete=4, blocks=2, num_bins=15)
    EMA_ONLY = ("slow_value_head.", "target_value_head.")


class TestDIAMONDAtari:
    """Three networks trained by three separate losses, so three tests."""

    CONFIG = dict(
        sample_size=16,
        unet_channels=(8, 8),
        unet_layers=(1, 1),
        reward_channels=(8, 8),
        reward_layers=(1, 1),
        actor_channels=(8, 8),
        actor_layers=(1, 1),
        cond_dim=16,
        reward_cond_dim=8,
        reward_lstm_dim=16,
        actor_lstm_dim=16,
        num_actions=4,
        horizon=3,
    )

    def _batch(self) -> tuple[lucid.Tensor, lucid.Tensor]:
        return (
            lucid.randn((2, 4, 3, 16, 16)),
            lucid.zeros((2, 4), dtype=lucid.int64),
        )

    def test_the_denoiser_trains(self) -> None:
        model = models.diamond(**self.CONFIG)  # type: ignore[arg-type]
        frames, actions = self._batch()
        out = model(frames, actions, lucid.randn((2, 3, 16, 16)))
        out.loss.backward()
        _assert_trained(model.denoiser)

    def test_the_reward_and_termination_model_trains(self) -> None:
        model = models.diamond_world_model(**self.CONFIG)  # type: ignore[arg-type]
        frames, actions = self._batch()
        loss = model.reward_end_loss(
            frames, actions, lucid.randn((2, 4)), lucid.zeros((2, 4))
        )
        loss.backward()
        _assert_trained(model.diamond.reward_end)

    def test_the_actor_critic_trains_in_imagination(self) -> None:
        model = models.diamond_world_model(**self.CONFIG)  # type: ignore[arg-type]
        frames, actions = self._batch()
        out = model(frames, actions)
        (out.policy_loss + out.value_loss).backward()
        _assert_trained(model.diamond.actor_critic)

    def test_acting_and_imagining_run(self) -> None:
        model = models.diamond_world_model(**self.CONFIG).eval()  # type: ignore[arg-type]
        frames, actions = self._batch()
        with lucid.no_grad():
            action = model.act(frames[:, -1])
            imagined = model.diamond.imagine_frame(frames, actions, steps=2)
        assert action.shape == (2,)
        _assert_finite(imagined, (2, 3, 16, 16))


class TestDIAMONDCSGO:
    """The world model with no agent, at a resolution that is not square."""

    CONFIG = dict(
        unet_channels=(8, 8),
        unet_layers=(1, 1),
        cond_dim=16,
        attn_depths=(0, 0),
        upsampler_channels=(4, 4),
        upsampler_layers=(1, 1),
        upsampler_attn_depths=(0, 0),
        num_actions=4,
    )

    def test_the_denoiser_trains(self) -> None:
        model = models.diamond_csgo(**self.CONFIG)  # type: ignore[arg-type]
        out = model(
            lucid.randn((1, 4, 3, 30, 56)),
            lucid.zeros((1, 4), dtype=lucid.int64),
            lucid.randn((1, 3, 30, 56)),
        )
        out.loss.backward()
        _assert_trained(model.denoiser)

    def test_a_rollout_reaches_full_resolution(self) -> None:
        """30x56 out of the world model, 150x280 out of the upsampler.

        This is the test the family shipped without.  The upsampler was
        built, loaded from the checkpoint, and callable from nowhere — so
        a rollout stopped at the low resolution and the second diffusion
        model was 51M parameters of dead weight.
        """
        model = models.diamond_csgo(**self.CONFIG).eval()  # type: ignore[arg-type]
        frames = lucid.randn((1, 4, 3, 30, 56))
        actions = lucid.zeros((1, 4), dtype=lucid.int64)
        with lucid.no_grad():
            low = model.imagine_frame(frames, actions, steps=1)
            full = model.upsample_frame(low, steps=1)
        _assert_finite(low, (1, 3, 30, 56))
        _assert_finite(full, (1, 3, 150, 280))

    def test_the_upsampler_trains(self) -> None:
        model = models.diamond_csgo(**self.CONFIG)  # type: ignore[arg-type]
        low = lucid.randn((1, 3, 30, 56))
        target = lucid.randn((1, 3, 150, 280))
        loss = (model.upsample_frame(low, steps=1) - target).square().mean()
        loss.backward()
        _assert_trained(model.upsampler)

    def test_the_upsampler_is_reachable_only_where_it_exists(self) -> None:
        """An Atari model has none, and says so rather than crashing."""
        model = models.diamond(**TestDIAMONDAtari.CONFIG)  # type: ignore[arg-type]
        assert model.upsampler is None
        with pytest.raises(ValueError, match="no upsampler"):
            model.upsample_frame(lucid.randn((1, 3, 16, 16)))
