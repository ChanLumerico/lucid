"""Unit tests for Phase 5 base layer — generative-domain infrastructure.

Mirrors ``test_models_text_infra.py``: covers configs, output dataclasses,
shared utils, schedulers, and the :class:`DiffusionMixin` sampling loop.
The first concrete family (VAE / DDPM / NCSN) gets its own test file.
"""

import pytest

import lucid
import lucid.nn as nn
from lucid.models import (
    DDPMScheduler,
    DiffusionMixin,
    DiffusionModelConfig,
    DiffusionModelOutput,
    GenerationOutput,
    GenerativeModelConfig,
    DiffusionScheduler,
    VAEOutput,
)
from lucid.models._base import PretrainedModel
from lucid.models._utils import (
    extract_into_tensor,
    gaussian_kl_divergence,
    generative_activation,
    make_beta_schedule,
    reparameterize,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configs
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerativeConfigs:
    def test_generative_base_defaults(self) -> None:
        cfg = GenerativeModelConfig()
        assert cfg.sample_size == 32
        assert cfg.in_channels == 3
        assert cfg.out_channels == 3
        assert cfg.act_fn == "silu"
        assert cfg.model_type == "generative"

    def test_tuple_sample_size(self) -> None:
        cfg = GenerativeModelConfig(sample_size=(16, 24))
        assert cfg.sample_size == (16, 24)

    def test_negative_sample_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="sample_size"):
            GenerativeModelConfig(sample_size=0)

    def test_bad_tuple_rejected(self) -> None:
        with pytest.raises(ValueError, match="sample_size tuple"):
            GenerativeModelConfig(sample_size=(16, 0))

    def test_diffusion_defaults(self) -> None:
        cfg = DiffusionModelConfig()
        assert cfg.num_train_timesteps == 1000
        assert cfg.beta_schedule == "linear"
        assert cfg.prediction_type == "epsilon"

    def test_beta_invariant(self) -> None:
        with pytest.raises(ValueError, match="beta_start"):
            DiffusionModelConfig(beta_start=0.5, beta_end=0.3)

    def test_negative_timesteps_rejected(self) -> None:
        with pytest.raises(ValueError, match="num_train_timesteps"):
            DiffusionModelConfig(num_train_timesteps=0)


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestBetaSchedule:
    def test_linear_endpoints(self) -> None:
        betas = make_beta_schedule(100, "linear", beta_start=1e-4, beta_end=0.02)
        assert tuple(betas.shape) == (100,)
        assert abs(float(betas[0].item()) - 1e-4) < 1e-6
        assert abs(float(betas[-1].item()) - 0.02) < 1e-6

    def test_cosine_monotonic_alpha_bar(self) -> None:
        # Cosine schedule's cumulative-α (ᾱ_t) must be monotonically decreasing.
        betas = make_beta_schedule(50, "cosine")
        alphas = 1.0 - betas
        cum: list[float] = []
        acc = 1.0
        for i in range(50):
            acc *= float(alphas[i].item())
            cum.append(acc)
        for i in range(1, 50):
            assert cum[i] < cum[i - 1], f"ᾱ not decreasing at i={i}"

    def test_unknown_schedule_rejected(self) -> None:
        with pytest.raises(ValueError, match="schedule"):
            make_beta_schedule(10, "exp")


class TestExtractIntoTensor:
    def test_shape_broadcast(self) -> None:
        arr = lucid.tensor([0.1, 0.2, 0.3, 0.4]).float()
        ts = lucid.tensor([0, 2, 3]).long()
        out = extract_into_tensor(arr, ts, (3, 4, 8, 8))
        assert tuple(out.shape) == (3, 1, 1, 1)
        assert abs(float(out[0, 0, 0, 0].item()) - 0.1) < 1e-6
        assert abs(float(out[1, 0, 0, 0].item()) - 0.3) < 1e-6
        assert abs(float(out[2, 0, 0, 0].item()) - 0.4) < 1e-6


class TestGaussianKL:
    def test_zero_for_standard_normal(self) -> None:
        # mu=0, logvar=0 → KL(N(0,I) ‖ N(0,I)) = 0
        mu = lucid.zeros((8, 4))
        logvar = lucid.zeros((8, 4))
        kl = gaussian_kl_divergence(mu, logvar)
        assert abs(float(kl.item())) < 1e-6

    def test_positive_for_displaced_mean(self) -> None:
        mu = lucid.ones((4, 4))  # μ = 1
        logvar = lucid.zeros((4, 4))
        kl = gaussian_kl_divergence(mu, logvar)
        # KL = 0.5 * sum(μ²) per sample = 0.5 * 4 = 2
        assert abs(float(kl.item()) - 2.0) < 1e-5

    def test_reduction_modes(self) -> None:
        mu = lucid.ones((4, 4))
        logvar = lucid.zeros((4, 4))
        per = gaussian_kl_divergence(mu, logvar, reduction="none")
        assert tuple(per.shape) == (4,)
        s = gaussian_kl_divergence(mu, logvar, reduction="sum")
        assert abs(float(s.item()) - 8.0) < 1e-5


class TestReparameterize:
    def test_shape_preserved(self) -> None:
        mu = lucid.zeros((4, 8))
        logvar = lucid.zeros((4, 8))
        z = reparameterize(mu, logvar)
        assert tuple(z.shape) == (4, 8)


class TestGenerativeActivation:
    def test_silu_alias(self) -> None:
        x = lucid.tensor([1.0, -1.0]).float()
        out_silu = generative_activation("silu", x)
        out_swish = generative_activation("swish", x)
        diff = float(((out_silu - out_swish) ** 2).sum().item())
        assert diff < 1e-10

    def test_unknown_rejected(self) -> None:
        with pytest.raises(ValueError, match="activation"):
            generative_activation("nope", lucid.tensor([0.0]).float())


# ─────────────────────────────────────────────────────────────────────────────
# TimestepEmbedding
# ─────────────────────────────────────────────────────────────────────────────


class TestTimestepEmbedding:
    def test_shape(self) -> None:
        te = nn.TimestepEmbedding(in_dim=16, out_dim=64).eval()
        out = te(lucid.tensor([0, 10, 100, 500]).long())
        assert tuple(out.shape) == (4, 64)

    def test_odd_in_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="even in_dim"):
            nn.TimestepEmbedding(in_dim=7, out_dim=32)

    def test_distinct_per_timestep(self) -> None:
        """Different t must map to different embedding vectors."""
        te = nn.TimestepEmbedding(in_dim=16, out_dim=32).eval()
        out = te(lucid.tensor([0, 1, 2, 3]).long())
        for i in range(1, 4):
            diff = float(((out[i] - out[0]) ** 2).sum().item())
            assert diff > 1e-6, f"row {i} equals row 0"


# ─────────────────────────────────────────────────────────────────────────────
# DDPM scheduler
# ─────────────────────────────────────────────────────────────────────────────


class TestDDPMScheduler:
    def test_default_timesteps(self) -> None:
        sched = DDPMScheduler(num_train_timesteps=100)
        assert tuple(sched.timesteps.shape) == (100,)
        # Default schedule is reverse (99 → 0).
        assert int(sched.timesteps[0].item()) == 99
        assert int(sched.timesteps[-1].item()) == 0

    def test_set_inference_timesteps(self) -> None:
        sched = DDPMScheduler(num_train_timesteps=100)
        sched.set_timesteps(10)
        assert tuple(sched.timesteps.shape) == (10,)

    def test_inference_exceeds_train_rejected(self) -> None:
        sched = DDPMScheduler(num_train_timesteps=10)
        with pytest.raises(ValueError, match="exceeds"):
            sched.set_timesteps(20)

    def test_add_noise_shape(self) -> None:
        sched = DDPMScheduler(num_train_timesteps=50)
        x0 = lucid.randn((2, 3, 8, 8))
        noise = lucid.randn((2, 3, 8, 8))
        t = lucid.tensor([10, 40]).long()
        xt = sched.add_noise(x0, noise, t)
        assert tuple(xt.shape) == (2, 3, 8, 8)

    def test_step_shape(self) -> None:
        sched = DDPMScheduler(num_train_timesteps=50)
        sample = lucid.randn((1, 3, 8, 8))
        pred = lucid.randn((1, 3, 8, 8))
        prev = sched.step(pred, 25, sample)
        assert tuple(prev.shape) == (1, 3, 8, 8)

    def test_step_t_zero_is_deterministic(self) -> None:
        """At t=0 the reverse step returns mean only (no Gaussian noise)."""
        sched = DDPMScheduler(num_train_timesteps=10)
        sample = lucid.zeros((1, 1, 4, 4))
        pred = lucid.zeros((1, 1, 4, 4))  # predicting zero noise
        out1 = sched.step(pred, 0, sample)
        out2 = sched.step(pred, 0, sample)
        diff = float(((out1 - out2) ** 2).sum().item())
        assert diff < 1e-10, "t=0 step should be deterministic"

    def test_prediction_type_validation(self) -> None:
        with pytest.raises(ValueError, match="prediction_type"):
            DDPMScheduler(prediction_type="bogus")


# ─────────────────────────────────────────────────────────────────────────────
# DiffusionMixin — end-to-end sampling loop
# ─────────────────────────────────────────────────────────────────────────────


class _DummyDiffusionModel(PretrainedModel, DiffusionMixin):
    """Identity-noise predictor over a single conv — minimal mixin host."""

    config_class = DiffusionModelConfig
    base_model_prefix = "dummy"

    def __init__(self, config: DiffusionModelConfig) -> None:
        super().__init__(config)
        self.conv = nn.Conv2d(config.in_channels, config.in_channels, 3, padding=1)

    def forward(self, sample, timestep):  # type: ignore[override]
        return DiffusionModelOutput(sample=self.conv(sample))


class TestDiffusionMixin:
    def _tiny_cfg(self) -> DiffusionModelConfig:
        return DiffusionModelConfig(
            sample_size=8, in_channels=3, num_train_timesteps=20
        )

    def test_generate_shape(self) -> None:
        cfg = self._tiny_cfg()
        m = _DummyDiffusionModel(cfg).eval()
        sched = DDPMScheduler(num_train_timesteps=cfg.num_train_timesteps)
        out = m.generate(sched, n_samples=2, num_inference_steps=4)
        assert isinstance(out, GenerationOutput)
        assert tuple(out.samples.shape) == (2, 3, 8, 8)
        assert out.intermediates is None

    def test_generate_intermediates(self) -> None:
        cfg = self._tiny_cfg()
        m = _DummyDiffusionModel(cfg).eval()
        sched = DDPMScheduler(num_train_timesteps=cfg.num_train_timesteps)
        out = m.generate(
            sched, n_samples=1, num_inference_steps=3, return_intermediates=True
        )
        assert out.intermediates is not None
        assert len(out.intermediates) == 3
        for inter in out.intermediates:
            assert tuple(inter.shape) == (1, 3, 8, 8)

    def test_generate_no_config_no_shape_raises(self) -> None:
        cfg = self._tiny_cfg()
        m = _DummyDiffusionModel(cfg).eval()
        m.config = None  # type: ignore[assignment]
        sched = DDPMScheduler(num_train_timesteps=cfg.num_train_timesteps)
        with pytest.raises(RuntimeError, match="generator_shape"):
            m.generate(sched, n_samples=1, num_inference_steps=2)


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclasses sanity
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerativeOutputs:
    def test_diffusion_output(self) -> None:
        out = DiffusionModelOutput(sample=lucid.zeros((1, 3, 8, 8)))
        assert out.loss is None

    def test_vae_output_fields(self) -> None:
        out = VAEOutput(
            sample=lucid.zeros((1, 3, 8, 8)),
            latent=lucid.zeros((1, 8)),
            mu=lucid.zeros((1, 8)),
            logvar=lucid.zeros((1, 8)),
        )
        assert out.recon_loss is None and out.kl_loss is None

    def test_generation_output(self) -> None:
        out = GenerationOutput(samples=lucid.zeros((4, 3, 8, 8)))
        assert out.intermediates is None


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler ABC sanity
# ─────────────────────────────────────────────────────────────────────────────


class TestSchedulerABC:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            DiffusionScheduler()  # type: ignore[abstract]

    def test_ddpm_satisfies_contract(self) -> None:
        assert issubclass(DDPMScheduler, DiffusionScheduler)
