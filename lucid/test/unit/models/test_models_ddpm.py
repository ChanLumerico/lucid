"""Unit tests for DDPM (Ho et al., 2020).

Validates the U-Net architecture (encoder/middle/decoder skip topology),
ResBlock + AttentionBlock building blocks, training loss path, and the
end-to-end sampling loop via :class:`DiffusionMixin`.

Uses a tiny config (16×16, base_channels=16, 2-stage) so the full suite
runs in under a second.
"""

import pytest

import lucid
from lucid.models import (
    DDPMConfig,
    DDPMForImageGeneration,
    DDPMModel,
    DDPMScheduler,
    DDPMUNet,
    DiffusionModelOutput,
    GenerationOutput,
    create_model,
    is_model,
)


def _tiny_cfg(**overrides: object) -> DDPMConfig:
    base = {
        "sample_size": 16,
        "in_channels": 3,
        "out_channels": 3,
        "base_channels": 16,
        "channel_mult": (1, 2),
        "num_res_blocks": 1,
        "attention_resolutions": (8,),
        "num_heads": 2,
        "resnet_groups": 8,
        "num_train_timesteps": 20,
    }
    base.update(overrides)
    return DDPMConfig(**base)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class TestDDPMConfig:
    def test_paper_defaults(self) -> None:
        cfg = DDPMConfig()
        # Ho 2020 CIFAR setup defaults
        assert cfg.base_channels == 128
        assert cfg.channel_mult == (1, 2, 2, 2)
        assert cfg.num_res_blocks == 2
        assert cfg.attention_resolutions == (16,)
        assert cfg.num_heads == 1
        assert cfg.dropout == 0.1
        assert cfg.resnet_groups == 32
        assert cfg.num_train_timesteps == 1000
        assert cfg.beta_schedule == "linear"
        assert cfg.learn_sigma is False
        assert cfg.model_type == "ddpm"

    def test_sample_size_must_divide(self) -> None:
        # sample_size=30 not divisible by 2^(L-1) = 2^3 = 8 (with default mult).
        with pytest.raises(ValueError, match="divisible"):
            DDPMConfig(sample_size=30)

    def test_invalid_base_channels(self) -> None:
        with pytest.raises(ValueError, match="base_channels"):
            DDPMConfig(base_channels=0)

    def test_empty_channel_mult(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            DDPMConfig(channel_mult=())

    def test_dropout_bounds(self) -> None:
        with pytest.raises(ValueError, match="dropout"):
            DDPMConfig(dropout=1.0)

    def test_groups_must_divide_channels(self) -> None:
        # 17 doesn't divide 128 cleanly
        with pytest.raises(ValueError, match="resnet_groups"):
            DDPMConfig(base_channels=128, resnet_groups=17)

    def test_learn_sigma_doubles_output(self) -> None:
        cfg = DDPMConfig(learn_sigma=True)
        assert cfg.out_channels_effective == 2 * cfg.in_channels

    def test_no_learn_sigma_keeps_output(self) -> None:
        cfg = DDPMConfig(learn_sigma=False)
        assert cfg.out_channels_effective == cfg.in_channels


# ─────────────────────────────────────────────────────────────────────────────
# U-Net architecture
# ─────────────────────────────────────────────────────────────────────────────


class TestDDPMUNet:
    def test_forward_shape(self) -> None:
        cfg = _tiny_cfg()
        unet = DDPMUNet(cfg).eval()
        x = lucid.randn((2, 3, 16, 16))
        t = lucid.tensor([5, 15]).long()
        out = unet(x, t)
        assert tuple(out.shape) == (2, 3, 16, 16)

    def test_scalar_timestep_is_broadcast(self) -> None:
        cfg = _tiny_cfg()
        unet = DDPMUNet(cfg).eval()
        x = lucid.randn((2, 3, 16, 16))
        # Scalar tensor (rank-0) — should expand to batch.
        t = lucid.tensor(5).long()
        out = unet(x, t)
        assert tuple(out.shape) == (2, 3, 16, 16)

    def test_learn_sigma_doubles_channels(self) -> None:
        cfg = _tiny_cfg(learn_sigma=True)
        unet = DDPMUNet(cfg).eval()
        x = lucid.randn((1, 3, 16, 16))
        out = unet(x, lucid.tensor([0]).long())
        assert tuple(out.shape) == (1, 6, 16, 16)

    def test_attention_only_at_listed_resolutions(self) -> None:
        """Counting attention blocks: only stages whose spatial size is in
        ``attention_resolutions`` should have non-Identity attention slots."""
        cfg = _tiny_cfg(attention_resolutions=(8,))  # only the 8×8 stage
        unet = DDPMUNet(cfg).eval()
        # Encoder side: stage 0 (16×16) → no attn; stage 1 (8×8 after down) → attn
        non_id_down = sum(
            1 for blk in unet.down_attn if not isinstance(blk, lucid.nn.Identity)
        )
        # Tiny config has num_res_blocks=1 per stage.  Stage-0 spatial == 16
        # (no attn), stage-1 spatial == 8 (attn).  So expect 1.
        assert non_id_down == 1

    def test_different_sample_size(self) -> None:
        cfg = _tiny_cfg(sample_size=32)
        unet = DDPMUNet(cfg).eval()
        x = lucid.randn((1, 3, 32, 32))
        out = unet(x, lucid.tensor([0]).long())
        assert tuple(out.shape) == (1, 3, 32, 32)


# ─────────────────────────────────────────────────────────────────────────────
# DDPMModel + ForImageGeneration
# ─────────────────────────────────────────────────────────────────────────────


class TestDDPMModel:
    def test_returns_diffusion_output(self) -> None:
        cfg = _tiny_cfg()
        m = DDPMModel(cfg).eval()
        out = m(
            lucid.randn((1, 3, 16, 16)),
            lucid.tensor([0]).long(),
        )
        assert isinstance(out, DiffusionModelOutput)
        assert tuple(out.sample.shape) == (1, 3, 16, 16)
        assert out.loss is None


class TestDDPMForImageGeneration:
    def test_loss_when_target_provided(self) -> None:
        cfg = _tiny_cfg()
        m = DDPMForImageGeneration(cfg).eval()
        x = lucid.randn((2, 3, 16, 16))
        t = lucid.tensor([5, 15]).long()
        target = lucid.randn((2, 3, 16, 16))
        out = m(x, t, target=target)
        assert out.loss is not None
        assert float(out.loss.item()) >= 0.0

    def test_no_loss_without_target(self) -> None:
        cfg = _tiny_cfg()
        m = DDPMForImageGeneration(cfg).eval()
        out = m(
            lucid.randn((1, 3, 16, 16)),
            lucid.tensor([0]).long(),
        )
        assert out.loss is None

    def test_learn_sigma_raises_not_implemented(self) -> None:
        """Improved-DDPM ``learn_sigma=True`` requires the hybrid
        ``L_simple + L_vlb`` loss that Lucid does not yet implement —
        the constructor refuses rather than silently emitting an unusable
        variance head."""
        cfg = _tiny_cfg(learn_sigma=True)
        with pytest.raises(NotImplementedError):
            DDPMForImageGeneration(cfg)


class TestDDPMSampling:
    def test_generate_via_diffusion_mixin(self) -> None:
        cfg = _tiny_cfg()
        m = DDPMForImageGeneration(cfg).eval()
        sched = DDPMScheduler(num_train_timesteps=cfg.num_train_timesteps)
        out = m.generate(sched, n_samples=2, num_inference_steps=4)
        assert isinstance(out, GenerationOutput)
        assert tuple(out.samples.shape) == (2, 3, 16, 16)

    def test_generate_intermediates(self) -> None:
        cfg = _tiny_cfg()
        m = DDPMForImageGeneration(cfg).eval()
        sched = DDPMScheduler(num_train_timesteps=cfg.num_train_timesteps)
        out = m.generate(
            sched, n_samples=1, num_inference_steps=3, return_intermediates=True
        )
        assert out.intermediates is not None
        assert len(out.intermediates) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


class TestDDPMRegistry:
    @pytest.mark.parametrize(
        "name",
        [
            "ddpm_cifar",
            "ddpm_lsun",
            "ddpm_imagenet64",
            "ddpm_cifar_gen",
            "ddpm_lsun_gen",
            "ddpm_imagenet64_gen",
        ],
    )
    def test_registered(self, name: str) -> None:
        assert is_model(name)

    def test_factory_with_override(self) -> None:
        m = create_model(
            "ddpm_cifar",
            sample_size=16,
            base_channels=16,
            channel_mult=(1, 2),
            num_res_blocks=1,
            attention_resolutions=(8,),
            num_heads=2,
            resnet_groups=8,
            num_train_timesteps=20,
        )
        assert isinstance(m, DDPMModel)
        out = m.eval()(
            lucid.randn((1, 3, 16, 16)),
            lucid.tensor([0]).long(),
        )
        assert tuple(out.sample.shape) == (1, 3, 16, 16)

    def test_auto_image_generation_dispatch(self) -> None:
        from lucid.models import AutoModelForImageGeneration
        from lucid.models._registry import _registry_lookup

        entry = _registry_lookup(
            "ddpm_cifar_gen", task=AutoModelForImageGeneration._task
        )
        assert entry.model_class is DDPMForImageGeneration
