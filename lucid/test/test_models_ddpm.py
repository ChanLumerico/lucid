import pytest

import lucid
import lucid.nn as nn

from lucid.models import DDPM, DDPMConfig


class _ToyNoisePredictor(nn.Module):
    def forward(self, x: lucid.Tensor, t: lucid.Tensor) -> lucid.Tensor:
        del t
        return lucid.zeros_like(x)


class _ToyDiffuser(nn.Module):
    def __init__(self, timesteps: int = 4) -> None:
        super().__init__()
        self.timesteps = timesteps

    def sample_timesteps(self, batch_size: int) -> lucid.Tensor:
        return lucid.zeros((batch_size,), dtype=lucid.Int32)

    def add_noise(
        self, x_start: lucid.Tensor, t: lucid.Tensor, noise: lucid.Tensor
    ) -> lucid.Tensor:
        del t
        return x_start + 0.5 * noise

    def denoise(
        self,
        model: nn.Module,
        x: lucid.Tensor,
        t: lucid.Tensor,
        clip_denoised: bool,
    ) -> lucid.Tensor:
        del t
        out = x + model(x, lucid.zeros((x.shape[0],), dtype=lucid.Int32))
        return out.clip(0.0, 1.0) if clip_denoised else out


def _small_ddpm_config(**kwargs: object) -> DDPMConfig:
    params = {
        "model": _ToyNoisePredictor(),
        "image_size": 8,
        "channels": 3,
        "timesteps": 4,
        "diffuser": _ToyDiffuser(4),
        "clip_denoised": False,
    }
    params.update(kwargs)
    return DDPMConfig(**params)


def test_ddpm_public_imports() -> None:
    assert DDPM is not None
    assert DDPMConfig is not None


def test_ddpm_accepts_config_and_runs_loss_and_sample() -> None:
    config = _small_ddpm_config()
    model = DDPM(config)
    x = lucid.ones(2, 3, 8, 8)

    loss = model.get_loss(x)
    samples = model.sample(2)

    assert model.config is config
    assert loss.size == 1
    assert samples.shape == (2, 3, 8, 8)


def test_ddpm_builds_default_components_from_config() -> None:
    config = DDPMConfig(image_size=8, channels=3, timesteps=4)
    model = DDPM(config)

    loss = model.get_loss(lucid.ones(1, 3, 8, 8))

    assert model.config is config
    assert isinstance(model.model, nn.Module)
    assert isinstance(model.diffuser, nn.Module)
    assert loss.size == 1


@pytest.mark.parametrize(
    "kwargs",
    (
        {"model": object()},
        {"image_size": 0},
        {"channels": 0},
        {"timesteps": 0},
        {"diffuser": object()},
        {"clip_denoised": 1},
        {"timesteps": 5, "diffuser": _ToyDiffuser(4)},
    ),
)
def test_ddpm_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        _small_ddpm_config(**kwargs)
