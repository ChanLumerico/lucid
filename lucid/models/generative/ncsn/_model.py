"""NCSN model (Song & Ermon, 2019).

Reuses :class:`DDPMUNet` for the score network (every NCSN variant from
v1 onward converged on the same modern U-Net architecture as diffusion).
The NCSN-specific pieces are:

    1. Geometric ``σ`` schedule (via :func:`make_sigma_schedule`).
    2. Denoising score matching (DSM) loss — Song 2019 Eq. (6).
    3. Annealed Langevin dynamics sampler — Algorithm 1.

Sampling uses the per-σ step-size rule from §4.3:
    ``ε_i = ε · σ_i² / σ_L²`` (where σ_L is the *smallest* noise).
"""

import math
from typing import ClassVar, cast

import lucid
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._output import DiffusionModelOutput, GenerationOutput
from lucid.models._utils._generative import make_sigma_schedule
from lucid.models.generative.ddpm._config import DDPMConfig
from lucid.models.generative.ddpm._model import DDPMUNet
from lucid.models.generative.ncsn._config import NCSNConfig

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _to_unet_config(cfg: NCSNConfig) -> DDPMConfig:
    """Build a :class:`DDPMConfig` shadow so we can instantiate the shared
    ``DDPMUNet`` from an :class:`NCSNConfig` (the U-Net never sees the
    sigma-schedule / Langevin knobs)."""
    return DDPMConfig(
        sample_size=cfg.sample_size,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        act_fn=cfg.act_fn,
        base_channels=cfg.base_channels,
        channel_mult=cfg.channel_mult,
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=cfg.attention_resolutions,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        resnet_groups=cfg.resnet_groups,
        # The DDPM-side knobs below aren't used by the U-Net but the config
        # still validates them — keep defaults that pass.
        num_train_timesteps=max(cfg.num_noise_levels, 2),
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="linear",
        learn_sigma=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bare NCSN trunk
# ─────────────────────────────────────────────────────────────────────────────


class NCSNModel(PretrainedModel):
    """Score network ``s_θ(x̃, σ_i)`` over the geometric σ schedule.

    Forward consumes the **integer noise-level index** ``i ∈ [0, L)`` rather
    than the raw σ value, matching the way NCSNv2 / NCSN++ feed the
    conditioning signal through the U-Net's :class:`TimestepEmbedding`.
    The raw σ table is kept as a non-persistent buffer so it travels with
    ``.to(device=...)``.
    """

    config_class: ClassVar[type[NCSNConfig]] = NCSNConfig
    base_model_prefix: ClassVar[str] = "ncsn"

    sigmas: Tensor

    def __init__(self, config: NCSNConfig) -> None:
        super().__init__(config)
        self.unet = DDPMUNet(_to_unet_config(config))
        sigmas = make_sigma_schedule(
            config.num_noise_levels,
            sigma_max=config.sigma_max,
            sigma_min=config.sigma_min,
        )
        self.register_buffer("sigmas", sigmas, persistent=False)
        self._num_levels = config.num_noise_levels

    @property
    def num_noise_levels(self) -> int:
        return self._num_levels

    def forward(  # type: ignore[override]
        self,
        sample: Tensor,
        sigma_idx: Tensor,
    ) -> DiffusionModelOutput:
        """Predict the score at ``sample`` for the given noise-level indices.

        Args:
            sample:    ``(B, in_channels, H, W)`` perturbed image ``x̃``.
            sigma_idx: ``(B,)`` integer indices into the σ schedule.

        Returns:
            :class:`DiffusionModelOutput` whose ``sample`` field is the raw
            score prediction (same shape as ``sample``).
        """
        score = cast(Tensor, self.unet(sample, sigma_idx))
        return DiffusionModelOutput(sample=score)


# ─────────────────────────────────────────────────────────────────────────────
# Task wrapper — DSM loss + annealed Langevin sampling
# ─────────────────────────────────────────────────────────────────────────────


class NCSNForImageGeneration(PretrainedModel):
    """NCSN + DSM training loss + annealed Langevin ``.generate()``.

    Training contract:
        ``forward(x)`` samples a random noise level per image, perturbs
        ``x`` accordingly, runs the score network, and returns the DSM
        loss (Song 2019 Eq. 6).

    Sampling:
        ``.generate(n_samples)`` runs annealed Langevin dynamics across
        the full σ schedule.  Does **not** reuse
        :class:`DiffusionMixin.generate` because NCSN's nested
        (per-σ × per-step) loop has different semantics from the DDPM
        scheduler step.
    """

    config_class: ClassVar[type[NCSNConfig]] = NCSNConfig
    base_model_prefix: ClassVar[str] = "ncsn"

    sigmas: Tensor

    def __init__(self, config: NCSNConfig) -> None:
        super().__init__(config)
        self.ncsn = NCSNModel(config)
        # Mirror the σ buffer for convenience in loss / sampling code.
        self.register_buffer(
            "sigmas",
            make_sigma_schedule(
                config.num_noise_levels,
                sigma_max=config.sigma_max,
                sigma_min=config.sigma_min,
            ),
            persistent=False,
        )
        self._num_levels = config.num_noise_levels
        self._sigma_min = config.sigma_min
        self._langevin_steps = config.langevin_steps
        self._langevin_eps = config.langevin_eps
        self._in_channels = config.in_channels
        if isinstance(config.sample_size, tuple):
            self._h = int(config.sample_size[0])
            self._w = int(config.sample_size[1])
        else:
            self._h = self._w = int(config.sample_size)

    # ── Training ────────────────────────────────────────────────────────────

    def forward(self, sample: Tensor) -> DiffusionModelOutput:  # type: ignore[override]
        """Denoising Score Matching loss (Song 2019 Eq. 6).

        For each image we sample ``i ~ Uniform({0, …, L-1})``, perturb
        ``x̃ = x + σ_i · z`` with ``z ~ N(0, I)``, and minimise

            L = ½ · E ‖ σ_i · s_θ(x̃, i) + z ‖²

        which is the σ-weighted DSM loss.
        """
        B = int(sample.shape[0])
        dev = sample.device.type
        # Random noise level per image.
        idx_list = [
            int(lucid.randint(0, self._num_levels, (1,)).item()) for _ in range(B)
        ]
        sigma_idx = lucid.tensor(idx_list, device=dev).long()

        # Look up per-image σ values, reshape to (B, 1, 1, 1) for broadcast.
        sigma_vals = [float(self.sigmas[i].item()) for i in idx_list]
        sigma = lucid.tensor(sigma_vals, device=dev).reshape(B, 1, 1, 1)

        z = lucid.randn(sample.shape, device=dev)
        x_tilde = sample + sigma * z
        score = cast(
            DiffusionModelOutput, self.ncsn(x_tilde, sigma_idx)
        ).sample  # ŝ_θ(x̃, i)

        residual = sigma * score + z  # (B, C, H, W)
        loss = 0.5 * (residual * residual).mean()
        return DiffusionModelOutput(sample=score, loss=loss)

    # ── Sampling ────────────────────────────────────────────────────────────

    @lucid.no_grad()
    def generate(
        self,
        n_samples: int = 1,
        *,
        langevin_steps: int | None = None,
        return_intermediates: bool = False,
        device: str = "cpu",
    ) -> GenerationOutput:
        """Annealed Langevin dynamics — Song 2019 Algorithm 1.

        Initialises ``x ~ N(0, σ_max² I)`` then iterates over the σ
        schedule from largest to smallest noise.  At each σ_i takes
        ``langevin_steps`` Langevin steps with step size
        ``α_i = ε · σ_i² / σ_L²``:

            x ← x + (α_i / 2) · s_θ(x, i) + √α_i · z      z ~ N(0, I)

        Args:
            n_samples:           Batch size of the generated images.
            langevin_steps:      Override ``config.langevin_steps`` for this
                                 call (typically lowered in unit tests).
            return_intermediates: If True, also returns one sample per σ
                                  level (after its inner Langevin loop).
            device:              Where to allocate the initial noise.

        Returns:
            :class:`GenerationOutput` with the final ``(n_samples, C, H, W)``
            samples and optional per-level intermediates.
        """
        T = langevin_steps if langevin_steps is not None else self._langevin_steps

        shape = (n_samples, self._in_channels, self._h, self._w)
        # Initial sample at the largest σ (Song 2019 §4.3 — "uniform noise"
        # over [0, 1] also works; we follow NCSNv2's Gaussian init).
        x = lucid.randn(shape, device=device) * float(self.sigmas[0].item())

        intermediates: list[Tensor] = []
        sigma_min_sq = self._sigma_min**2
        for i in range(self._num_levels):
            sigma_i = float(self.sigmas[i].item())
            alpha_i = self._langevin_eps * (sigma_i**2) / sigma_min_sq
            sigma_idx = lucid.tensor([i] * n_samples, device=device).long()
            sqrt_alpha = math.sqrt(alpha_i)
            for _ in range(T):
                score = cast(
                    DiffusionModelOutput, self.ncsn(x, sigma_idx)
                ).sample  # (B, C, H, W)
                z = lucid.randn(shape, device=device)
                x = x + (alpha_i / 2.0) * score + sqrt_alpha * z
            if return_intermediates:
                intermediates.append(x)

        return GenerationOutput(
            samples=x,
            intermediates=tuple(intermediates) if return_intermediates else None,
        )
