"""Generative-model helpers shared across families.

    * Beta-schedule construction (``make_beta_schedule``)        — diffusion
    * Per-batch schedule indexing (``extract_into_tensor``)      — diffusion
    * Diagonal Gaussian KL divergence (``gaussian_kl_divergence``) — VAE
    * Reparameterisation sample (``reparameterize``)              — VAE
    * Activation dispatch (``generative_activation``)             — both

All operations are stateless and side-effect free.
"""

import math

import lucid
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor

__all__ = [
    "make_beta_schedule",
    "make_sigma_schedule",
    "extract_into_tensor",
    "gaussian_kl_divergence",
    "reparameterize",
    "generative_activation",
]


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion: β / α / cumulative-α schedule
# ─────────────────────────────────────────────────────────────────────────────


def make_beta_schedule(
    num_steps: int,
    schedule: str = "linear",
    *,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    device: str = "cpu",
) -> Tensor:
    """Build the forward-process noise schedule ``β_1, …, β_T``.

    Args:
        num_steps:  Number of diffusion timesteps ``T``.
        schedule:   ``"linear"`` (Ho 2020) or ``"cosine"`` (Nichol-Dhariwal
                    2021).  Cosine uses the formula from Improved DDPM §3:
                    ``ᾱ_t = cos((t/T + s) / (1 + s) · π/2)² with s = 0.008``.
        beta_start: Start of the linear schedule (ignored for cosine).
        beta_end:   End of the linear schedule (ignored for cosine).
        device:     Where to allocate the resulting buffer.

    Returns:
        ``(num_steps,)`` float tensor.
    """
    if schedule == "linear":
        step = (beta_end - beta_start) / max(num_steps - 1, 1)
        values = [beta_start + i * step for i in range(num_steps)]
        return lucid.tensor(values, device=device)

    if schedule == "cosine":
        s = 0.008
        # ᾱ schedule in [0, 1] then β_t = clip(1 - ᾱ_t / ᾱ_{t-1}, 0, 0.999)
        f_values: list[float] = [
            math.cos(((i / num_steps) + s) / (1.0 + s) * math.pi / 2.0) ** 2
            for i in range(num_steps + 1)
        ]
        alpha_bar = [v / f_values[0] for v in f_values]
        betas: list[float] = []
        for i in range(1, num_steps + 1):
            b = 1.0 - alpha_bar[i] / max(alpha_bar[i - 1], 1e-8)
            betas.append(min(max(b, 0.0), 0.999))
        return lucid.tensor(betas, device=device)

    raise ValueError(f"Unsupported beta schedule {schedule!r}")


def extract_into_tensor(
    arr: Tensor,
    timesteps: Tensor,
    broadcast_shape: tuple[int, ...],
) -> Tensor:
    """Gather entries of a 1-D schedule at given timesteps and reshape for
    broadcasting against an image batch.

    Standard diffusion idiom — used by every scheduler ``add_noise`` /
    ``step`` to lift the per-step scalar ``α_t``, ``β_t`` … into the same
    rank as the noisy image so they multiply elementwise.

    Args:
        arr:             ``(T,)`` schedule tensor.
        timesteps:       ``(B,)`` int timesteps to gather.
        broadcast_shape: Target rank.  Usually ``(B, C, H, W)``.

    Returns:
        ``(B, 1, 1, …)`` (rank == ``len(broadcast_shape)``) so it broadcasts
        elementwise against an ``(B, C, H, W)`` image.
    """
    B = int(timesteps.shape[0])
    # Manual gather — ``arr`` is small (≤ a few thousand entries) and we
    # want a backend-agnostic implementation.
    vals = [float(arr[int(timesteps[b].item())].item()) for b in range(B)]
    out = lucid.tensor(vals, device=arr.device.type)
    while out.ndim < len(broadcast_shape):
        out = out.unsqueeze(-1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# VAE: Gaussian KL + reparameterisation trick
# ─────────────────────────────────────────────────────────────────────────────


def gaussian_kl_divergence(
    mu: Tensor,
    logvar: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """KL divergence ``KL(N(μ, σ²) ‖ N(0, I))`` of a diagonal Gaussian.

    Closed form: ``½ Σ (μ² + σ² - log σ² - 1)``.

    Args:
        mu:        ``(B, D)`` posterior means.
        logvar:    ``(B, D)`` posterior log-variances.
        reduction: ``"none"`` (per-sample), ``"sum"`` (over batch), or
                   ``"mean"`` (default — averaged over batch dim).

    Returns:
        Scalar (or ``(B,)`` if ``reduction="none"``).
    """
    var = logvar.exp()
    per_dim = 0.5 * (mu * mu + var - logvar - 1.0)  # (B, D)
    per_sample = per_dim.sum(dim=-1)  # (B,)
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError(f"Unsupported reduction {reduction!r}")


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """Sample ``z = μ + σ·ε`` with ``ε ~ N(0, I)`` (VAE reparameterisation)."""
    std = (0.5 * logvar).exp()
    eps = lucid.randn(mu.shape, device=mu.device.type)
    return mu + std * eps


# ─────────────────────────────────────────────────────────────────────────────
# Activation dispatch (mirrors text_activation pattern)
# ─────────────────────────────────────────────────────────────────────────────


def generative_activation(name: str, x: Tensor) -> Tensor:
    """Apply the activation referenced by a :data:`GenerativeActivation` literal.

    Args:
        name: One of ``"silu"`` / ``"swish"`` / ``"relu"`` / ``"gelu"``.
        x:    Input tensor (elementwise).

    Returns:
        Same shape as ``x``.

    Raises:
        ValueError: If ``name`` is not a supported alias.
    """
    if name in ("silu", "swish"):
        return F.silu(x)
    if name == "relu":
        return F.relu(x)
    if name == "gelu":
        return F.gelu(x, approximate="none")
    raise ValueError(f"Unsupported generative activation {name!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Score-based: geometric sigma schedule
# ─────────────────────────────────────────────────────────────────────────────


def make_sigma_schedule(
    num_noise_levels: int,
    *,
    sigma_max: float,
    sigma_min: float,
    device: str = "cpu",
) -> Tensor:
    """Geometric noise schedule for score-based models (Song & Ermon, 2019).

    Returns ``(L,)`` tensor ``σ_1 > σ_2 > … > σ_L`` with

        σ_i = σ_max · (σ_min / σ_max) ** (i / (L - 1))    for i = 0..L-1

    so ``σ_0 == σ_max`` and ``σ_{L-1} == σ_min``.  Geometric (log-uniform)
    spacing — the choice every NCSN / NCSNv2 / NCSN++ variant uses.

    Args:
        num_noise_levels: Number of levels ``L`` (paper L=10 originally;
            NCSNv2 found large L ∈ [200, 500] works better).
        sigma_max: Largest noise σ_1.
        sigma_min: Smallest noise σ_L.
        device:    Where to allocate the buffer.

    Returns:
        ``(num_noise_levels,)`` float tensor in *descending* order.
    """
    if num_noise_levels <= 0:
        raise ValueError(f"num_noise_levels must be positive, got {num_noise_levels}")
    if not 0.0 < sigma_min < sigma_max:
        raise ValueError(
            f"require 0 < sigma_min < sigma_max, got "
            f"min={sigma_min}, max={sigma_max}"
        )
    if num_noise_levels == 1:
        return lucid.tensor([sigma_max], device=device)

    ratio = sigma_min / sigma_max
    sigmas = [
        sigma_max * (ratio ** (i / (num_noise_levels - 1)))
        for i in range(num_noise_levels)
    ]
    return lucid.tensor(sigmas, device=device)
