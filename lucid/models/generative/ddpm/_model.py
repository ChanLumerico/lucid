"""DDPM model (Ho et al., 2020) — U-Net denoiser with timestep conditioning.

Architecture summary
--------------------
Time embedding:
    t → SinusoidalEmbedding(base_channels) → Linear(4·base_channels) → SiLU → Linear

Encoder (per stage l = 0 … L-1):
    L_l ResBlocks (in_ch → base_channels · channel_mult[l])
    optional AttentionBlock at the spatial resolutions in ``attention_resolutions``
    Downsample (stride-2 Conv) — skipped for the last stage

Middle:
    ResBlock → AttentionBlock → ResBlock

Decoder (per stage l = L-1 … 0):
    L_l + 1 ResBlocks consuming concatenated skip features
    optional AttentionBlock
    Upsample (nearest + Conv) — skipped for stage 0

Out:
    GroupNorm → SiLU → Conv 3×3 → ``out_channels_effective``

ResBlock conditioning recipe (Ho 2020 §3.2 + Appendix B):
    h = Conv₂(SiLU(Norm₂(SiLU(t·Linear)+ Conv₁(SiLU(Norm₁(x))))))
    out = h + Skip(x)
"""

import math
from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import DiffusionMixin
from lucid.models._output import DiffusionModelOutput
from lucid.models.generative.ddpm._config import DDPMConfig

# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────


class _ResBlock(nn.Module):
    """Residual block with timestep conditioning.

    Two GroupNorm + SiLU + Conv 3×3 sub-layers, with a per-channel additive
    timestep embedding injected between them.  Identity (or 1×1 Conv) skip.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        *,
        dropout: float = 0.0,
        groups: int = 32,
    ) -> None:
        super().__init__()
        # GroupNorm requires channels % groups == 0; clamp ``groups`` down
        # when the user-supplied count exceeds the available channels.
        g_in = min(groups, in_channels) if in_channels % groups != 0 else groups
        g_out = min(groups, out_channels) if out_channels % groups != 0 else groups

        self.norm1 = nn.GroupNorm(g_in, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(g_out, out_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip: nn.Module = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:  # type: ignore[override]
        h = cast(Tensor, self.conv1(F.silu(cast(Tensor, self.norm1(x)))))
        # (B, out_ch) → (B, out_ch, 1, 1) so it broadcasts across spatial.
        t = cast(Tensor, self.time_proj(F.silu(t_emb))).unsqueeze(-1).unsqueeze(-1)
        h = h + t
        h = cast(Tensor, self.norm2(h))
        h = F.silu(h)
        h = cast(Tensor, self.dropout(h))
        h = cast(Tensor, self.conv2(h))
        return h + cast(Tensor, self.skip(x))


class _AttentionBlock(nn.Module):
    """Self-attention on a 2-D feature map (Ho 2020 §3.2).

    GroupNorm → 1×1 Conv → split QKV → scaled-dot-product → 1×1 Conv proj.
    Output is added to the input (residual).
    """

    def __init__(
        self,
        channels: int,
        *,
        num_heads: int = 1,
        groups: int = 32,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        g = min(groups, channels) if channels % groups != 0 else groups
        self.norm = nn.GroupNorm(g, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        B, C, H, W = x.shape
        N = int(H) * int(W)
        h = cast(Tensor, self.norm(x))
        qkv = cast(Tensor, self.qkv(h))  # (B, 3C, H, W)
        # Reshape to (B, 3, num_heads, head_dim, N) then permute heads to first.
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, N)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, H, N, D)
        q: Tensor = qkv[0]  # (B, H, N, D)
        k: Tensor = qkv[1]
        v: Tensor = qkv[2]
        scores = q @ k.permute(0, 1, 3, 2) * self.scale  # (B, H, N, N)
        attn = F.softmax(scores, dim=-1)
        out: Tensor = attn @ v  # (B, H, N, D)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + cast(Tensor, self.proj(out))


class _Downsample(nn.Module):
    """Stride-2 3×3 conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.op(x))


class _Upsample(nn.Module):
    """Nearest-neighbour ×2 + 3×3 conv (Ho 2020 §3.2)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        B, C, H, W = x.shape
        x_up = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return cast(Tensor, self.op(x_up))


# ─────────────────────────────────────────────────────────────────────────────
# DDPM U-Net trunk
# ─────────────────────────────────────────────────────────────────────────────


class DDPMUNet(nn.Module):
    """U-Net denoiser shared by every DDPM variant.

    Spatial layout per stage ``l ∈ [0, L)``:
        * ``num_res_blocks`` ResBlocks (with optional attention).
        * Downsample → next stage (skipped for ``l == L - 1``).

    Decoder mirrors the encoder with concatenated skip features; the extra
    ResBlock per stage (``num_res_blocks + 1`` total in the decoder side)
    consumes the post-downsample skip.
    """

    def __init__(self, config: DDPMConfig) -> None:
        super().__init__()
        self._config = config
        L = len(config.channel_mult)
        base = config.base_channels
        time_dim = base * 4

        self.time_mlp = nn.TimestepEmbedding(in_dim=base, out_dim=time_dim)

        self.conv_in = nn.Conv2d(config.in_channels, base, 3, padding=1)

        # ── Encoder ─────────────────────────────────────────────────────────
        if isinstance(config.sample_size, tuple):
            h0 = config.sample_size[0]
        else:
            h0 = int(config.sample_size)
        spatial = h0  # use H

        self.down_res: nn.ModuleList = nn.ModuleList()  # ResBlocks
        self.down_attn: nn.ModuleList = nn.ModuleList()  # AttentionBlock | None
        self.down_sample: nn.ModuleList = nn.ModuleList()  # Downsample | None per stage
        skip_channels: list[int] = [base]  # initial conv output
        ch = base
        for level, mult in enumerate(config.channel_mult):
            out_ch = base * mult
            for _ in range(config.num_res_blocks):
                self.down_res.append(
                    _ResBlock(
                        ch,
                        out_ch,
                        time_dim,
                        dropout=config.dropout,
                        groups=config.resnet_groups,
                    )
                )
                ch = out_ch
                if spatial in config.attention_resolutions:
                    self.down_attn.append(
                        _AttentionBlock(
                            ch,
                            num_heads=config.num_heads,
                            groups=config.resnet_groups,
                        )
                    )
                else:
                    self.down_attn.append(nn.Identity())
                skip_channels.append(ch)
            if level != L - 1:
                self.down_sample.append(_Downsample(ch))
                skip_channels.append(ch)
                spatial //= 2
            else:
                self.down_sample.append(nn.Identity())

        # Cache the per-block stage so forward can decide when to downsample.
        # Encoder loop visits N = sum_l num_res_blocks ResBlocks; we apply
        # downsample once after each stage except the last.
        self._encoder_stages = L

        # ── Middle ──────────────────────────────────────────────────────────
        self.mid_block1 = _ResBlock(
            ch,
            ch,
            time_dim,
            dropout=config.dropout,
            groups=config.resnet_groups,
        )
        self.mid_attn = _AttentionBlock(
            ch,
            num_heads=config.num_heads,
            groups=config.resnet_groups,
        )
        self.mid_block2 = _ResBlock(
            ch,
            ch,
            time_dim,
            dropout=config.dropout,
            groups=config.resnet_groups,
        )

        # ── Decoder ─────────────────────────────────────────────────────────
        self.up_res: nn.ModuleList = nn.ModuleList()
        self.up_attn: nn.ModuleList = nn.ModuleList()
        self.up_sample: nn.ModuleList = nn.ModuleList()
        for level, mult in list(enumerate(config.channel_mult))[::-1]:
            out_ch = base * mult
            for _ in range(config.num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                self.up_res.append(
                    _ResBlock(
                        ch + skip_ch,
                        out_ch,
                        time_dim,
                        dropout=config.dropout,
                        groups=config.resnet_groups,
                    )
                )
                ch = out_ch
                if spatial in config.attention_resolutions:
                    self.up_attn.append(
                        _AttentionBlock(
                            ch,
                            num_heads=config.num_heads,
                            groups=config.resnet_groups,
                        )
                    )
                else:
                    self.up_attn.append(nn.Identity())
            if level != 0:
                self.up_sample.append(_Upsample(ch))
                spatial *= 2
            else:
                self.up_sample.append(nn.Identity())

        # ── Out ─────────────────────────────────────────────────────────────
        g_out = (
            min(config.resnet_groups, ch)
            if ch % config.resnet_groups != 0
            else config.resnet_groups
        )
        self.norm_out = nn.GroupNorm(g_out, ch)
        self.conv_out = nn.Conv2d(ch, config.out_channels_effective, 3, padding=1)

        # Pre-computed "blocks per stage" for forward dispatch.
        self._blocks_per_down_stage = config.num_res_blocks
        self._blocks_per_up_stage = config.num_res_blocks + 1

    def forward(self, sample: Tensor, timestep: Tensor) -> Tensor:  # type: ignore[override]
        """Predict noise (or ``[noise, σ²]``) at the given timestep.

        Args:
            sample:   ``(B, in_channels, H, W)`` noisy image ``x_t``.
            timestep: ``(B,)`` or ``()`` int timestep tensor.  Scalar tensors
                are broadcast.

        Returns:
            ``(B, out_channels_effective, H, W)`` raw network output.
        """
        if timestep.ndim == 0:
            timestep = timestep.reshape((1,)).expand((int(sample.shape[0]),))
        t_emb = cast(Tensor, self.time_mlp(timestep))  # (B, time_dim)

        h = cast(Tensor, self.conv_in(sample))
        skips: list[Tensor] = [h]

        # Encoder.
        L = self._encoder_stages
        blocks_per = self._blocks_per_down_stage
        idx = 0
        for level in range(L):
            for _ in range(blocks_per):
                h = cast(Tensor, self.down_res[idx](h, t_emb))
                h = cast(Tensor, self.down_attn[idx](h))
                skips.append(h)
                idx += 1
            if level != L - 1:
                h = cast(Tensor, self.down_sample[level](h))
                skips.append(h)
            # last stage: down_sample slot holds an Identity; do nothing.

        # Middle.
        h = cast(Tensor, self.mid_block1(h, t_emb))
        h = cast(Tensor, self.mid_attn(h))
        h = cast(Tensor, self.mid_block2(h, t_emb))

        # Decoder.
        blocks_up = self._blocks_per_up_stage
        idx = 0
        for level in range(L - 1, -1, -1):
            for _ in range(blocks_up):
                skip = skips.pop()
                h = lucid.cat([h, skip], dim=1)
                h = cast(Tensor, self.up_res[idx](h, t_emb))
                h = cast(Tensor, self.up_attn[idx](h))
                idx += 1
            if level != 0:
                # up_sample list has one entry per stage; index ``L - 1 - level``
                # is the stage's upsample (skip for level 0).
                h = cast(Tensor, self.up_sample[L - 1 - level](h))

        h = cast(Tensor, self.norm_out(h))
        h = F.silu(h)
        return cast(Tensor, self.conv_out(h))


# ─────────────────────────────────────────────────────────────────────────────
# Bare DDPM model — wraps the U-Net, returns DiffusionModelOutput
# ─────────────────────────────────────────────────────────────────────────────


class DDPMModel(PretrainedModel, DiffusionMixin):
    """DDPM trunk + :class:`DiffusionMixin` for sampling.

    Returns a :class:`DiffusionModelOutput` whose ``sample`` field is the
    raw network output (interpreted per ``config.prediction_type``).  When
    ``config.learn_sigma`` is ``True`` the channels are
    ``2 * in_channels`` — first half is the mean prediction, second half
    the (raw) variance prediction; downstream code splits if needed.
    """

    config_class: ClassVar[type[DDPMConfig]] = DDPMConfig
    base_model_prefix: ClassVar[str] = "ddpm"

    def __init__(self, config: DDPMConfig) -> None:
        super().__init__(config)
        self.unet = DDPMUNet(config)

    def forward(  # type: ignore[override]
        self, sample: Tensor, timestep: Tensor
    ) -> DiffusionModelOutput:
        out = cast(Tensor, self.unet(sample, timestep))
        return DiffusionModelOutput(sample=out)


# ─────────────────────────────────────────────────────────────────────────────
# Task wrapper — training loss
# ─────────────────────────────────────────────────────────────────────────────


class DDPMForImageGeneration(PretrainedModel, DiffusionMixin):
    """DDPM + training loss + :meth:`DiffusionMixin.generate` for inference.

    Training:
        * ``forward(sample, timestep, noise=None)`` returns the raw network
          output.
        * ``forward(sample, timestep, noise=None, target=None)`` (target =
          ground-truth noise for ``epsilon`` parameterisation) also fills the
          ``loss`` field with the MSE loss of the paper's simple objective
          (Ho 2020 Eq. 14).

    Sampling:
        * Use :meth:`generate` from :class:`DiffusionMixin` — pass any
          :class:`Scheduler` (``DDPMScheduler`` for ancestral sampling).
    """

    config_class: ClassVar[type[DDPMConfig]] = DDPMConfig
    base_model_prefix: ClassVar[str] = "ddpm"

    def __init__(self, config: DDPMConfig) -> None:
        super().__init__(config)
        self.unet = DDPMUNet(config)
        self._in_channels = config.in_channels
        self._learn_sigma = config.learn_sigma
        if config.learn_sigma:
            # Improved-DDPM (Nichol & Dhariwal 2021) trains the learned
            # variance branch with a hybrid L_simple + λ·L_vlb objective.
            # We currently only emit the L_simple term, so silently learning
            # the variance head would produce a model that *can't* be used
            # for sampling with the predicted variance.  Refuse outright
            # rather than ship a half-implementation.
            raise NotImplementedError(
                "DDPM learn_sigma=True is not yet supported — Improved-DDPM "
                "requires the hybrid L_simple + L_vlb loss (Nichol & Dhariwal "
                "2021 §3.1) which Lucid does not implement yet."
            )

    def _split_output(self, raw: Tensor) -> tuple[Tensor, Tensor | None]:
        """When ``learn_sigma=True`` the network emits ``2·in_channels`` —
        split into ``(mean_pred, log_variance_pred)``."""
        if not self._learn_sigma:
            return raw, None
        C = int(raw.shape[1])
        half = C // 2
        return raw[:, :half], raw[:, half:]

    def forward(  # type: ignore[override]
        self,
        sample: Tensor,
        timestep: Tensor,
        target: Tensor | None = None,
    ) -> DiffusionModelOutput:
        raw = cast(Tensor, self.unet(sample, timestep))
        mean_pred, _logvar_pred = self._split_output(raw)

        loss: Tensor | None = None
        if target is not None:
            # Ho 2020 §3.2 "simple" objective — MSE on ε (or x_0 / v,
            # depending on ``prediction_type``; caller decides what's in
            # ``target``).
            diff = (mean_pred - target) ** 2
            loss = diff.mean()

        return DiffusionModelOutput(sample=mean_pred, loss=loss)
