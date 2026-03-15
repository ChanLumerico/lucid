from dataclasses import dataclass, field
from typing import Literal, Sequence

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.models.vision import unet as unet

__all__ = [
    "AttentionUNetGateConfig",
    "AttentionUNetConfig",
    "AttentionUNet2d",
    "AttentionUNet3d",
]


_AttentionGateMode = Literal["additive"]
_AttentionGateAct = Literal["relu"]
_AttentionCoeffAct = Literal["sigmoid"]
_AttentionResample = Literal["bilinear", "trilinear", "nearest"]


@dataclass
class AttentionUNetGateConfig:
    enabled: bool = True

    mode: _AttentionGateMode = "additive"
    gate_activation: _AttentionGateAct = "relu"
    attention_activation: _AttentionCoeffAct = "sigmoid"
    use_grid_attention: bool = True

    inter_channels: int | tuple[int, ...] | list[int] | None = None
    attention_channels: int = 1

    resample_mode: _AttentionResample = "bilinear"
    gate_on_skips: tuple[bool, ...] | list[bool] | None = None

    skip_low_level_gates: bool = False
    use_multi_scale_gating: bool = True

    project_skip_with_1x1: bool = True
    project_gating_with_1x1: bool = True

    init_pass_through: bool = True

    def __post_init__(self) -> None:
        if self.attention_channels <= 0:
            raise ValueError("attention_channels must be greater than 0.")

        if isinstance(self.inter_channels, list):
            self.inter_channels = tuple(self.inter_channels)
        if isinstance(self.inter_channels, tuple):
            if len(self.inter_channels) == 0:
                raise ValueError("inter_channels must not be empty.")
            if any(ch <= 0 for ch in self.inter_channels):
                raise ValueError("inter_channels values must be positive.")

        if self.gate_on_skips is not None:
            self.gate_on_skips = tuple(self.gate_on_skips)
            if len(self.gate_on_skips) == 0:
                raise ValueError("gate_on_skips must not be empty.")


@dataclass
class AttentionUNetConfig(unet.UNetConfig):
    block: Literal["basic"] = "basic"
    norm: unet._UNetNormName = "batch"
    act: unet._UNetActName = "relu"

    skip_merge: Literal["concat"] = "concat"
    downsample_mode: unet._UNetDownsample = "maxpool"
    upsample_mode: unet._UNetUpsample = "bilinear"

    deep_supervision: bool = True
    attention: AttentionUNetGateConfig = field(default_factory=AttentionUNetGateConfig)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.attention, AttentionUNetGateConfig):
            raise TypeError("attention must be an AttentionUNetGateConfig.")

        decoder_depth = len(self.decoder_stages)

        if self.attention.gate_on_skips is None:
            if decoder_depth == 0:
                self.attention.gate_on_skips = tuple()
            elif self.attention.skip_low_level_gates:
                self.attention.gate_on_skips = (True,) * decoder_depth
            else:
                self.attention.gate_on_skips = (True,) * max(decoder_depth - 1, 0) + (
                    False,
                )
        else:
            if len(self.attention.gate_on_skips) != decoder_depth:
                raise ValueError(
                    "attention.gate_on_skips must have 'len(decoder_stages)' entries."
                )

        if isinstance(self.attention.inter_channels, tuple):
            num_gated = sum(self.attention.gate_on_skips)
            if len(self.attention.inter_channels) not in (decoder_depth, num_gated):
                raise ValueError(
                    "attention.inter_channels must match decoder depth or "
                    "the number of enabled gates."
                )

        if self.block != "basic":
            raise ValueError("Attention U-Net requires block='basic'")
        if self.skip_merge != "concat":
            raise ValueError("Attention U-Net requires skip_merge='concat'")
        if self.attention.mode != "additive":
            raise ValueError("Attention U-Net requires additive gates")
        if self.attention.attention_activation != "sigmoid":
            raise ValueError("Attention U-Net requires sigmoid gate coefficients")
        if not self.attention.use_grid_attention:
            raise ValueError("Attention U-Net requires grid attention")


class _IdentityAttentionGate(nn.Module):
    def forward(self, skip: Tensor, *args, **kwargs) -> Tensor:
        return skip


class _AttentionGate2d(nn.Module):
    def __init__(
        self,
        skip_channels: int,
        gating_channels: int,
        inter_channels: int,
        gate_config: AttentionUNetGateConfig,
        *,
        align_corners: bool,
    ) -> None:
        super().__init__()
        self.gate_config = gate_config
        self.align_corners = align_corners

        gate_bias = True
        self.theta_x = (
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=gate_bias)
            if gate_config.project_skip_with_1x1
            else nn.Identity()
        )
        self.phi_g = (
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, bias=gate_bias)
            if gate_config.project_gating_with_1x1
            else nn.Identity()
        )
        self.psi = nn.Conv2d(
            inter_channels, gate_config.attention_channels, kernel_size=1, bias=True
        )

        if gate_config.gate_activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(
                f"Unsupported gate activation: '{gate_config.gate_activation}'"
            )

        if gate_config.attention_activation == "sigmoid":
            self.coeff_activation = nn.Sigmoid()
        else:
            raise ValueError("Attention U-Net requries sigmoid attention coefficients.")

        self.alpha_proj = (
            nn.Conv2d(
                gate_config.attention_channels, skip_channels, kernel_size=1, bias=True
            )
            if gate_config.attention_channels != 1
            else None
        )

        if gate_config.init_pass_through:
            nn.init.constant(self.psi.weight, 0.0)
            if self.psi.bias is not None:
                nn.init.constant(self.psi.bias, 2.0)

            if self.alpha_proj is not None:
                nn.init.constant(self.alpha_proj.weight, 0.0)
                if self.alpha_proj.bias is not None:
                    nn.init.constant(self.alpha_proj.bias, 1.0)

    def _resize(
        self, x: Tensor, size: tuple[int, int], mode: _AttentionResample | None = None
    ) -> Tensor:
        mode = self.gate_config.resample_mode if mode is None else mode
        return F.interpolate(
            x,
            size=size,
            mode=mode,
            align_corners=self.align_corners if mode == "bilinear" else False,
        )

    def forward(self, skip: Tensor, gating: Tensor) -> Tensor:
        theta_x = self.theta_x(skip)
        phi_g = self.phi_g(gating)

        if theta_x.shape[1] != phi_g.shape[1]:
            raise ValueError(
                "Projected skip and gating features must have same channel width."
            )

        if self.gate_config.use_grid_attention:
            phi_g = self._resize(phi_g, theta_x.shape[-2:])
        else:
            phi_g = lucid.mean(phi_g, axis=(-2, -1), keepdims=True)
            phi_g = lucid.broadcast_to(phi_g, theta_x.shape)

        f = self.activation(theta_x + phi_g)
        alpha = self.coeff_activation(self.psi(f))

        if self.alpha_proj is not None:
            alpha = self.alpha_proj(alpha)

        alpha = self._resize(alpha, skip.shape[-2:])
        return alpha * skip


class _AttentionGate3d(nn.Module):
    def __init__(
        self,
        skip_channels: int,
        gating_channels: int,
        inter_channels: int,
        gate_config: AttentionUNetGateConfig,
        *,
        align_corners: bool,
    ) -> None:
        super().__init__()
        self.gate_config = gate_config
        self.align_corners = align_corners

        gate_bias = True
        self.theta_x = (
            nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=gate_bias)
            if gate_config.project_skip_with_1x1
            else nn.Identity()
        )
        self.phi_g = (
            nn.Conv3d(gating_channels, inter_channels, kernel_size=1, bias=gate_bias)
            if gate_config.project_gating_with_1x1
            else nn.Identity()
        )
        self.psi = nn.Conv3d(
            inter_channels, gate_config.attention_channels, kernel_size=1, bias=True
        )

        if gate_config.gate_activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(
                f"Unsupported gate activation: '{gate_config.gate_activation}'"
            )

        if gate_config.attention_activation == "sigmoid":
            self.coeff_activation = nn.Sigmoid()
        else:
            raise ValueError("Attention U-Net requires sigmoid attention coefficients.")

        self.alpha_proj = (
            nn.Conv3d(
                gate_config.attention_channels, skip_channels, kernel_size=1, bias=True
            )
            if gate_config.attention_channels != 1
            else None
        )

        if gate_config.init_pass_through:
            nn.init.constant(self.psi.weight, 0.0)
            if self.psi.bias is not None:
                nn.init.constant(self.psi.bias, 2.0)

            if self.alpha_proj is not None:
                nn.init.constant(self.alpha_proj.weight, 0.0)
                if self.alpha_proj.bias is not None:
                    nn.init.constant(self.alpha_proj.bias, 1.0)

    def _resize(
        self,
        x: Tensor,
        size: tuple[int, int, int],
        mode: _AttentionResample | None = None,
    ) -> Tensor:
        mode = self.gate_config.resample_mode if mode is None else mode
        if mode == "bilinear":
            mode = "trilinear"
        return F.interpolate(
            x,
            size=size,
            mode=mode,
            align_corners=self.align_corners if mode == "trilinear" else False,
        )

    def forward(self, skip: Tensor, gating: Tensor) -> Tensor:
        theta_x = self.theta_x(skip)
        phi_g = self.phi_g(gating)

        if theta_x.shape[1] != phi_g.shape[1]:
            raise ValueError(
                "Projected skip and gating features must have same channel width."
            )

        if self.gate_config.use_grid_attention:
            phi_g = self._resize(phi_g, theta_x.shape[-3:])
        else:
            phi_g = lucid.mean(phi_g, axis=(-3, -2, -1), keepdims=True)
            phi_g = lucid.broadcast_to(phi_g, theta_x.shape)

        f = self.activation(theta_x + phi_g)
        alpha = self.coeff_activation(self.psi(f))

        if self.alpha_proj is not None:
            alpha = self.alpha_proj(alpha)

        alpha = self._resize(alpha, skip.shape[-3:])
        return alpha * skip


class AttentionUNet2d(unet.UNet2d):
    def __init__(self, config: AttentionUNetConfig) -> None:
        super().__init__(config)
        self.config = config

        if not isinstance(config, AttentionUNetConfig):
            raise TypeError("config must be an AttentionUNetConfig.")
        if not isinstance(config.attention, AttentionUNetGateConfig):
            raise TypeError("config.attention must be an AttentionUNetGateConfig.")

        self.attention_gates: Sequence[_IdentityAttentionGate | _AttentionGate2d] = (
            nn.ModuleList()
        )

        decoder_depth = len(config.decoder_stages)
        gate_on_skips = config.attention.gate_on_skips
        if gate_on_skips is None:
            raise ValueError("attention.gate_on_skips must be resolved in config.")

        skip_channels = tuple(
            stage.channels for stage in reversed(config.encoder_stages[:-1])
        )
        gating_channels = (
            config.bottleneck.channels,
            *(stage.channels for stage in config.decoder_stages[:-1]),
        )

        for idx in range(decoder_depth):
            enabled = config.attention.enabled and gate_on_skips[idx]
            if not enabled:
                self.attention_gates.append(_IdentityAttentionGate())
                continue

            self.attention_gates.append(
                _AttentionGate2d(
                    skip_channels=skip_channels[idx],
                    gating_channels=gating_channels[idx],
                    inter_channels=self._resolve_gate_inter_channels(
                        idx, skip_channels[idx], gating_channels[idx]
                    ),
                    gate_config=config.attention,
                    align_corners=config.align_corners,
                )
            )

    def _resolve_gate_inter_channels(
        self, idx: int, skip_channels: int, gating_channels: int
    ) -> int:
        inter_channels = self.config.attention.inter_channels
        if inter_channels is None:
            return max(1, min(skip_channels, gating_channels) // 2)

        if isinstance(inter_channels, int):
            return inter_channels

        enabled_indices = [
            i
            for i, enabled in enumerate(self.config.attention.gate_on_skips)
            if enabled
        ]

        if len(inter_channels) == len(self.config.decoder_stages):
            return inter_channels[idx]

        enabled_pos = enabled_indices.index(idx)
        return inter_channels[enabled_pos]

    def forward(self, x: Tensor) -> Tensor | dict[str, Tensor | list[Tensor]]:
        input_size = x.shape[-2:]
        x = self.stem(x)

        skips: list[Tensor] = []
        for idx, stage in enumerate(self.encoder):
            x = stage(x)
            skips.append(x)

            if idx < len(self.downsamplers):
                x = self.downsamplers[idx](x)

        x = self.bottleneck(x)

        aux_outputs: list[Tensor] = []
        decode_skips = list(reversed(skips[:-1]))

        for idx, (gate, upsample, stage, skip) in enumerate(
            zip(self.attention_gates, self.upsamplers, self.decoder, decode_skips)
        ):
            gated_skip = gate(skip, x)
            x = upsample(x, target_size=gated_skip.shape[-2:])
            x = self._merge_skip(x, gated_skip)
            x = stage(x)

            if self.config.deep_supervision and idx < len(self.aux_heads):
                aux = self.aux_heads[idx](x)
                aux_outputs.append(self._resize_to_input(aux, input_size))

        out = self.head(x)
        out = self._resize_to_input(out, input_size)

        if self.config.deep_supervision:
            return {"out": out, "aux": aux_outputs}

        return out


class AttentionUNet3d(unet.UNet3d):
    def __init__(self, config: AttentionUNetConfig) -> None:
        super().__init__(config)
        self.config = config

        if not isinstance(config, AttentionUNetConfig):
            raise TypeError("config must be an AttentionUNetConfig.")
        if not isinstance(config.attention, AttentionUNetGateConfig):
            raise TypeError("config.attention must be an AttentionUNetGateConfig.")

        self.attention_gates: Sequence[_IdentityAttentionGate | _AttentionGate3d] = (
            nn.ModuleList()
        )

        decoder_depth = len(config.decoder_stages)
        gate_on_skips = config.attention.gate_on_skips
        if gate_on_skips is None:
            raise ValueError("attention.gate_on_skips must be resolved in config.")

        skip_channels = tuple(
            stage.channels for stage in reversed(config.encoder_stages[:-1])
        )
        gating_channels = (
            config.bottleneck.channels,
            *(stage.channels for stage in config.decoder_stages[:-1]),
        )

        for idx in range(decoder_depth):
            enabled = config.attention.enabled and gate_on_skips[idx]
            if not enabled:
                self.attention_gates.append(_IdentityAttentionGate())
                continue

            self.attention_gates.append(
                _AttentionGate3d(
                    skip_channels=skip_channels[idx],
                    gating_channels=gating_channels[idx],
                    inter_channels=self._resolve_gate_inter_channels(
                        idx, skip_channels[idx], gating_channels[idx]
                    ),
                    gate_config=config.attention,
                    align_corners=config.align_corners,
                )
            )

    def _resolve_gate_inter_channels(
        self, idx: int, skip_channels: int, gating_channels: int
    ) -> int:
        inter_channels = self.config.attention.inter_channels
        if inter_channels is None:
            return max(1, min(skip_channels, gating_channels) // 2)

        if isinstance(inter_channels, int):
            return inter_channels

        enabled_indices = [
            i
            for i, enabled in enumerate(self.config.attention.gate_on_skips)
            if enabled
        ]

        if len(inter_channels) == len(self.config.decoder_stages):
            return inter_channels[idx]

        enabled_pos = enabled_indices.index(idx)
        return inter_channels[enabled_pos]

    def forward(self, x: Tensor) -> Tensor | dict[str, Tensor | list[Tensor]]:
        input_size = x.shape[-3:]
        x = self.stem(x)

        skips: list[Tensor] = []
        for idx, stage in enumerate(self.encoder):
            x = stage(x)
            skips.append(x)

            if idx < len(self.downsamplers):
                x = self.downsamplers[idx](x)

        x = self.bottleneck(x)

        aux_outputs: list[Tensor] = []
        decode_skips = list(reversed(skips[:-1]))

        for idx, (gate, upsample, stage, skip) in enumerate(
            zip(self.attention_gates, self.upsamplers, self.decoder, decode_skips)
        ):
            gated_skip = gate(skip, x)
            x = upsample(x, target_size=gated_skip.shape[-3:])
            x = self._merge_skip(x, gated_skip)
            x = stage(x)

            if self.config.deep_supervision and idx < len(self.aux_heads):
                aux = self.aux_heads[idx](x)
                aux_outputs.append(self._resize_to_input(aux, input_size))

        out = self.head(x)
        out = self._resize_to_input(out, input_size)

        if self.config.deep_supervision:
            return {"out": out, "aux": aux_outputs}

        return out
