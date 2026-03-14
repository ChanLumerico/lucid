import math

from dataclasses import dataclass, replace
from typing import Literal

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor

__all__ = ["UNetConfig", "UNetStageConfig", "UNet", "ResUNet"]


@dataclass
class UNetStageConfig:
    channels: int
    num_blocks: int = 2
    kernel_size: int = 3
    dilation: int = 1
    use_attention: bool = False
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.channels <= 0:
            raise ValueError("channels must be greater than 0.")
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be greater than 0.")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if self.dilation <= 0:
            raise ValueError("dilation must be greater than 0.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")


_UNetNormName = Literal["batch", "group", "instance", "none"]
_UNetActName = Literal["relu", "leaky_relu", "gelu", "silu"]

_UNetBlockName = Literal["basic", "res", "convnext"]
_UNetSkipMerge = Literal["concat", "add"]

_UNetDownsample = Literal["conv", "maxpool", "avgpool"]
_UNetUpsample = Literal["transpose", "bilinear", "nearest"]


@dataclass
class UNetConfig:
    in_channels: int
    out_channels: int

    encoder_stages: tuple[UNetStageConfig, ...] | list[UNetStageConfig]
    decoder_stages: tuple[UNetStageConfig, ...] | list[UNetStageConfig] | None = None
    bottleneck: UNetStageConfig | None = None

    block: _UNetBlockName = "basic"
    norm: _UNetNormName = "batch"
    act: _UNetActName = "relu"

    skip_merge: _UNetSkipMerge = "concat"
    downsample_mode: _UNetDownsample = "conv"
    upsample_mode: _UNetUpsample = "transpose"

    stem_channels: int | None = None
    final_kernel_size: int = 1

    deep_supervision: bool = False
    align_corners: bool = False
    bias: bool | None = None

    def __post_init__(self) -> None:
        self.encoder_stages = tuple(self.encoder_stages)
        if len(self.encoder_stages) == 0:
            raise ValueError("encoder_stages must contain at least one stage.")

        if self.decoder_stages is None:
            self.decoder_stages = tuple(
                UNetStageConfig(
                    channels=stage.channels,
                    num_blocks=stage.num_blocks,
                    kernel_size=stage.kernel_size,
                    dilation=stage.dilation,
                    use_attention=stage.use_attention,
                    dropout=stage.dropout,
                )
                for stage in reversed(self.encoder_stages[:-1])
            )
        else:
            self.decoder_stages = tuple(self.decoder_stages)

        if self.bottleneck is None:
            self.bottleneck = UNetStageConfig(
                channels=self.encoder_stages[-1].channels,
                num_blocks=2,
            )

        if self.in_channels <= 0:
            raise ValueError("in_channels must be greater than 0.")
        if self.out_channels <= 0:
            raise ValueError("out_channels must be greater than 0.")
        if self.stem_channels is not None and self.stem_channels <= 0:
            raise ValueError("stem_channels must be greater than 0.")
        if self.final_kernel_size <= 0 or self.final_kernel_size % 2 == 0:
            raise ValueError("final_kernel_size must be a positive odd integer.")

        expected_decoder_depth = max(len(self.encoder_stages) - 1, 0)
        if len(self.decoder_stages) != expected_decoder_depth:
            raise ValueError(
                "decoder_stages must have 'len(encoder_stages) - 1' stages."
            )

        if self.skip_merge == "add":
            for enc_stage, dec_stage in zip(
                reversed(self.encoder_stages[:-1]), self.decoder_stages
            ):
                if enc_stage.channels != dec_stage.channels:
                    raise ValueError(
                        "skip_merge='add' requires matching encoder/decoder channels."
                    )

        if self.bias is None:
            self.bias = self.norm == "none"

    @classmethod
    def from_channels(
        cls,
        in_channels: int,
        out_channels: int,
        channels: tuple[int, ...] | list[int],
        num_blocks: int | tuple[int, ...] | list[int] = 2,
        **kwargs,
    ) -> UNetConfig:
        channels = tuple(channels)
        if isinstance(num_blocks, int):
            num_blocks = (num_blocks,) * len(channels)
        else:
            num_blocks = tuple(num_blocks)

        if len(channels) == 0 or len(channels) != len(num_blocks):
            raise ValueError(
                "channels and num_blocks must have the same non-zero length."
            )

        encoder_stages = tuple(
            UNetStageConfig(channels=ch, num_blocks=nb)
            for ch, nb in zip(channels, num_blocks)
        )
        return cls(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_stages=encoder_stages,
            **kwargs,
        )


def _same_padding(kernel_size: int, dilation: int = 1) -> int:
    return dilation * (kernel_size - 1) // 2


def _make_norm(norm: _UNetNormName, channels: int) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels)
    if norm == "group":
        num_groups = min(32, channels)
        while channels % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups, channels)
    if norm == "none":
        return nn.Identity()

    raise ValueError(f"Unsupported norm: '{norm}'")


def _make_act(act: _UNetActName) -> nn.Module:
    if act == "relu":
        return nn.ReLU()
    if act == "leaky_relu":
        return nn.LeakyReLU()
    if act == "gelu":
        return nn.GELU()
    if act == "silu":
        return nn.Swish()
    raise ValueError(f"Unsupported activation: '{act}'")


class _BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        norm: _UNetNormName,
        act: _UNetActName,
        dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()
        padding = _same_padding(kernel_size, dilation)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias,
            ),
            _make_norm(norm, out_channels),
            _make_act(act),
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class _ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        norm: _UNetNormName,
        act: _UNetActName,
        dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()
        padding = _same_padding(kernel_size, dilation)

        self.norm1 = _make_norm(norm, in_channels)
        self.act1 = _make_act(act)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        self.norm2 = _make_norm(norm, out_channels)
        self.act2 = _make_act(act)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.shortcut(x)


class _SelfAttention2d(nn.Module):
    def __init__(self, channels: int, *, norm: _UNetNormName, bias: bool) -> None:
        super().__init__()
        self.norm = _make_norm(norm, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, axis=1)

        q = q.reshape(b, c, h * w)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)

        attn = (q.mT @ k) / math.sqrt(c)
        attn = F.softmax(attn, axis=-1)

        out = v @ attn.mT
        out = out.reshape(b, c, h, w)

        return x + self.proj(out)


class _UNetStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stage_config: UNetStageConfig,
        *,
        block: _UNetBlockName,
        norm: _UNetNormName,
        act: _UNetActName,
        bias: bool,
    ) -> None:
        super().__init__()
        if block == "basic":
            block_cls = _BasicBlock
        elif block == "res":
            block_cls = _ResBlock
        else:
            raise ValueError(
                f"Unsupported block type: {block}.",
                "This implementation supports only 'basic' and 'res'.",
            )

        layers = []
        cur_channels = in_channels
        for _ in range(stage_config.num_blocks):
            layers.append(
                block_cls(
                    cur_channels,
                    stage_config.channels,
                    kernel_size=stage_config.kernel_size,
                    dilation=stage_config.dilation,
                    norm=norm,
                    act=act,
                    dropout=stage_config.dropout,
                    bias=bias,
                )
            )
            cur_channels = stage_config.channels

        self.blocks = nn.Sequential(*layers)
        self.attn = (
            _SelfAttention2d(stage_config.channels, norm=norm, bias=bias)
            if stage_config.use_attention
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        x = self.attn(x)
        return x


class _Downsample(nn.Module):
    def __init__(self, channels: int, mode: _UNetDownsample, *, bias: bool) -> None:
        super().__init__()
        if mode == "conv":
            self.op = nn.Conv2d(
                channels, channels, kernel_size=3, stride=2, padding=1, bias=bias
            )
        elif mode == "maxpool":
            self.op = nn.MaxPool2d(kernel_size=2, stride=2)
        elif mode == "avgpool":
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError(f"Unsupported downsample mode: '{mode}'")

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


class _Upsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: _UNetUpsample,
        *,
        align_corners: bool,
        bias: bool,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

        if mode == "transpose":
            self.op = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2, bias=bias
            )
        elif mode in {"bilinear", "nearest"}:
            self.op = nn.Sequential(
                nn.Upsample(
                    scale_factor=2,
                    mode=mode,
                    align_corners=align_corners if mode == "bilinear" else False,
                ),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            )
        else:
            raise ValueError(f"Unsupported upsample mode: '{mode}'")

    def forward(self, x: Tensor, target_size: tuple[int, int] | None = None) -> Tensor:
        x = self.op(x)
        if target_size is not None and target_size != x.shape[-2:]:
            x = F.interpolate(
                x, size=target_size, mode="bilinear", align_corners=self.align_corners
            )
        return x


class UNet(nn.Module):
    def __init__(self, config: UNetConfig) -> None:
        super().__init__()
        self.config = config

        stem_channels = (
            config.stem_channels
            if config.stem_channels is not None
            else config.encoder_stages[0].channels
        )
        head_padding = _same_padding(config.final_kernel_size)

        self.stem = nn.Sequential(
            nn.Conv2d(
                config.in_channels,
                stem_channels,
                kernel_size=3,
                padding=1,
                bias=config.bias,
            ),
            _make_norm(config.norm, stem_channels),
            _make_act(config.act),
        )
        self.encoder = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        cur_channels = stem_channels
        for idx, stage_cfg in enumerate(config.encoder_stages):
            self.encoder.append(
                _UNetStage(
                    cur_channels,
                    stage_cfg,
                    block=config.block,
                    norm=config.norm,
                    act=config.act,
                    bias=config.bias,
                )
            )
            cur_channels = stage_cfg.channels

            if idx < len(config.encoder_stages) - 1:
                self.downsamplers.append(
                    _Downsample(cur_channels, config.downsample_mode, bias=config.bias)
                )

        self.bottleneck = _UNetStage(
            cur_channels,
            config.bottleneck,
            block=config.block,
            norm=config.norm,
            act=config.act,
            bias=config.bias,
        )
        cur_channels = config.bottleneck.channels

        self.upsampleers = nn.ModuleList()
        self.decoder = nn.ModuleList()

        skip_channels = [stage.channels for stage in config.encoder_stages[:-1]]
        reversed_skip_channels = list(reversed(skip_channels))

        for dec_cfg, skip_ch in zip(config.decoder_stages, reversed_skip_channels):
            self.upsampleers.append(
                _Upsample(
                    cur_channels,
                    dec_cfg.channels,
                    config.upsample_mode,
                    align_corners=config.align_corners,
                    bias=config.bias,
                )
            )

            stage_in_channels = (
                dec_cfg.channels + skip_ch
                if config.skip_merge == "concat"
                else dec_cfg.channels
            )

            self.decoder.append(
                _UNetStage(
                    stage_in_channels,
                    dec_cfg,
                    block=config.block,
                    norm=config.norm,
                    act=config.act,
                    bias=config.bias,
                )
            )
            cur_channels = dec_cfg.channels

        self.head = nn.Conv2d(
            cur_channels,
            config.out_channels,
            kernel_size=config.final_kernel_size,
            padding=head_padding,
            bias=True,
        )

        self.aux_heads = nn.ModuleList()
        if config.deep_supervision:
            for dec_cfg in config.decoder_stages[:-1]:
                self.aux_heads.append(
                    nn.Conv2d(
                        dec_cfg.channels, config.out_channels, kernel_size=1, bias=True
                    )
                )

    def _merge_skip(self, x: Tensor, skip: Tensor) -> Tensor:
        if self.config.skip_merge == "concat":
            return lucid.concatenate([x, skip], axis=1)
        if self.config.skip_merge == "add":
            return x + skip
        raise ValueError(f"Unsupported skip_merge: '{self.config.skip_merge}'")

    def _resize_to_input(self, x: Tensor, input_size: tuple[int, int]) -> Tensor:
        if x.shape[-2:] == input_size:
            return x
        return F.interpolate(
            x, size=input_size, mode="bilinear", align_corners=self.config.align_corners
        )

    def forward(self, x: Tensor) -> Tensor | dict[str, Tensor]:
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

        for idx, (upsample, stage, skip) in enumerate(
            zip(self.upsampleers, self.decoder, decode_skips)
        ):
            x = upsample(x, target_size=skip.shape[-2:])
            x = self._merge_skip(x, skip)
            x = stage(x)

            if self.config.deep_supervision and idx < len(self.aux_heads):
                aux = self.aux_heads[idx](x)
                aux_outputs.append(self._resize_to_input(aux, input_size))

        out = self.head(x)
        out = self._resize_to_input(out, input_size)

        if self.config.deep_supervision:
            return {"out": out, "aux": aux_outputs}

        return out


class ResUNet(UNet):
    def __init__(self, config: UNetConfig) -> None:
        if config.block != "res":
            config = replace(config, block="res")
        super().__init__(config)
