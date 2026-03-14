from dataclasses import dataclass, field
from typing import Literal

__all__ = ["UNetConfig"]


@dataclass
class _UNetStageConfig:
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

    encoder_stages: tuple[_UNetStageConfig, ...] | list[_UNetStageConfig]
    decoder_stages: tuple[_UNetStageConfig, ...] | list[_UNetStageConfig] | None = None
    bottleneck: _UNetStageConfig | None = None

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
                _UNetStageConfig(
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
            self.bottleneck = _UNetStageConfig(
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
        **kwargs
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
            _UNetStageConfig(channels=ch, num_blocks=nb)
            for ch, nb in zip(channels, num_blocks)
        )
        return cls(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_stages=encoder_stages,
            **kwargs
        )
