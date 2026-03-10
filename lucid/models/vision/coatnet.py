from dataclasses import dataclass
from typing import Type

import lucid
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor

__all__ = [
    "CoAtNet",
    "CoAtNetConfig",
    "coatnet_0",
    "coatnet_1",
    "coatnet_2",
    "coatnet_3",
    "coatnet_4",
    "coatnet_5",
    "coatnet_6",
    "coatnet_7",
]


def _normalize_positive_int_sequence(
    values: tuple[int, ...] | list[int],
    name: str,
    *,
    expected_length: int | None = None,
) -> tuple[int, ...]:
    normalized = tuple(values)
    if expected_length is not None and len(normalized) != expected_length:
        raise ValueError(f"{name} must contain exactly {expected_length} values")
    if len(normalized) == 0:
        raise ValueError(f"{name} must contain at least one value")
    if any(not isinstance(value, int) or value <= 0 for value in normalized):
        raise ValueError(f"{name} values must be positive integers")
    return normalized


@dataclass
class CoAtNetConfig:
    img_size: tuple[int, int] | list[int]
    in_channels: int
    num_blocks: tuple[int, ...] | list[int]
    channels: tuple[int, ...] | list[int]
    num_classes: int = 1000
    num_heads: int = 32
    block_types: tuple[str, ...] | list[str] = ("C", "C", "T", "T")
    scaled_num_blocks: tuple[int, int] | list[int] | None = None
    scaled_channels: tuple[int, int] | list[int] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.img_size, (list, tuple)) or len(self.img_size) != 2:
            raise ValueError("img_size must contain exactly two values")
        self.img_size = tuple(self.img_size)
        if any(not isinstance(value, int) or value < 32 for value in self.img_size):
            raise ValueError(
                "img_size values must be integers greater than or equal to 32"
            )

        if self.in_channels <= 0:
            raise ValueError("in_channels must be greater than 0")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be greater than 0")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be greater than 0")

        self.num_blocks = _normalize_positive_int_sequence(
            self.num_blocks, "num_blocks", expected_length=5
        )
        self.channels = _normalize_positive_int_sequence(
            self.channels, "channels", expected_length=5
        )

        if not isinstance(self.block_types, (list, tuple)):
            raise TypeError("block_types must be a sequence")
        self.block_types = tuple(self.block_types)
        if len(self.block_types) != 4:
            raise ValueError("block_types must contain exactly 4 values")
        if any(block_type not in {"C", "T"} for block_type in self.block_types):
            raise ValueError("block_types values must be 'C' or 'T'")

        if (self.scaled_num_blocks is None) ^ (self.scaled_channels is None):
            raise ValueError(
                "scaled_num_blocks and scaled_channels must both be provided together"
            )

        if self.scaled_num_blocks is not None:
            self.scaled_num_blocks = _normalize_positive_int_sequence(
                self.scaled_num_blocks,
                "scaled_num_blocks",
                expected_length=2,
            )
            self.scaled_channels = _normalize_positive_int_sequence(
                self.scaled_channels,
                "scaled_channels",
                expected_length=2,
            )


def conv_3x3_bn(
    in_channels: int, out_channels: int, downsample: bool = False, **kwargs
) -> nn.Sequential:
    stride = 1 if not downsample else 2
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
    )


class _PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module, norm: Type[nn.Module]) -> Tensor:
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.fn(self.norm(x), **kwargs)


class _SEBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, reduction: float = 0.25
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(out_channels, int(in_channels * reduction), bias=False),
            nn.GELU(),
            nn.Linear(int(in_channels * reduction), out_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.avgpool(x)
        y = y.reshape(y.shape[0], -1)
        y = self.fc(y).unsqueeze(axis=(-1, -2))

        return x * y


class _FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ff(x)


class _MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = False,
        expansion: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.downsample = downsample
        stride = 1 if not self.downsample else 2
        hidden_dim = int(in_channels * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        if expansion == 1:
            conv = nn.Sequential(
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, hidden_dim, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                _SEBlock(in_channels, hidden_dim),
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.conv = _PreNorm(in_channels, conv, nn.BatchNorm2d)

    def forward(self, x: Tensor) -> Tensor:
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)


class _Attention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: tuple[int, int],
        num_heads: int = 8,
        dim_head: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * num_heads
        project_out = not (num_heads == 1 and dim_head == in_channels)

        self.ih, self.iw = img_size
        self.num_heads = num_heads
        self.scale = dim_head**0.5

        self.relative_bias_table = nn.Parameter(
            lucid.zeros((2 * self.ih - 1) * (2 * self.iw - 1), num_heads)
        )

        coords = lucid.stack(
            lucid.meshgrid(lucid.arange(self.ih), lucid.arange(self.iw))
        )
        coords = coords.reshape(coords.shape[0], -1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1

        relative_coords = lucid.einops.rearrange(relative_coords, "c h w -> h w c")
        relative_index = relative_coords.sum(axis=-1).flatten().unsqueeze(axis=1)

        self.relative_index: nn.Buffer
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(axis=-1)
        self.to_qkv = nn.Linear(in_channels, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, out_channels), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        qkv = self.to_qkv(x).chunk(3, axis=-1)
        q, k, v = map(
            lambda tensor: lucid.einops.rearrange(
                tensor,
                "b n (h d) -> b h n d",
                h=self.num_heads,
                d=tensor.shape[-1] // self.num_heads,
            ),
            qkv,
        )
        dots = (q @ k.mT) * self.scale

        rel_idx = self.relative_index.repeat(self.num_heads, axis=1).to(x.device)
        col_indices = (
            (
                lucid.arange(self.relative_bias_table.shape[1])
                .unsqueeze(axis=0)
                .broadcast_to(rel_idx.shape)
            )
            .astype(lucid.Int)
            .to(x.device)
        )

        relative_bias = self.relative_bias_table[
            rel_idx.astype(lucid.Int), col_indices.astype(lucid.Int)
        ]
        relative_bias = lucid.einops.rearrange(
            relative_bias, "(h w) c -> c h w", h=self.ih * self.iw, w=self.ih * self.iw
        ).unsqueeze(axis=0)

        dots += relative_bias

        attn = self.attend(dots)
        out = attn @ v
        out = lucid.einops.rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        return out


class _Transformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: tuple[int, int],
        num_heads: int = 8,
        dim_head: int = 32,
        downsample: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(in_channels * 4)

        self.ih, self.iw = img_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        attn = _Attention(
            in_channels, out_channels, img_size, num_heads, dim_head, dropout
        )
        ff = _FeedForward(out_channels, hidden_dim, dropout)

        self.attn = nn.Sequential(
            nn.Rearrange("b c ih iw -> b (ih iw) c"),
            _PreNorm(in_channels, attn, nn.LayerNorm),
            nn.Rearrange("b (ih iw) c -> b c ih iw", ih=self.ih, iw=self.iw),
        )
        self.ff = nn.Sequential(
            nn.Rearrange("b c ih iw -> b (ih iw) c"),
            _PreNorm(out_channels, ff, nn.LayerNorm),
            nn.Rearrange("b (ih iw) c -> b c ih iw", ih=self.ih, iw=self.iw),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x += self.attn(x)
        x += self.ff(x)
        return x


class CoAtNet(nn.Module):
    def __init__(self, config: CoAtNetConfig) -> None:
        super().__init__()
        self.config = config
        block = {"C": _MBConv, "T": _Transformer}
        get_block = lambda i: block[config.block_types[i]]

        self.ih, self.iw = config.img_size
        self.channels = config.channels
        self.num_heads = config.num_heads

        self._do_scale = (
            config.scaled_num_blocks is not None and config.scaled_channels is not None
        )

        self.s0 = self._make_layer(
            0, conv_3x3_bn, config.in_channels, config.channels[0], config.num_blocks[0]
        )
        self.s1 = self._make_layer(
            1,
            get_block(0),
            config.channels[0],
            config.channels[1],
            config.num_blocks[1],
        )
        self.s2 = self._make_layer(
            2,
            get_block(1),
            config.channels[1],
            config.channels[2],
            config.num_blocks[2],
        )
        self.s3 = (
            self._make_layer(
                3,
                get_block(2),
                config.channels[2],
                config.channels[3],
                config.num_blocks[3],
            )
            if not self._do_scale
            else nn.Identity()
        )
        self.s4 = self._make_layer(
            4,
            get_block(3),
            config.channels[3] if not self._do_scale else config.scaled_channels[1],
            config.channels[4],
            config.num_blocks[4],
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(config.channels[-1], config.num_classes, bias=False)

        if self._do_scale:
            self._scale_s3(config.scaled_num_blocks, config.scaled_channels)

    def _make_layer(
        self,
        i: int,
        block: Type[nn.Module],
        in_channels: int,
        out_channels: int,
        depth: int,
        downsample_first: bool = True,
    ) -> nn.Sequential:
        img_size = (self.ih // (2 ** (i + 1)), self.iw // (2 ** (i + 1)))

        layers = nn.ModuleList()
        for block_index in range(depth):
            new_block = block(
                in_channels if block_index == 0 else out_channels,
                out_channels,
                img_size=img_size,
                num_heads=self.num_heads,
                downsample=downsample_first if block_index == 0 else False,
            )
            layers.append(new_block)

        return nn.Sequential(*layers)

    def _scale_s3(
        self,
        depths: tuple[int, int],
        channels: tuple[int, int],
    ) -> None:
        self.s3_tandem = nn.Sequential(
            self._make_layer(3, _MBConv, self.channels[2], channels[0], depths[0]),
            self._make_layer(
                4,
                _Transformer,
                channels[0],
                channels[1],
                depths[1],
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x) if not self._do_scale else self.s3_tandem(x)
        x = self.s4(x)

        x = self.pool(x).reshape(-1, x.shape[1])
        x = self.fc(x)

        return x


def _raise_for_locked_factory_kwargs(
    kwargs: dict[str, object],
    locked_fields: set[str],
    message: str,
) -> None:
    if locked_fields & kwargs.keys():
        raise TypeError(message)


def _build_coatnet_config(
    *,
    img_size: tuple[int, int],
    in_channels: int,
    num_blocks: tuple[int, ...],
    channels: tuple[int, ...],
    num_classes: int,
    **kwargs,
) -> CoAtNetConfig:
    return CoAtNetConfig(
        img_size=img_size,
        in_channels=in_channels,
        num_blocks=num_blocks,
        channels=channels,
        num_classes=num_classes,
        **kwargs,
    )


@register_model
def coatnet_0(num_classes: int = 1000, **kwargs) -> CoAtNet:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"img_size", "in_channels", "num_blocks", "channels"},
        "factory variants do not allow overriding preset img_size, in_channels, num_blocks, or channels",
    )
    config = _build_coatnet_config(
        img_size=(224, 224),
        in_channels=3,
        num_blocks=(2, 2, 3, 5, 2),
        channels=(64, 96, 192, 384, 768),
        num_classes=num_classes,
        **kwargs,
    )
    return CoAtNet(config)


@register_model
def coatnet_1(num_classes: int = 1000, **kwargs) -> CoAtNet:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"img_size", "in_channels", "num_blocks", "channels"},
        "factory variants do not allow overriding preset img_size, in_channels, num_blocks, or channels",
    )
    config = _build_coatnet_config(
        img_size=(224, 224),
        in_channels=3,
        num_blocks=(2, 2, 6, 14, 2),
        channels=(64, 96, 192, 384, 768),
        num_classes=num_classes,
        **kwargs,
    )
    return CoAtNet(config)


@register_model
def coatnet_2(num_classes: int = 1000, **kwargs) -> CoAtNet:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"img_size", "in_channels", "num_blocks", "channels"},
        "factory variants do not allow overriding preset img_size, in_channels, num_blocks, or channels",
    )
    config = _build_coatnet_config(
        img_size=(224, 224),
        in_channels=3,
        num_blocks=(2, 2, 6, 14, 2),
        channels=(128, 128, 256, 512, 1024),
        num_classes=num_classes,
        **kwargs,
    )
    return CoAtNet(config)


@register_model
def coatnet_3(num_classes: int = 1000, **kwargs) -> CoAtNet:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"img_size", "in_channels", "num_blocks", "channels"},
        "factory variants do not allow overriding preset img_size, in_channels, num_blocks, or channels",
    )
    config = _build_coatnet_config(
        img_size=(224, 224),
        in_channels=3,
        num_blocks=(2, 2, 6, 14, 2),
        channels=(192, 192, 384, 768, 1536),
        num_classes=num_classes,
        **kwargs,
    )
    return CoAtNet(config)


@register_model
def coatnet_4(num_classes: int = 1000, **kwargs) -> CoAtNet:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"img_size", "in_channels", "num_blocks", "channels"},
        "factory variants do not allow overriding preset img_size, in_channels, num_blocks, or channels",
    )
    config = _build_coatnet_config(
        img_size=(224, 224),
        in_channels=3,
        num_blocks=(2, 2, 12, 28, 2),
        channels=(192, 192, 384, 768, 1536),
        num_classes=num_classes,
        **kwargs,
    )
    return CoAtNet(config)


@register_model
def coatnet_5(num_classes: int = 1000, **kwargs) -> CoAtNet:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"img_size", "in_channels", "num_blocks", "channels", "num_heads"},
        "factory variants do not allow overriding preset img_size, in_channels, num_blocks, channels, or num_heads",
    )
    config = _build_coatnet_config(
        img_size=(224, 224),
        in_channels=3,
        num_blocks=(2, 2, 12, 28, 2),
        channels=(192, 256, 512, 1280, 2048),
        num_classes=num_classes,
        num_heads=64,
        **kwargs,
    )
    return CoAtNet(config)


@register_model
def coatnet_6(num_classes: int = 1000, **kwargs) -> CoAtNet:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {
            "img_size",
            "in_channels",
            "num_blocks",
            "channels",
            "num_heads",
            "scaled_num_blocks",
            "scaled_channels",
        },
        "factory variants do not allow overriding preset img_size, in_channels, num_blocks, channels, num_heads, scaled_num_blocks, or scaled_channels",
    )
    config = _build_coatnet_config(
        img_size=(224, 224),
        in_channels=3,
        num_blocks=(2, 2, 4, 8, 2),
        channels=(192, 192, 384, 768, 2048),
        num_classes=num_classes,
        num_heads=128,
        scaled_num_blocks=(8, 42),
        scaled_channels=(768, 1536),
        **kwargs,
    )
    return CoAtNet(config)


@register_model
def coatnet_7(num_classes: int = 1000, **kwargs) -> CoAtNet:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {
            "img_size",
            "in_channels",
            "num_blocks",
            "channels",
            "num_heads",
            "scaled_num_blocks",
            "scaled_channels",
        },
        "factory variants do not allow overriding preset img_size, in_channels, num_blocks, channels, num_heads, scaled_num_blocks, or scaled_channels",
    )
    config = _build_coatnet_config(
        img_size=(224, 224),
        in_channels=3,
        num_blocks=(2, 2, 4, 8, 2),
        channels=(192, 256, 512, 1024, 3072),
        num_classes=num_classes,
        num_heads=128,
        scaled_num_blocks=(8, 42),
        scaled_channels=(1024, 2048),
        **kwargs,
    )
    return CoAtNet(config)
