from typing import Callable
from functools import partial
import math

import lucid
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


__all__ = [
    "MaxViT",
    "maxvit_tiny",
    "maxvit_small",
    "maxvit_base",
    "maxvit_large",
    "maxvit_xlarge",
]


def _to_2tuple(val: int | float) -> tuple[int | float, ...]:
    return (val, val)


class _MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_feature: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] | None = None,
        bias: bool = True,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_feature = hidden_feature or in_features

        bias = _to_2tuple(bias)
        drop_probs = _to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_feature, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_feature) if norm_layer is not None else nn.Identity()
        )

        self.fc2 = nn.Linear(hidden_feature, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(self.norm(x)))
        return x


class _SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_channels: int,
        rd_ratio: float = 0.25,
        rd_channels: int | None = None,
        act_layer: type[nn.Module] = nn.ReLU,
        gate_layer: type[nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        if rd_channels is None:
            rd_channels = int(in_channels * rd_ratio)

        self.conv_reduce = nn.Conv2d(in_channels, rd_channels, kernel_size=1)
        self.act = act_layer()
        self.conv_expand = nn.Conv2d(rd_channels, in_channels, kernel_size=1)
        self.gate = gate_layer()

    def forward(self, x: Tensor) -> Tensor:
        x_se = x.mean(axis=(2, 3), keepdims=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act(x_se)
        x_se = self.conv_expand(x_se)

        return x * self.gate(x_se)


class _DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        act_layer: type[nn.Module] = nn.ReLU,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.act = act_layer()
        self.bn = norm_layer(out_channels)
        self.drop_path = (
            nn.DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.drop_path(self.act(self.bn(x)))

        return x


class _MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downscale: bool = False,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.drop_path_rate = drop_path
        if not downscale:
            assert (
                in_channels == out_channels
            ), "in/out channels must be equal when downscale=True."
        self.in_channels = in_channels
        self.main_path = nn.Sequential(
            norm_layer(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            _DepthwiseSeparableConv(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2 if downscale else 1,
                act_layer=act_layer,
                drop_path_rate=drop_path,
            ),
            _SqueezeExcite(out_channels, rd_ratio=0.25),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )
        self.skip_path = (
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
            if downscale
            else nn.Identity()
        )
        self.drop = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.main_path(x)
        out = self.drop(out)
        out += self.skip_path(x)

        return out


def _window_partition(x: Tensor, window_size: tuple[int, int] = (7, 7)) -> Tensor:
    B, C, H, W = x.shape
    windows = x.reshape(
        B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1]
    )
    windows = windows.transpose((0, 2, 4, 3, 5, 1)).reshape(-1, *window_size, C)
    return windows


def _window_reverse(
    windows: Tensor,
    original_size: tuple[int, int],
    window_size: tuple[int, int] = (7, 7),
) -> Tensor:
    H, W = original_size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))

    out = windows.reshape(B, H // window_size[0], W // window_size[1], *window_size, -1)
    out = out.transpose((0, 5, 1, 3, 2, 4)).reshape(B, -1, H, W)
    return out


def _grid_partition(x: Tensor, grid_size: tuple[int, int] = (7, 7)) -> Tensor:
    B, C, H, W = x.shape
    grid = x.reshape(
        B, C, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1]
    )
    grid = grid.transpose((0, 3, 5, 2, 4, 1)).reshape(-1, *grid_size, C)
    return grid


def _grid_reverse(
    grid: Tensor, original_size: tuple[int, int], grid_size: tuple[int, int] = (7, 7)
) -> Tensor:
    (H, W), C = original_size, grid.shape[-1]
    B = int(grid.shape[0] / (H * W / grid_size[0] / grid_size[1]))
    out = grid.reshape(B, H // grid_size[0], W // grid_size[1], *grid_size, C)
    out = out.transpose((0, 5, 3, 1, 4, 2)).reshape(B, C, H, W)
    return out


def _get_relative_position_index(win_h: int, win_w: int) -> Tensor:
    coords = lucid.stack(lucid.meshgrid(lucid.arange(win_h), lucid.arange(win_w)))
    coords_flatten = lucid.flatten(coords, axis=1)

    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.transpose((1, 2, 0))

    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= win_w - 1

    return relative_coords.sum(axis=-1).astype(lucid.Int)


class _RelativeSelfAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 32,
        grid_window_size: tuple[int, int] = (7, 7),
        attn_drop: float = 0.0,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.grid_window_size = grid_window_size

        self.scale = num_heads**-0.5
        self.attn_area = grid_window_size[0] * grid_window_size[1]

        self.qkv_mapping = nn.Linear(in_channels, 3 * in_channels)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(in_channels, in_channels)
        self.proj_drop = nn.Dropout(drop)
        self.softmax = nn.Softmax(axis=-1)

        self.relative_pos_bias_table = nn.Parameter(
            lucid.zeros(
                (2 * grid_window_size[0] - 1) * (2 * grid_window_size[1] - 1), num_heads
            )
        )
        nn.init.normal(self.relative_pos_bias_table, std=0.02)

        self.relative_pos_index: nn.Buffer
        self.register_buffer(
            "relative_pos_index", _get_relative_position_index(*grid_window_size)
        )

    def _get_relative_positional_bias(self) -> Tensor:
        rel_pos_bias = self.relative_pos_bias_table[
            self.relative_pos_index.reshape(-1)
        ].reshape(self.attn_area, self.attn_area, -1)

        rel_pos_bias = rel_pos_bias.transpose((2, 0, 1))
        return rel_pos_bias.unsqueeze(axis=0)

    def forward(self, x: Tensor) -> Tensor:
        B_, N, _ = x.shape
        qkv = (
            self.qkv_mapping(x)
            .reshape(B_, N, 3, self.num_heads, -1)
            .transpose((2, 0, 3, 1, 4))
        )
        q, k, v = qkv.chunk(3)
        q *= self.scale

        attn = self.softmax(q @ k.mT + self._get_relative_positional_bias())
        out = (attn @ v).swapaxes(1, 2).reshape(B_, N, -1)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class _MaxViTTransformerBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        partition_func: Callable,
        reverse_func: Callable,
        num_heads: int = 32,
        grid_window_size: tuple[int, int] = (7, 7),
        attn_drop: float = 0.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        mlp_ratio: float = 4.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.partition_func: Callable[[Tensor, tuple], Tensor] = partition_func
        self.reverse_func: Callable[[Tensor, tuple, tuple], Tensor] = reverse_func
        self.grid_window_size = grid_window_size

        self.norm_1 = norm_layer(in_channels)
        self.attention = _RelativeSelfAttention(
            in_channels, num_heads, grid_window_size, attn_drop, drop
        )
        self.drop_path = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm_2 = norm_layer(in_channels)
        self.mlp = _MLP(
            in_channels, int(mlp_ratio * in_channels), act_layer=act_layer, drop=drop
        )

    def forward(self, x: Tensor) -> Tensor:
        _, C, H, W = x.shape
        input_part = self.partition_func(x, self.grid_window_size)
        input_part = input_part.reshape(-1, math.prod(self.grid_window_size), C)

        out = input_part + self.drop_path(self.attention(self.norm_1(input_part)))
        out += self.drop_path(self.mlp(self.norm_2(out)))
        out = self.reverse_func(out, (H, W), self.grid_window_size)

        return out


class _MaxViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downscale: bool = False,
        num_heads: int = 32,
        grid_window_size: tuple[int, int] = (7, 7),
        attn_drop: float = 0.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        mlp_ratio: float = 4.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        norm_layer_tf: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.mb_conv = _MBConv(
            in_channels, out_channels, downscale, act_layer, norm_layer, drop_path
        )
        base_tf = partial(
            _MaxViTTransformerBlock,
            in_channels=out_channels,
            num_heads=num_heads,
            grid_window_size=grid_window_size,
            attn_drop=attn_drop,
            drop=drop,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer_tf,
        )

        self.block_tf = base_tf(
            partition_func=_window_partition, reverse_func=_window_reverse
        )
        self.grid_tf = base_tf(
            partition_func=_grid_partition, reverse_func=_grid_reverse
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.mb_conv(x)
        out = self.block_tf(out)
        out = self.grid_tf(out)

        return out


class _MaxViTStage(nn.Module):
    def __init__(
        self,
        depth: int,
        in_channels: int,
        out_channels: int,
        num_heads: int = 32,
        grid_window_size: tuple[int, int] = (7, 7),
        attn_drop: float = 0.0,
        drop: float = 0.0,
        drop_path: list[float] | float = 0.0,
        mlp_ratio: float = 4.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        norm_layer_tf: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        blocks = [
            _MaxViTBlock(
                in_channels=in_channels if idx == 0 else out_channels,
                out_channels=out_channels,
                downscale=idx == 0,
                num_heads=num_heads,
                grid_window_size=grid_window_size,
                attn_drop=attn_drop,
                drop=drop,
                drop_path=drop_path if isinstance(drop_path, float) else drop_path[idx],
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer,
                norm_layer_tf=norm_layer_tf,
            )
            for idx in range(depth)
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class MaxViT(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        depths: tuple[int, ...] = (2, 2, 5, 2),
        channels: tuple[int, ...] = (64, 128, 256, 512),
        num_classes: int = 1000,
        embed_dim: int = 64,
        num_heads: int = 32,
        grid_window_size: tuple[int, int] = (7, 7),
        attn_drop: float = 0.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        mlp_ratio: float = 4.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        norm_layer_tf: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert len(depths) == len(
            channels
        ), "Channel dims must be given for each stage."

        self.num_classes = num_classes
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=2, padding=1),
            act_layer(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            act_layer(),
        )

        drop_path = lucid.linspace(0, drop_path, sum(depths))
        stages = []
        for idx, (depth, channel) in enumerate(zip(depths, channels)):
            stages.append(
                _MaxViTStage(
                    depth=depth,
                    in_channels=embed_dim if idx == 0 else channels[idx - 1],
                    out_channels=channel,
                    num_heads=num_heads,
                    grid_window_size=grid_window_size,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path[sum(depths[:idx]) : sum(depths[: idx + 1])],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    norm_layer_tf=norm_layer_tf,
                )
            )

        self.stages = nn.ModuleList(stages)
        self.head = nn.Linear(channels[-1], num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)

        x = x.mean(axis=(2, 3))
        out = self.head(x)
        return out


@register_model
def maxvit_tiny(in_channels: int = 3, num_classes: int = 1000, **kwargs) -> MaxViT:
    return MaxViT(
        in_channels=in_channels,
        num_classes=num_classes,
        depths=(2, 2, 5, 2),
        channels=(64, 128, 256, 512),
        embed_dim=64,
        **kwargs
    )


@register_model
def maxvit_small(in_channels: int = 3, num_classes: int = 1000, **kwargs) -> MaxViT:
    return MaxViT(
        in_channels=in_channels,
        num_classes=num_classes,
        depths=(2, 2, 5, 2),
        channels=(96, 192, 384, 768),
        embed_dim=64,
        **kwargs
    )


@register_model
def maxvit_base(in_channels: int = 3, num_classes: int = 1000, **kwargs) -> MaxViT:
    return MaxViT(
        in_channels=in_channels,
        num_classes=num_classes,
        depths=(2, 6, 14, 2),
        channels=(96, 192, 384, 768),
        embed_dim=64,
        **kwargs
    )


@register_model
def maxvit_large(in_channels: int = 3, num_classes: int = 1000, **kwargs) -> MaxViT:
    return MaxViT(
        in_channels=in_channels,
        num_classes=num_classes,
        depths=(2, 6, 14, 2),
        channels=(128, 256, 512, 1024),
        embed_dim=128,
        **kwargs
    )


@register_model
def maxvit_xlarge(in_channels: int = 3, num_classes: int = 1000, **kwargs) -> MaxViT:
    return MaxViT(
        in_channels=in_channels,
        num_classes=num_classes,
        depths=(2, 6, 14, 2),
        channels=(192, 384, 768, 1536),
        embed_dim=192,
        **kwargs
    )
