from typing import Type

import lucid
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


__all__ = ["CoAtNet", "coatnet_0"]


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
        **kwargs
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

        rel_idx = self.relative_index.repeat(self.num_heads, axis=1)
        col_indices = (
            lucid.arange(self.relative_bias_table.shape[1])
            .unsqueeze(axis=0)
            .broadcast_to(rel_idx.shape)
        ).astype(int)

        relative_bias = self.relative_bias_table[
            rel_idx.astype(int), col_indices.astype(int)
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


class Transformer(nn.Module):
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
    def __init__(
        self,
        img_size: tuple[int, int],
        in_channels: int,
        num_blocks: list[int],
        channels: list[int],
        num_classes: int = 1000,
        block_types: list[str] = ["C", "C", "T", "T"],
    ) -> None:
        super().__init__()
        ih, iw = img_size
        block = {"C": _MBConv, "T": Transformer}
        get_block = lambda i: block[block_types[i]]

        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2)
        )
        self.s1 = self._make_layer(
            get_block(0), channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4)
        )
        self.s2 = self._make_layer(
            get_block(1), channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8)
        )
        self.s3 = self._make_layer(
            get_block(2), channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16)
        )
        self.s4 = self._make_layer(
            get_block(3), channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32)
        )

        self.pool = nn.AvgPool2d(kernel_size=ih // 32)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def _make_layer(
        self,
        block: Type[nn.Module],
        in_channels: int,
        out_channels: int,
        depth: int,
        img_size: tuple[int, int],
    ) -> nn.Sequential:
        layers = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                layers.append(
                    block(in_channels, out_channels, img_size=img_size, downsample=True)
                )
            else:
                layers.append(block(out_channels, out_channels, img_size=img_size))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)

        x = self.pool(x).reshape(-1, x.shape[1])
        x = self.fc(x)

        return x


# @register_model
def coatnet_0(num_classes: int = 1000, **kwargs) -> CoAtNet:
    num_blocks = [2, 2, 3, 5, 2]
    channels = [64, 96, 192, 384, 768]
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes, **kwargs)
