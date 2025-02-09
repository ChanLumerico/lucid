from typing import Type

import lucid
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


def conv_3x3_bn(
    in_channels: int, out_channels: int, downsample: bool = False
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
            nn.Linear(int(in_channels, reduction), out_channels, bias=False),
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
    ) -> None:
        super().__init__()
        self.downsample = downsample
        stride = 1 if not self.downsample else 2
        hidden_dim = int(in_channels * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
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
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, hidden_dim, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=1,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                _SEBlock(in_channels, hidden_dim),
                nn.Conv2d(hidden_dim, out_channels, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.conv = _PreNorm(in_channels, self.conv, nn.BatchNorm2d)

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

        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(axis=-1)
        self.to_qkv = nn.Linear(in_channels, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, out_channels), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        qkv = self.to_qkv(x)  # NOTE: need to implement `lucid.Tensor.chunk`
