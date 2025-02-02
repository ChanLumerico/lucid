import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor
from lucid.types import _Scalar


def _make_divisible(
    value: float,
    divisor: int,
    min_value: float | None = None,
    round_down_protect: bool = True,
) -> int:
    if min_value is None:
        min_value = divisor

    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor

    return int(new_value)


def _make_conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    groups: int = 1,
    bias: bool = False,
    norm: bool = True,
    act: bool = True,
) -> nn.Sequential:
    conv = nn.Sequential()
    conv.add_module(
        "conv",
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding="same",
            bias=bias,
            groups=groups,
        ),
    )
    if norm:
        conv.add_module("bn", nn.BatchNorm2d(out_channels))
    if act:
        conv.add_module("act", nn.ReLU6())

    return conv


class _InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
        act: bool = False,
        se: bool = False,
    ) -> None:
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride

        hid_channels = int(round(in_channels * expand_ratio))
        self.block = nn.Sequential()

        if expand_ratio != 1:
            self.block.add_module(
                "exp_1x1",
                _make_conv_block(
                    in_channels, hid_channels, kernel_size=3, stride=stride
                ),
            )
        if se:
            self.block.add_module(
                "conv_3x3",
                _make_conv_block(
                    hid_channels,
                    hid_channels,
                    kernel_size=3,
                    stride=stride,
                    groups=hid_channels,
                ),
            )

        self.block.add_module(
            "red_1x1",
            _make_conv_block(
                hid_channels, out_channels, kernel_size=1, stride=1, act=act
            ),
        )
        self.use_residual = self.stride == 1 and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        if self.use_residual:
            return x + self.block(x)

        return self.block(x)


class _UniversalInvertedBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        start_dw_kernel_size: int,
        mid_dw_kernel_size: int,
        mid_dw_downsample: bool,
        stride: int,
        expand_ratio: int,
    ) -> None:
        super().__init__()
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            stride_ = stride if not mid_dw_downsample else 1
            self.start_dw = _make_conv_block(
                in_channels,
                out_channels,
                kernel_size=start_dw_kernel_size,
                stride=stride_,
                groups=in_channels,
                act=False,
            )

        expand_filters = _make_divisible(in_channels * expand_ratio, divisor=8)
        self.expand_conv = _make_conv_block(in_channels, expand_filters, kernel_size=1)

        self.mid_dw_kernel_size = mid_dw_kernel_size
        if self.mid_dw_kernel_size:
            stride_ = stride if mid_dw_downsample else 1
            self.middle_dw = _make_conv_block(
                expand_filters,
                expand_filters,
                kernel_size=mid_dw_kernel_size,
                stride=stride_,
                groups=expand_filters,
            )

        self.proj_conv = _make_conv_block(
            expand_filters, out_channels, kernel_size=1, act=False
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.start_dw_kernel_size:
            x = self.start_dw(x)

        x = self.expand_conv(x)
        if self.mid_dw_kernel_size:
            x = self.middle_dw(x)

        x = self.proj_conv(x)
        return x


class _MultiQueryAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        query_h_stride: int,
        query_w_stride: int,
        kv_stride: int,
        dw_kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_h_stride = query_h_stride
        self.query_w_stride = query_w_stride
        self.kv_stride = kv_stride
        self.dw_kernel_size = dw_kernel_size
        self.dropout = dropout

        self.head_dim = key_dim // num_heads

        if self.query_h_stride > 1 or query_w_stride > 1:
            self.query_downsample_norm = nn.BatchNorm2d(in_channels)
        self.query_proj = _make_conv_block(
            in_channels, num_heads * key_dim, kernel_size=1, norm=False, act=False
        )

        if self.kv_stride > 1:
            self.key_dw_conv = _make_conv_block(
                in_channels,
                in_channels,
                kernel_size=dw_kernel_size,
                stride=kv_stride,
                groups=in_channels,
                act=False,
            )
            self.value_dw_conv = _make_conv_block(
                in_channels,
                in_channels,
                kernel_size=dw_kernel_size,
                stride=kv_stride,
                groups=in_channels,
                act=False,
            )

        self.key_proj = _make_conv_block(
            in_channels, key_dim, kernel_size=1, norm=False, act=False
        )
        self.value_proj = _make_conv_block(
            in_channels, key_dim, kernel_size=1, norm=False, act=False
        )
        self.output_proj = _make_conv_block(
            num_heads * key_dim, in_channels, kernel_size=1, norm=False, act=False
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        N = x.shape[0]
        if self.query_h_stride > 1 or self.query_w_stride > 1:
            q = F.avg_pool2d(
                x,
                kernel_size=(self.query_h_stride, self.query_w_stride),
                stride=(self.query_h_stride, self.query_w_stride),
            )
            q = self.query_downsample_norm(q)
            q = self.query_proj(q)
        else:
            q = self.query_proj(x)

        px = q.shape[2]
        q = q.reshape(N, self.num_heads, -1, self.key_dim)

        if self.kv_stride > 1:
            k = self.key_dw_conv(x)
            k = self.key_proj(k)
            v = self.value_dw_conv(x)
            v = self.value_proj(v)
        else:
            k = self.key_proj(x)
            v = self.value_proj(x)

        k = k.reshape(N, 1, self.key_dim, -1)
        v = v.reshape(N, 1, -1, self.key_dim)

        attn_score = (q @ k) / (self.head_dim**0.5)
        attn_score = self.dropout(attn_score)
        attn_score = F.softmax(attn_score, axis=-1)

        context = attn_score @ v
        context = context.shape(N, self.num_heads * self.key_dim, px, px)

        out = self.output_proj(context)
        return out


class _LayerScale(nn.Module):
    def __init__(self, in_channels: int, init_value: _Scalar) -> None:
        super().__init__()
        self.init_value = init_value
        self.gamma = nn.Parameter(self.init_value * lucid.ones(in_channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class _MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        query_h_stride: int,
        query_w_stride: int,
        kv_stride: int,
        use_layer_scale: bool,
        use_multi_query: bool,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.query_h_stride = query_h_stride
        self.query_w_stride = query_w_stride
        self.kv_stride = kv_stride
        self.use_layer_scale = use_layer_scale
        self.use_multi_query = use_multi_query
        self.use_residual = use_residual

        self.input_norm = nn.BatchNorm2d(in_channels)
        if self.use_multi_query:
            self.multi_query_attention = _MultiQueryAttention(
                in_channels,
                num_heads,
                key_dim,
                value_dim,
                query_h_stride,
                query_w_stride,
                kv_stride,
            )
        else:
            self.multi_head_attention = ...  # TODO: Implement `nn.MultiHeadAttention`
