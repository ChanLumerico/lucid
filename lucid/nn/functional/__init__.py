"""
lucid.nn.functional — stateless function interface for nn ops.

Mirror of `torch.nn.functional`. All ops accept tensors and return
tensors with no learned-parameter state of their own (state is
managed by the matching nn.Module wrappers).
"""

from __future__ import annotations

from typing import Literal

from lucid._tensor import Tensor
from lucid.types import _ShapeLike, _Scalar, _DeviceType, Numeric

from lucid.nn.functional import (
    _activation, _attention, _linear, _conv, _pool, _drop,
    _norm, _loss, _spatial, _utils, _embedding,
)


# --------------------------------------------------------------------------- #
# Linear
# --------------------------------------------------------------------------- #

def linear(input_: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    return _linear.linear(input_, weight, bias)


def bilinear(
    input_1: Tensor, input_2: Tensor, weight: Tensor, bias: Tensor | None = None
) -> Tensor:
    return _linear.bilinear(input_1, input_2, weight, bias)


# --------------------------------------------------------------------------- #
# Activations
# --------------------------------------------------------------------------- #

def relu(input_: Tensor) -> Tensor:                            return _activation.relu(input_)
def leaky_relu(input_: Tensor, negative_slope: float = 0.01):  return _activation.leaky_relu(input_, negative_slope)
def elu(input_: Tensor, alpha: float = 1.0) -> Tensor:         return _activation.elu(input_, alpha)
def selu(input_: Tensor) -> Tensor:                            return _activation.selu(input_)
def gelu(input_: Tensor) -> Tensor:                            return _activation.gelu(input_)
def sigmoid(input_: Tensor) -> Tensor:                         return _activation.sigmoid(input_)
def tanh(input_: Tensor) -> Tensor:                            return _activation.tanh(input_)
def silu(input_: Tensor) -> Tensor:                            return _activation.silu(input_)
def softmax(input_: Tensor, axis: int = -1) -> Tensor:         return _activation.softmax(input_, axis)
def softplus(input_: Tensor) -> Tensor:                        return _activation.softplus(input_)
def mish(input_: Tensor) -> Tensor:                            return _activation.mish(input_)
def hard_sigmoid(input_: Tensor) -> Tensor:                    return _activation.hard_sigmoid(input_)
def hard_swish(input_: Tensor) -> Tensor:                      return _activation.hard_swish(input_)
def relu6(input_: Tensor) -> Tensor:                           return _activation.relu6(input_)


# --------------------------------------------------------------------------- #
# Convolution
# --------------------------------------------------------------------------- #

def unfold(
    input_: Tensor,
    filter_size: tuple[int, ...], stride: tuple[int, ...],
    padding: tuple[int, ...], dilation: tuple[int, ...],
) -> Tensor:
    return _conv.unfold(input_, filter_size, stride, padding, dilation)


def _normalize_conv_args(stride, padding, dilation, n: int):
    def _to_n(v):
        return (v,) * n if isinstance(v, int) else tuple(v)
    return _to_n(stride), _to_n(padding), _to_n(dilation)


def conv1d(input_: Tensor, weight: Tensor, bias: Tensor | None = None,
           stride=1, padding=0, dilation=1, groups: int = 1) -> Tensor:
    s, p, d = _normalize_conv_args(stride, padding, dilation, 1)
    return _conv.conv(input_, weight, bias, s, p, d, groups)


def conv2d(input_: Tensor, weight: Tensor, bias: Tensor | None = None,
           stride=1, padding=0, dilation=1, groups: int = 1) -> Tensor:
    s, p, d = _normalize_conv_args(stride, padding, dilation, 2)
    return _conv.conv(input_, weight, bias, s, p, d, groups)


def conv3d(input_: Tensor, weight: Tensor, bias: Tensor | None = None,
           stride=1, padding=0, dilation=1, groups: int = 1) -> Tensor:
    s, p, d = _normalize_conv_args(stride, padding, dilation, 3)
    return _conv.conv(input_, weight, bias, s, p, d, groups)


def _normalize_conv_t_args(stride, padding, output_padding, dilation, n: int):
    def _to_n(v):
        return (v,) * n if isinstance(v, int) else tuple(v)
    return (_to_n(stride), _to_n(padding), _to_n(output_padding), _to_n(dilation))


def conv_transpose1d(input_: Tensor, weight: Tensor, bias: Tensor | None = None,
                     stride=1, padding=0, output_padding=0, dilation=1,
                     groups: int = 1) -> Tensor:
    s, p, op, d = _normalize_conv_t_args(stride, padding, output_padding, dilation, 1)
    return _conv.conv_transpose(input_, weight, bias, s, p, op, d, groups)


def conv_transpose2d(input_: Tensor, weight: Tensor, bias: Tensor | None = None,
                     stride=1, padding=0, output_padding=0, dilation=1,
                     groups: int = 1) -> Tensor:
    s, p, op, d = _normalize_conv_t_args(stride, padding, output_padding, dilation, 2)
    return _conv.conv_transpose(input_, weight, bias, s, p, op, d, groups)


def conv_transpose3d(input_: Tensor, weight: Tensor, bias: Tensor | None = None,
                     stride=1, padding=0, output_padding=0, dilation=1,
                     groups: int = 1) -> Tensor:
    s, p, op, d = _normalize_conv_t_args(stride, padding, output_padding, dilation, 3)
    return _conv.conv_transpose(input_, weight, bias, s, p, op, d, groups)


# --------------------------------------------------------------------------- #
# Pooling
# --------------------------------------------------------------------------- #

def avg_pool1d(input_, kernel_size=1, stride=1, padding=0): return _pool.avg_pool1d(input_, kernel_size, stride, padding)
def avg_pool2d(input_, kernel_size=1, stride=1, padding=0): return _pool.avg_pool2d(input_, kernel_size, stride, padding)
def avg_pool3d(input_, kernel_size=1, stride=1, padding=0): return _pool.avg_pool3d(input_, kernel_size, stride, padding)
def max_pool1d(input_, kernel_size=1, stride=1, padding=0): return _pool.max_pool1d(input_, kernel_size, stride, padding)
def max_pool2d(input_, kernel_size=1, stride=1, padding=0): return _pool.max_pool2d(input_, kernel_size, stride, padding)
def max_pool3d(input_, kernel_size=1, stride=1, padding=0): return _pool.max_pool3d(input_, kernel_size, stride, padding)
def adaptive_avg_pool1d(input_, output_size):       return _pool.adaptive_pool1d(input_, output_size, "avg")
def adaptive_avg_pool2d(input_, output_size):       return _pool.adaptive_pool2d(input_, output_size, "avg")
def adaptive_avg_pool3d(input_, output_size):       return _pool.adaptive_pool3d(input_, output_size, "avg")
def adaptive_max_pool1d(input_, output_size):       return _pool.adaptive_pool1d(input_, output_size, "max")
def adaptive_max_pool2d(input_, output_size):       return _pool.adaptive_pool2d(input_, output_size, "max")
def adaptive_max_pool3d(input_, output_size):       return _pool.adaptive_pool3d(input_, output_size, "max")


# --------------------------------------------------------------------------- #
# Dropout
# --------------------------------------------------------------------------- #

def dropout(input_, p=0.5, training=True):       return _drop.dropout(input_, p, training)
def dropout1d(input_, p=0.5, training=True):     return _drop.dropoutnd(input_, p, training)
def dropout2d(input_, p=0.5, training=True):     return _drop.dropoutnd(input_, p, training)
def dropout3d(input_, p=0.5, training=True):     return _drop.dropoutnd(input_, p, training)
def alpha_dropout(input_, p=0.5, training=True): return _drop.alpha_dropout(input_, p, training)


def drop_block(input_, block_size, p=0.1, eps=1e-7):
    if input_.ndim != 4:
        raise ValueError("Only supports 4D tensors; Shape of '(N, C, H, W)'.")
    return _drop.drop_block(input_, block_size, p, eps)


def drop_path(input_, p=0.1, scale_by_keep=True): return _drop.drop_path(input_, p, scale_by_keep)


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #

def normalize(input_, ord=2, axis=1, eps=1e-12): return _norm.normalize(input_, ord, axis, eps)


def batch_norm(input_, running_mean=None, running_var=None,
               weight=None, bias=None, training=True, momentum=0.1, eps=1e-5):
    return _norm.batch_norm(input_, running_mean, running_var, weight, bias,
                             training, momentum, eps)


def layer_norm(input_, normalized_shape, weight=None, bias=None, eps=1e-5):
    return _norm.layer_norm(input_, normalized_shape, weight, bias, eps)


def instance_norm(input_, running_mean=None, running_var=None,
                  weight=None, bias=None, training=True, momentum=0.1, eps=1e-5):
    return _norm.instance_norm(input_, running_mean, running_var, weight, bias,
                                training, momentum, eps)


def group_norm(input_, num_groups, weight=None, bias=None, eps=1e-5):
    return _norm.group_norm(input_, num_groups, weight, bias, eps)


def global_response_norm(input_, gamma, beta, eps=1e-6):
    return _norm.global_response_norm(input_, gamma, beta, eps)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #

_ReductionType = Literal["mean", "sum"]


def mse_loss(input_, target, reduction: _ReductionType | None = "mean"):
    return _loss.mse_loss(input_, target, reduction)


def binary_cross_entropy(input_, target, weight=None,
                         reduction: _ReductionType | None = "mean", eps=1e-7):
    return _loss.binary_cross_entropy(input_, target, weight, reduction, eps)


def binary_cross_entropy_with_logits(input_, target, weight=None, pos_weight=None,
                                      reduction: _ReductionType | None = "mean"):
    return _loss.binary_cross_entropy_with_logits(
        input_, target, weight, pos_weight, reduction)


def cross_entropy(input_, target, weight=None,
                  reduction: _ReductionType | None = "mean", eps=1e-7,
                  ignore_index: int | None = None):
    return _loss.cross_entropy(input_, target, weight, reduction, eps, ignore_index)


def nll_loss(input_, target, weight=None,
             reduction: _ReductionType | None = "mean",
             ignore_index: int | None = None):
    return _loss.nll_loss(input_, target, weight, reduction, ignore_index)


def huber_loss(input_, target, delta: float = 1.0,
               reduction: _ReductionType | None = "mean"):
    return _loss.huber_loss(input_, target, delta, reduction)


# --------------------------------------------------------------------------- #
# Spatial / interpolation
# --------------------------------------------------------------------------- #

_InterpolateType = Literal["bilinear", "trilinear", "nearest", "area"]


def interpolate(input_, size, mode: _InterpolateType = "bilinear",
                align_corners: bool = False):
    if mode == "bilinear":
        return _utils._interpolate_bilinear(input_, size, align_corners)
    if mode == "trilinear":
        return _utils._interpolate_trilinear(input_, size, align_corners)
    if mode == "nearest":
        if input_.ndim == 5:
            return _utils._interpolate_nearest_3d(input_, size, align_corners)
        return _utils._interpolate_nearest(input_, size, align_corners)
    if mode == "area":
        return _utils._interpolate_area(input_, size, align_corners)
    raise ValueError("Invalid interpolation type.")


def rotate(input_, angle: float, center=None):
    return _utils.rotate(input_, angle, center)


def scaled_dot_product_attention(query, key, value, attn_mask=None,
                                  dropout_p: float = 0.0, is_causal: bool = False,
                                  scale=None, output_weight: bool = False):
    return _attention.scaled_dot_product_attention(
        query, key, value, attn_mask, dropout_p, is_causal, scale, output_weight)


def affine_grid(theta, size, align_corners: bool = True):
    return _spatial.affine_grid(theta, size, align_corners)


_PaddingType = Literal["zeros", "border"]


def grid_sample(input_, grid, mode: _InterpolateType = "bilinear",
                padding_mode: _PaddingType = "zeros",
                align_corners: bool = True):
    return _spatial.grid_sample(input_, grid, mode, padding_mode, align_corners)


def one_hot(input_, num_classes: int = -1, dtype=None):
    return _utils.one_hot(input_, num_classes, dtype)


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #

def embedding(input_, weight, padding_idx=None, max_norm=None, norm_type: float = 2.0):
    return _embedding.embedding(input_, weight, padding_idx, max_norm, norm_type)


def sinusoidal_pos_embedding(seq_len: int, embed_dim: int,
                              device: _DeviceType = "cpu", dtype=None):
    if seq_len <= 0 or embed_dim <= 0:
        raise ValueError("seq_len and embed_dim must be positive.")
    return _embedding.sinusoidal_pos_embedding(seq_len, embed_dim, device, dtype)


def rotary_pos_embedding(input_, position_ids=None, interleaved: bool = True):
    return _embedding.rotary_pos_embedding(input_, position_ids, interleaved)
