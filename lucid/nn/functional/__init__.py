from typing import Literal
from lucid._tensor import Tensor
from lucid.types import _ShapeLike, _Scalar

from lucid.nn.functional import (
    _activation,
    _attention,
    _linear,
    _conv,
    _pool,
    _drop,
    _norm,
    _loss,
    _spatial,
    _util,
)


def linear(input_: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    return _linear.linear(input_, weight, bias)


def bilinear(
    input_1: Tensor, input_2: Tensor, weight: Tensor, bias: Tensor | None = None
) -> Tensor:
    return _linear.bilinear(input_1, input_2, weight, bias)


def relu(input_: Tensor) -> Tensor:
    return _activation.relu(input_)


def leaky_relu(input_: Tensor, negative_slope: float = 0.01) -> Tensor:
    return _activation.leaky_relu(input_, negative_slope)


def elu(input_: Tensor, alpha: float = 1.0) -> Tensor:
    return _activation.elu(input_, alpha)


def selu(input_: Tensor) -> Tensor:
    return _activation.selu(input_)


def gelu(input_: Tensor) -> Tensor:
    return _activation.gelu(input_)


def sigmoid(input_: Tensor) -> Tensor:
    return _activation.sigmoid(input_)


def tanh(input_: Tensor) -> Tensor:
    return _activation.tanh(input_)


def softmax(input_: Tensor, axis: int = -1) -> Tensor:
    return _activation.softmax(input_, axis)


def unfold(
    input_: Tensor,
    filter_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
) -> Tensor:
    return _conv.unfold(input_, filter_size, stride, padding, dilation)


def conv1d(
    input_: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    groups: int = 1,
) -> Tensor:
    if isinstance(stride, int):
        stride = (stride,)
    if isinstance(padding, int):
        padding = (padding,)
    if isinstance(dilation, int):
        dilation = (dilation,)

    return _conv.conv(input_, weight, bias, stride, padding, dilation, groups)


def conv2d(
    input_: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    groups: int = 1,
) -> Tensor:
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    return _conv.conv(input_, weight, bias, stride, padding, dilation, groups)


def conv3d(
    input_: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    groups: int = 1,
) -> Tensor:
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    return _conv.conv(input_, weight, bias, stride, padding, dilation, groups)


def conv_transpose1d(
    input_: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    output_padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    groups: int = 1,
) -> Tensor:
    if isinstance(stride, int):
        stride = (stride,)
    if isinstance(padding, int):
        padding = (padding,)
    if isinstance(output_padding, int):
        output_padding = (output_padding,)
    if isinstance(dilation, int):
        dilation = (dilation,)

    return _conv.conv_transpose(
        input_, weight, bias, stride, padding, output_padding, dilation, groups
    )


def conv_transpose2d(
    input_: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    output_padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    groups: int = 1,
) -> Tensor:
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    return _conv.conv_transpose(
        input_, weight, bias, stride, padding, output_padding, dilation, groups
    )


def conv_transpose3d(
    input_: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    output_padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    groups: int = 1,
) -> Tensor:
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding, output_padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    return _conv.conv_transpose(
        input_, weight, bias, stride, padding, output_padding, dilation, groups
    )


def avg_pool1d(
    input_: Tensor,
    kernel_size: int | tuple[int] = 1,
    stride: int | tuple[int] = 1,
    padding: int | tuple[int] = 0,
) -> Tensor:
    return _pool.avg_pool1d(input_, kernel_size, stride, padding)


def avg_pool2d(
    input_: Tensor,
    kernel_size: int | tuple[int, int] = 1,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> Tensor:
    return _pool.avg_pool2d(input_, kernel_size, stride, padding)


def avg_pool3d(
    input_: Tensor,
    kernel_size: int | tuple[int, int, int] = 1,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
) -> Tensor:
    return _pool.avg_pool3d(input_, kernel_size, stride, padding)


def max_pool1d(
    input_: Tensor,
    kernel_size: int | tuple[int] = 1,
    stride: int | tuple[int] = 1,
    padding: int | tuple[int] = 0,
) -> Tensor:
    return _pool.max_pool1d(input_, kernel_size, stride, padding)


def max_pool2d(
    input_: Tensor,
    kernel_size: int | tuple[int, int] = 1,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> Tensor:
    return _pool.max_pool2d(input_, kernel_size, stride, padding)


def max_pool3d(
    input_: Tensor,
    kernel_size: int | tuple[int, int, int] = 1,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
) -> Tensor:
    return _pool.max_pool3d(input_, kernel_size, stride, padding)


def adaptive_avg_pool1d(input_: Tensor, output_size: int) -> Tensor:
    return _pool.adaptive_pool1d(input_, output_size, avg_or_max="avg")


def adaptive_avg_pool2d(input_: Tensor, output_size: int | tuple[int, int]) -> Tensor:
    return _pool.adaptive_pool2d(input_, output_size, avg_or_max="avg")


def adaptive_avg_pool3d(
    input_: Tensor, output_size: int | tuple[int, int, int]
) -> Tensor:
    return _pool.adaptive_pool3d(input_, output_size, avg_or_max="avg")


def adaptive_max_pool1d(input_: Tensor, output_size: int) -> Tensor:
    return _pool.adaptive_pool1d(input_, output_size, avg_or_max="max")


def adaptive_max_pool2d(input_: Tensor, output_size: int | tuple[int, int]) -> Tensor:
    return _pool.adaptive_pool2d(input_, output_size, avg_or_max="max")


def adaptive_max_pool3d(
    input_: Tensor, output_size: int | tuple[int, int, int]
) -> Tensor:
    return _pool.adaptive_pool3d(input_, output_size, avg_or_max="max")


def dropout(input_: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    return _drop.dropout(input_, p, training)


def dropout1d(input_: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    return _drop.dropoutnd(input_, p, training)


def dropout2d(input_: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    return _drop.dropoutnd(input_, p, training)


def dropout3d(input_: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    return _drop.dropoutnd(input_, p, training)


def alpha_dropout(input_: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    return _drop.alpha_dropout(input_, p, training)


def drop_block(
    input_: Tensor, block_size: int, p: float = 0.1, eps: float = 1e-7
) -> Tensor:
    if input_.ndim != 4:
        raise ValueError("Only supports 4D tensors; Shape of '(N, C, H, W)'.")
    return _drop.drop_block(input_, block_size, p, eps)


def drop_path(input_: Tensor, p: float = 0.1, scale_by_keep: bool = True) -> Tensor:
    return _drop.drop_path(input_, p, scale_by_keep)


def normalize(
    input_: Tensor, ord: int = 2, axis: int = 1, eps: float = 1e-12
) -> Tensor:
    return _norm.normalize(input_, ord, axis, eps)


def batch_norm(
    input_: Tensor,
    running_mean: Tensor | None = None,
    running_var: Tensor | None = None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    return _norm.batch_norm(
        input_, running_mean, running_var, weight, bias, training, momentum, eps
    )


def layer_norm(
    input_: Tensor,
    normalized_shape: _ShapeLike,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    return _norm.layer_norm(input_, normalized_shape, weight, bias, eps)


def instance_norm(
    input_: Tensor,
    running_mean: Tensor | None = None,
    running_var: Tensor | None = None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    return _norm.instance_norm(
        input_, running_mean, running_var, weight, bias, training, momentum, eps
    )


def global_response_norm(
    input_: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-6
) -> Tensor:
    return _norm.global_response_norm(input_, gamma, beta, eps)


_ReductionType = Literal["mean", "sum"]


def mse_loss(
    input_: Tensor, target: Tensor, reduction: _ReductionType | None = "mean"
) -> Tensor:
    return _loss.mse_loss(input_, target, reduction)


def binary_cross_entropy(
    input_: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    reduction: _ReductionType | None = "mean",
    eps: float = 1e-7,
) -> Tensor:
    return _loss.binary_cross_entropy(input_, target, weight, reduction, eps)


def cross_entropy(
    input_: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    reduction: _ReductionType | None = "mean",
    eps: float = 1e-7,
) -> Tensor:
    return _loss.cross_entropy(input_, target, weight, reduction, eps)


def nll_loss(
    input_: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    reduction: _ReductionType | None = "mean",
) -> Tensor:
    return _loss.nll_loss(input_, target, weight, reduction)


def huber_loss(
    input_: Tensor,
    target: Tensor,
    delta: float = 1.0,
    reduction: _ReductionType | None = "mean",
) -> Tensor:
    return _loss.huber_loss(input_, target, delta, reduction)


_InterpolateType = Literal["bilinear", "nearest", "area"]


def interpolate(
    input_: Tensor,
    size: tuple[int, int],
    mode: _InterpolateType = "bilinear",
    align_corners: bool = False,
) -> Tensor:
    match mode:
        case "bilinear":
            return _util._interpolate_bilinear(input_, size, align_corners)
        case "nearest":
            return _util._interpolate_nearest(input_, size, align_corners)
        case "area":
            return _util._interpolate_area(input_, size, align_corners)
        case _:
            raise ValueError("Invalid interpolation type.")


def rotate(
    input_: Tensor, angle: float, center: tuple[_Scalar, _Scalar] | None = None
) -> Tensor:
    return _util.rotate(input_, angle, center)


def embedding(
    input_: Tensor,
    weight: Tensor,
    padding_idx: int | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
) -> Tensor:
    return _util.embedding(input_, weight, padding_idx, max_norm, norm_type)


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: _Scalar | None = None,
) -> Tensor:
    return _attention.scaled_dot_product_attention(
        query, key, value, attn_mask, dropout_p, is_causal, scale
    )


def affine_grid(
    theta: Tensor, size: tuple[int, ...], align_corners: bool = True
) -> Tensor:
    return _spatial.affine_grid(theta, size, align_corners)


_PaddingType = Literal["zeros", "border"]


def grid_sample(
    input_: Tensor,
    grid: Tensor,
    mode: _InterpolateType = "bilinear",
    padding_mode: _PaddingType = "zeros",
    align_corners: bool = True,
) -> Tensor:
    return _spatial.grid_sample(input_, grid, mode, padding_mode, align_corners)
