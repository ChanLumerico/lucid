from lucid._tensor import Tensor
from lucid.types import _ShapeLike

from lucid.nn.functional import _linear, _non_linear, _conv, _pool, _drop, _norm


def linear(input_: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    return _linear.linear(input_, weight, bias)


def bilinear(
    input_1: Tensor, input_2: Tensor, weight: Tensor, bias: Tensor | None = None
) -> Tensor:
    return _linear.bilinear(input_1, input_2, weight, bias)


def relu(input_: Tensor) -> Tensor:
    return _non_linear.relu(input_)


def leaky_relu(input_: Tensor, negative_slope: float = 0.01) -> Tensor:
    return _non_linear.leaky_relu(input_, negative_slope)


def elu(input_: Tensor, alpha: float = 1.0) -> Tensor:
    return _non_linear.elu(input_, alpha)


def selu(input_: Tensor) -> Tensor:
    return _non_linear.selu(input_)


def gelu(input_: Tensor) -> Tensor:
    return _non_linear.gelu(input_)


def sigmoid(input_: Tensor) -> Tensor:
    return _non_linear.sigmoid(input_)


def tanh(input_: Tensor) -> Tensor:
    return _non_linear.tanh(input_)


def unfold(
    input_: Tensor,
    filter_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
) -> Tensor:
    return _conv.unfold(input_, filter_size, stride, padding)


def conv1d(
    input_: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
) -> Tensor:
    return _conv.conv1d(input_, weight, bias, stride, padding)


def conv2d(
    input_: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
) -> Tensor:
    return _conv.conv2d(input_, weight, bias, stride, padding)


def conv3d(
    input_: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
) -> Tensor:
    return _conv.conv3d(input_, weight, bias, stride, padding)


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


def batch_norm(
    input_: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
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
    running_mean: Tensor,
    running_var: Tensor,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    return _norm.instance_norm(
        input_, running_mean, running_var, weight, bias, training, momentum, eps
    )
