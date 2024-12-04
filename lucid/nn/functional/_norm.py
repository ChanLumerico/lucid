import lucid
from lucid._tensor import Tensor

from lucid.types import _ShapeLike


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
    C = input_.shape[1]
    spatial_dim = input_.ndim - 2

    if training:
        batch_mean = input_.mean(axis=(0, *range(2, input_.ndim)), keepdims=True)
        batch_var = input_.var(axis=(0, *range(2, input_.ndim)), keepdims=True)

        running_mean = momentum * batch_mean.flatten() + (1 - momentum) * running_mean
        running_var = momentum * batch_var.flatten() + (1 - momentum) * running_var

        mean = batch_mean
        var = batch_var
    else:
        mean = running_mean.reshape(1, C, *(1,) * spatial_dim)
        var = running_var.reshape(1, C, *(1,) * spatial_dim)

    normalized = (input_ - mean) / lucid.sqrt(var + eps)
    if weight is not None:
        weight = weight.reshape((1, C) + (1,) * spatial_dim)
        normalized *= weight

    if bias is not None:
        bias = bias.reshape((1, C) + (1,) * spatial_dim)
        normalized += bias

    return normalized


def layer_norm(
    input_: Tensor,
    normalized_shape: _ShapeLike,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    if input_.shape[-len(normalized_shape) :] != normalized_shape:
        raise ValueError(
            "Input tensor's normalized shape must match "
            + "the provided `normalized_shape`."
        )

    mean = input_.mean(axis=tuple(range(-len(normalized_shape), 0)), keepdims=True)
    var = input_.var(axis=tuple(range(-len(normalized_shape), 0)), keepdims=True)

    normalized = (input_ - mean) / lucid.sqrt(var + eps)
    if weight is not None:
        normalized *= weight.reshape(
            (1,) * (input_.ndim - len(normalized_shape)) + normalized_shape
        )
    if bias is not None:
        normalized += bias.reshape(
            (1,) * (input_.ndim - len(normalized_shape)) + normalized_shape
        )

    return normalized


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
    C = input_.shape[1]
    spatial_dims = input_.shape[2:]

    if training:
        batch_mean = input_.mean(axis=(2, *range(2, input_.ndim)), keepdims=True)
        batch_var = input_.var(axis=(2, *range(2, input_.ndim)), keepdims=True)

        running_mean = (
            momentum * batch_mean.mean(axis=0).flatten() + (1 - momentum) * running_mean
        )
        running_var = (
            momentum * batch_var.mean(axis=0).flatten() + (1 - momentum) * running_var
        )

        mean = batch_mean
        var = batch_var
    else:
        mean = running_mean.reshape(1, C, *(1,) * len(spatial_dims))
        var = running_var.reshape(1, C, *(1,) * len(spatial_dims))

    normalized = (input_ - mean) / lucid.sqrt(var + eps)
    if weight is not None:
        normalized *= weight.reshape(1, C, *(1,) * len(spatial_dims))
    if bias is not None:
        normalized += bias.reshape(1, C, *(1,) * len(spatial_dims))

    return normalized
