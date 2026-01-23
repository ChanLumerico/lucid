import lucid
import lucid.nn as nn

from lucid.types import _ShapeLike

from lucid._tensor import Tensor
from lucid._kernel.norm import layer_norm_kernel, batch_norm_kernel, group_norm_kernel


def normalize(
    input_: Tensor, ord: int = 2, axis: int = 1, eps: float = 1e-12
) -> Tensor:
    norm = lucid.sum(lucid.abs(input_) ** ord, axis=axis, keepdims=True) ** (1 / ord)
    norm = lucid.maximum(norm, lucid.tensor(eps))

    return input_ / norm


def batch_norm(
    input_: Tensor,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    C = input_.shape[1]
    if running_mean is None or running_var is None:
        running_mean = lucid.zeros((C,), device=input_.device)
        running_var = lucid.ones((C,), device=input_.device)
        has_running = False
    else:
        has_running = True

    if weight is None:
        weight = lucid.ones((C,), device=input_.device)
        has_weight = False
    else:
        has_weight = True

    if bias is None:
        bias = lucid.zeros((C,), device=input_.device)
        has_bias = False
    else:
        has_bias = True

    op = batch_norm_kernel(
        eps=eps,
        momentum=momentum,
        training=training,
        has_running=has_running,
        has_weight=has_weight,
        has_bias=has_bias,
    )
    return op(input_, running_mean, running_var, weight, bias)


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
    if weight is None:
        weight = lucid.ones(normalized_shape, device=input_.device)
        has_weight = False
    else:
        has_weight = True

    if bias is None:
        bias = lucid.zeros(normalized_shape, device=input_.device)
        has_bias = False
    else:
        has_bias = True

    op = layer_norm_kernel(
        normalized_shape, eps=eps, has_weight=has_weight, has_bias=has_bias
    )
    return op(input_, weight, bias)


def instance_norm(
    input_: Tensor,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    C = input_.shape[1]
    spatial_dims = input_.shape[2:]
    use_instance_stats = training or running_mean is None or running_var is None
    axes = tuple(range(2, input_.ndim))

    if use_instance_stats:
        instance_mean = input_.mean(axis=axes, keepdims=True)
        instance_var = input_.var(axis=axes, keepdims=True)

        if training and running_mean is not None and running_var is not None:
            running_stats_ = (
                momentum * instance_mean.mean(axis=0).flatten()
                + (1 - momentum) * running_mean,
                momentum * instance_var.mean(axis=0).flatten()
                + (1 - momentum) * running_var,
            )

            if isinstance(running_mean, nn.Buffer) and isinstance(
                running_var, nn.Buffer
            ):
                running_mean.data = running_stats_[0].data
                running_var.data = running_stats_[1].data
            else:
                running_mean, running_var = running_stats_

        mean = instance_mean
        var = instance_var
    else:
        mean = running_mean.reshape(1, C, *(1,) * len(spatial_dims))
        var = running_var.reshape(1, C, *(1,) * len(spatial_dims))

    normalized = (input_ - mean) / lucid.sqrt(var + eps)

    if weight is not None:
        weight = weight.reshape((1, C) + (1,) * len(spatial_dims))
        normalized *= weight

    if bias is not None:
        bias = bias.reshape((1, C) + (1,) * len(spatial_dims))
        normalized += bias

    return normalized


def group_norm(
    input_: Tensor,
    num_groups: int,
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float = 1e-5,
) -> Tensor:
    C = input_.shape[1]
    if weight is None:
        weight = lucid.ones((C,), device=input_.device)
        has_weight = False
    else:
        has_weight = True

    if bias is None:
        bias = lucid.zeros((C,), device=input_.device)
        has_bias = False
    else:
        has_bias = True

    op = group_norm_kernel(
        num_groups=num_groups, eps=eps, has_weight=has_weight, has_bias=has_bias
    )
    return op(input_, weight, bias)


def global_response_norm(
    input_: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-6
) -> Tensor:
    Gx: Tensor = lucid.linalg.norm(input_, ord=2, axis=(-1, -2), keepdims=True)
    Nx = Gx / (Gx.mean(axis=-1, keepdims=True) + eps)
    return gamma * (input_ * Nx) + beta * input_
