"""
lucid.nn.functional._norm — normalization layers.

All routes are 1:1 to engine ops:
  * batch_norm  → batch_norm{1,2,3}d (training) or batch_norm_eval (inference).
  * layer_norm  → engine.nn.layer_norm.
  * group_norm  → engine.nn.group_norm.
  * instance_norm → group_norm with num_groups = C (math identity).
  * normalize   → engine.nn.lp_normalize.
  * global_response_norm → engine.nn.global_response_norm.
"""

from __future__ import annotations

from lucid._C import engine as _C_engine
from lucid._C.engine import nn as _C_nn
from lucid._tensor import Tensor
from lucid._bridge import impl_of
from lucid.ops.gfunc import ones, zeros
from lucid.autograd import no_grad
from lucid.types import _ShapeLike


def normalize(
    input_: Tensor, ord: int = 2, axis: int = 1, eps: float = 1e-12
) -> Tensor:
    return Tensor._wrap(_C_nn.lp_normalize(
        impl_of(input_), float(ord), int(axis), float(eps)))


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
    weight_ = (weight if weight is not None
               else ones((C,), device=input_.device, dtype=input_.dtype))
    bias_ = (bias if bias is not None
             else zeros((C,), device=input_.device, dtype=input_.dtype))

    if training or running_mean is None or running_var is None:
        if input_.ndim == 3:
            out = _C_nn.batch_norm1d(impl_of(input_), impl_of(weight_),
                                       impl_of(bias_), float(eps))
        elif input_.ndim == 4:
            out = _C_nn.batch_norm(impl_of(input_), impl_of(weight_),
                                     impl_of(bias_), float(eps))
        elif input_.ndim == 5:
            out = _C_nn.batch_norm3d(impl_of(input_), impl_of(weight_),
                                       impl_of(bias_), float(eps))
        else:
            raise ValueError(f"batch_norm: unsupported ndim {input_.ndim}")

        if training and running_mean is not None and running_var is not None:
            axes = tuple([0] + list(range(2, input_.ndim)))
            with no_grad():
                batch_mean = input_.mean(axis=axes)
                batch_var = input_.var(axis=axes)
                new_mean = momentum * batch_mean + (1 - momentum) * running_mean
                new_var = momentum * batch_var + (1 - momentum) * running_var
                running_mean._impl = new_mean._impl
                running_var._impl = new_var._impl

        return Tensor._wrap(out)

    return Tensor._wrap(_C_nn.batch_norm_eval(
        impl_of(input_), impl_of(running_mean), impl_of(running_var),
        impl_of(weight_), impl_of(bias_), float(eps)))


def layer_norm(
    input_: Tensor,
    normalized_shape: _ShapeLike,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    if tuple(input_.shape[-len(normalized_shape):]) != tuple(normalized_shape):
        raise ValueError(
            "Input tensor's normalized shape must match `normalized_shape`.")
    weight_ = (weight if weight is not None
               else ones(normalized_shape, device=input_.device,
                                dtype=input_.dtype))
    bias_ = (bias if bias is not None
             else zeros(normalized_shape, device=input_.device,
                               dtype=input_.dtype))
    return Tensor._wrap(_C_nn.layer_norm(
        impl_of(input_), impl_of(weight_), impl_of(bias_), float(eps)))


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
    """InstanceNorm = GroupNorm with num_groups = C (each channel its own group)."""
    C = input_.shape[1]
    weight_ = (weight if weight is not None
               else ones((C,), device=input_.device, dtype=input_.dtype))
    bias_ = (bias if bias is not None
             else zeros((C,), device=input_.device, dtype=input_.dtype))

    out = _C_nn.group_norm(impl_of(input_), impl_of(weight_), impl_of(bias_),
                            int(C), float(eps))

    # Update running stats if both training and stats are tracked.
    if training and running_mean is not None and running_var is not None:
        axes = tuple(range(2, input_.ndim))
        with no_grad():
            inst_mean = input_.mean(axis=axes).mean(axis=0)  # average over batch
            inst_var = input_.var(axis=axes).mean(axis=0)
            new_mean = momentum * inst_mean + (1 - momentum) * running_mean
            new_var = momentum * inst_var + (1 - momentum) * running_var
            running_mean._impl = new_mean._impl
            running_var._impl = new_var._impl
    return Tensor._wrap(out)


def group_norm(
    input_: Tensor,
    num_groups: int,
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float = 1e-5,
) -> Tensor:
    C = input_.shape[1]
    weight_ = (weight if weight is not None
               else ones((C,), device=input_.device, dtype=input_.dtype))
    bias_ = (bias if bias is not None
             else zeros((C,), device=input_.device, dtype=input_.dtype))
    return Tensor._wrap(_C_nn.group_norm(
        impl_of(input_), impl_of(weight_), impl_of(bias_),
        int(num_groups), float(eps)))


def global_response_norm(
    input_: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-6
) -> Tensor:
    return Tensor._wrap(_C_nn.global_response_norm(
        impl_of(input_), impl_of(gamma), impl_of(beta), float(eps)))
