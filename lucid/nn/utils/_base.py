from typing import Callable, Iterable

import lucid

from lucid._tensor import Tensor
from lucid.types import _Scalar


__all__ = [
    "grad_norm",
    "get_total_norm",
    "clip_grad_norm",
    "clip_grad_value",
    "apply_chunking_to_forward",
]


def _as_iter(parameters: Iterable[Tensor] | Tensor) -> list[Tensor]:
    if isinstance(parameters, Tensor):
        return [parameters]
    return list(parameters)


def grad_norm(parameters: Iterable[Tensor] | Tensor, norm_type: int = 2) -> Tensor:
    parameters = _as_iter(parameters)
    device = parameters[0].device

    params: list[Tensor] = [p for p in parameters if p.grad is not None]
    if not params:
        return Tensor(0.0, device=device)

    norm_pow_sum = 0.0
    for p in params:
        param_norm = lucid.linalg.norm(lucid.ravel(p.grad), ord=norm_type).item()
        norm_pow_sum += param_norm**norm_type

    total_norm = norm_pow_sum ** (1.0 / norm_type)
    return Tensor(total_norm, device=device)


def get_total_norm(parameters: Iterable[Tensor] | Tensor, norm_type: int = 2) -> Tensor:
    parameters = _as_iter(parameters)
    if not parameters:
        return Tensor(0.0)

    device = parameters[0].device
    grads: list[Tensor] = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return Tensor(0.0, device=device)

    norm_pow_sum = 0.0
    for g in grads:
        grad_norm = lucid.linalg.norm(lucid.ravel(g), ord=norm_type).item()
        norm_pow_sum += grad_norm**norm_type

    total_norm = norm_pow_sum ** (1.0 / norm_type)
    return Tensor(total_norm, device=device)


def clip_grad_norm(
    parameters: Iterable[Tensor] | Tensor,
    max_norm: _Scalar,
    norm_type: int = 2,
    eps: float = 1e-7,
) -> float:
    params: list[Tensor] = [p for p in _as_iter(parameters) if p.grad is not None]
    total_norm = get_total_norm(params, norm_type=norm_type)

    clip_coef = float(max_norm) / (total_norm.item() + eps)
    if clip_coef < 1.0:
        for p in params:
            p.grad = p.grad * clip_coef

    return total_norm


def clip_grad_value(parameters: Iterable[Tensor] | Tensor, clip_value: _Scalar) -> None:
    params = [p for p in _as_iter(parameters) if p.grad is not None]
    if not params:
        return

    lo, hi = -float(clip_value), float(clip_value)
    for p in params:
        g_clip = lucid.clip(p.grad, lo, hi).data
        p.grad = g_clip


def apply_chunking_to_forward(
    forward_fn: Callable[..., Tensor],
    chunk_size: int,
    chunk_dim: int,
    *input_tensors: Tensor,
) -> Tensor:
    if len(input_tensors) == 0:
        raise ValueError("input_tensors must contain at least one tensor.")
    if chunk_size < 0:
        raise ValueError(f"chunk_size must be >= 0 (got {chunk_size}).")

    ref_ndim = input_tensors[0].ndim
    if chunk_dim < -ref_ndim or chunk_dim >= ref_ndim:
        raise ValueError(
            f"chunk_dim out of range for ndim={ref_ndim} (got {chunk_dim})."
        )
    if chunk_dim < 0:
        chunk_dim += ref_ndim

    ref_shape = input_tensors[0].shape
    for tensor in input_tensors[1:]:
        if tensor.ndim != ref_ndim:
            raise ValueError(
                "All input_tensors must have the same number of dimensions "
                f"(got {ref_ndim} and {tensor.ndim})."
            )
        if tensor.shape[chunk_dim] != ref_shape[chunk_dim]:
            raise ValueError(
                "All input_tensors must have the same length at chunk_dim "
                f"{chunk_dim} (got {ref_shape[chunk_dim]} and {tensor.shape[chunk_dim]})."
            )

    if chunk_size == 0:
        return forward_fn(*input_tensors)

    dim_size = ref_shape[chunk_dim]
    if dim_size % chunk_size != 0:
        raise ValueError(
            "The dimension to be chunked must be divisible by chunk_size "
            f"(got dim_size={dim_size}, chunk_size={chunk_size})."
        )

    num_chunks = dim_size // chunk_size
    tensor_chunks = [
        tensor.chunk(num_chunks, axis=chunk_dim) for tensor in input_tensors
    ]

    outputs: list[Tensor] = []
    for chunk_inputs in zip(*tensor_chunks):
        outputs.append(forward_fn(*chunk_inputs))

    return lucid.concatenate(tuple(outputs), axis=chunk_dim)
