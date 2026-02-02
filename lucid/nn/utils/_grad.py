from typing import Iterable

import lucid

from lucid._tensor import Tensor
from lucid.types import _Scalar


__all__ = ["grad_norm", "clip_grad_norm", "clip_grad_value"]


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


def clip_grad_norm(
    parameters: Iterable[Tensor] | Tensor,
    max_norm: _Scalar,
    norm_type: int = 2,
    eps: float = 1e-7,
) -> float:
    params: list[Tensor] = [p for p in _as_iter(parameters) if p.grad is not None]
    total_norm = grad_norm(params, norm_type=norm_type)

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
