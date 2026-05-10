# lucid/func/__init__.pyi
# Stub for lucid.func — functional transforms (vmap, grad, vjp, jvp, …).
# fmt: off
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

__all__: list[str]

# ── vmap ──────────────────────────────────────────────────────────────────────

def vmap(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    in_dims: int | tuple[int | None, ...] = ...,
    out_dims: int | tuple[int, ...] = ...,
    randomness: str = ...,
    *,
    chunk_size: int | None = ...,
) -> Callable[..., Tensor | tuple[Tensor, ...]]: ...

# ── grad / grad_and_value ─────────────────────────────────────────────────────

def grad(
    func: Callable[..., Tensor],
    argnums: int | tuple[int, ...] = ...,
    has_aux: bool = ...,
) -> Callable[..., Tensor | tuple[Tensor | None, ...]]: ...

def grad_and_value(
    func: Callable[..., Tensor],
    argnums: int | tuple[int, ...] = ...,
    has_aux: bool = ...,
) -> Callable[..., tuple[Tensor | tuple[Tensor | None, ...], Tensor]]: ...

# ── vjp / jvp ─────────────────────────────────────────────────────────────────

def vjp(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    *primals: Tensor,
    has_aux: bool = ...,
) -> tuple[Tensor | tuple[Tensor, ...], Callable[..., tuple[Tensor | None, ...]]]: ...

def jvp(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    primals: tuple[Tensor, ...],
    tangents: tuple[Tensor, ...],
    strict: bool = ...,
) -> tuple[Tensor | tuple[Tensor, ...], Tensor | tuple[Tensor, ...]]: ...

# ── linearize ─────────────────────────────────────────────────────────────────

def linearize(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    *primals: Tensor,
) -> tuple[Tensor | tuple[Tensor, ...], Callable[..., Tensor | tuple[Tensor, ...]]]: ...

# ── jacobians ─────────────────────────────────────────────────────────────────

def jacrev(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    argnums: int | tuple[int, ...] = ...,
    has_aux: bool = ...,
    *,
    chunk_size: int | None = ...,
) -> Callable[..., Tensor | tuple[Tensor | None, ...]]: ...

def jacfwd(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    argnums: int | tuple[int, ...] = ...,
    has_aux: bool = ...,
    *,
    randomness: str = ...,
) -> Callable[..., Tensor | tuple[Tensor | None, ...]]: ...

# ── hessian ───────────────────────────────────────────────────────────────────

def hessian(
    func: Callable[..., Tensor],
    argnums: int | tuple[int, ...] = ...,
) -> Callable[..., Tensor | tuple[Tensor | None, ...]]: ...
