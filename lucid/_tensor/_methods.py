"""
Auto-inject Tensor methods from the ops registry, and the in-place methods
no engine kernel writes.

Every registry method is generated from an ``OpEntry`` in ``_registry.py``;
signature translation (``dim=None|int|list``, ``keepdim``, ``correction``
...) lives in ``_adapters.py``.

The in-place forms whose op has no in-place kernel — the index writes
(``masked_fill_``, ``scatter_`` ...), and ``clamp_`` with a bound that is a
tensor or widens the computation — compute the out-of-place op and write its
result into ``self`` through ``lucid._ops.composite.indexing._write_inplace``:
into ``self``'s storage, refusing a result dtype ``self`` cannot hold,
bumping its version.  ``clamp_`` called the engine kernel without the
promotion ``clamp`` does, so ``int64.clamp_(0.5, 1.5)`` wrote ``[0, 1, 1, 1]``.

Called once at module import time by ``tensor.py``.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._dispatch import (
    _REAL_FLOATS,
    _cannot_cast_inplace,
    _unwrap,
    _wrap,
)
from lucid._tensor._indexing import _adopt_inplace
from lucid._ops._adapters import _clip_adapter, _clip_dtype, _clip_scalar_bounds
from lucid._ops._registry import _REGISTRY, OpEntry, _bind_leading

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid._types import TensorOrScalar


_OPERANDS: dict[str, tuple[str, ...]] = {}

#: Methods whose one argument is a list of ints, which the reference also
#: reads spread out — ``x.permute(1, 0)`` is ``x.permute((1, 0))`` — where
#: its free functions take the list alone (``lucid.permute(x, (1, 0))``).
_SPREAD_INT_LIST = frozenset(
    {"permute", "reshape", "view", "expand", "broadcast_to", "repeat", "tile", "flip"}
)


def _operand_names(e: OpEntry) -> tuple[str, ...]:
    """The advertised names of ``e``'s tensor operands after ``self``.

    Read from the free function's signature, so the method and
    ``lucid.<op>`` bind the same keywords — once per op, on its first
    keyword call: ``lucid._ops`` imports this package, so it cannot be
    imported while the methods are made.
    """
    names = _OPERANDS.get(e.name)
    if names is None:
        from lucid._ops import _leading_tensor_names

        names = _OPERANDS[e.name] = _leading_tensor_names(e)[1:]
    return names


def _fixed_arity_method(e: OpEntry, tensor_cls: type) -> object:
    """The Tensor method for ``e`` when ``self`` is followed by fixed arguments.

    Unwraps the leading ``Tensor`` positional args (up to the op's declared
    tensor-arg count), dispatches to the engine op, and either wraps the
    result as a Tensor or returns the raw value per ``e.returns_tensor``.
    """
    # ``self`` is the only operand of a one-tensor op: nothing to bind.
    binds = e.n_tensor_args > 1
    spreads = e.method_name in _SPREAD_INT_LIST

    def method(self: Tensor, *args: object, **kwargs: object) -> Tensor:
        """Standard-arity method body for registry-driven Tensor methods."""
        # No arguments at all is the empty list spread out: ``x.reshape(*s)``
        # for a 0-d ``s``, which the reference refuses and Lucid's callers
        # write for 0-d tensors.
        if spreads and (len(args) > 1 or not (args or kwargs)):
            args = (args,)
        if kwargs and binds:
            names = _operand_names(e)
            args, kwargs = _bind_leading(method.__name__, names, args, kwargs)
        proc_args: list[object] = []
        for i, a in enumerate(args):
            if i < (e.n_tensor_args - 1) and isinstance(a, tensor_cls):
                proc_args.append(_unwrap(a))  # type: ignore[arg-type]
            else:
                proc_args.append(a)

        result = e.engine_fn(self._impl, *proc_args, **kwargs)
        if e.inplace:
            return _adopt_inplace(self, result, e.name)
        if e.returns_tensor:
            if isinstance(result, (list, tuple)):
                return type(result)(_wrap(r) for r in result)  # type: ignore[return-value]
            return _wrap(result)
        return cast("Tensor", result)

    method.__name__ = e.method_name or e.name
    return method


def _masked_fill_(self: Tensor, mask: Tensor, value: float | Tensor) -> Tensor:
    """Fill elements of ``self`` where ``mask`` is ``True`` with ``value``, in place."""
    from lucid._ops.composite._shared import _is_tensor
    from lucid._ops.composite.indexing import _write_inplace

    if _is_tensor(value):
        if value.ndim != 0:
            raise RuntimeError(
                "masked_fill_ only supports a 0-dimensional value tensor"
            )
        if value.requires_grad:
            import lucid

            return _write_inplace(
                self,
                "masked_fill_",
                lambda base: lucid.where(mask, value.to(base.dtype), base),
                value,
            )
        scalar = float(value.item())
    else:
        scalar = float(value)
    return _write_inplace(
        self, "masked_fill_", lambda base: base.masked_fill(mask, scalar)
    )


def _scatter_(
    self: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor | float,
    reduce: str | None = None,
) -> Tensor:
    """Write ``src`` into ``self`` along ``dim`` at ``index``, in place."""
    from lucid._ops.composite._shared import _is_tensor
    from lucid._ops.composite.indexing import _write_inplace

    if _is_tensor(src):
        return _write_inplace(
            self,
            "scatter_",
            lambda base: base.scatter(dim, index, src, reduce=reduce),
            src,
        )
    return _write_inplace(
        self,
        "scatter_",
        lambda base: base.scatter(dim, index, src, reduce=reduce),
    )


def _scatter_add_(self: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    """Add ``src`` into ``self`` along ``dim`` at ``index``, in place."""
    from lucid._ops.composite.indexing import _write_inplace

    return _write_inplace(
        self,
        "scatter_add_",
        lambda base: base.scatter_add(dim, index, src),
        src,
    )


def _scatter_reduce(
    self: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor,
    reduce: str = "sum",
    *,
    include_self: bool = True,
) -> Tensor:
    """Out-of-place ``scatter_reduce`` along ``dim``."""
    from lucid._ops.composite.indexing import scatter_reduce as _sr

    return _sr(self, dim, index, src, reduce=reduce, include_self=include_self)


def _scatter_reduce_(
    self: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor,
    reduce: str = "sum",
    *,
    include_self: bool = True,
) -> Tensor:
    """In-place ``scatter_reduce_`` along ``dim``."""
    from lucid._ops.composite.indexing import (
        _write_inplace,
        scatter_reduce as _sr,
    )

    return _write_inplace(
        self,
        "scatter_reduce_",
        lambda base: _sr(
            base, dim, index, src, reduce=reduce, include_self=include_self
        ),
        src,
    )


def _put_(
    self: Tensor, index: Tensor, source: Tensor, accumulate: bool = False
) -> Tensor:
    """Write ``source`` into ``self`` at flat ``index`` positions, in place."""
    from lucid._ops.composite.indexing import _write_inplace, put as _put

    return _write_inplace(
        self,
        "put_",
        lambda base: _put(base, index, source, accumulate=accumulate),
        source,
    )


def _masked_scatter(self: Tensor, mask: Tensor, source: Tensor) -> Tensor:
    """Out-of-place ``masked_scatter`` where ``mask`` is ``True``."""
    from lucid._ops.composite.indexing import masked_scatter as _ms

    return _ms(self, mask, source)


def _masked_scatter_(self: Tensor, mask: Tensor, source: Tensor) -> Tensor:
    """In-place ``masked_scatter_`` where ``mask`` is ``True``."""
    from lucid._ops.composite.indexing import (
        _write_inplace,
        masked_scatter as _ms,
    )

    return _write_inplace(
        self,
        "masked_scatter_",
        lambda base: _ms(base, mask, source),
        source,
    )


def _bincount(
    self: Tensor, weights: Tensor | None = None, minlength: int = 0
) -> Tensor:
    """Count occurrences of each value in 1-D non-negative integer ``self``."""
    from lucid._ops.composite.statistics import bincount as _bc

    return _bc(self, weights=weights, minlength=minlength)


def _take_along_dim(self: Tensor, indices: Tensor, dim: int | None = None) -> Tensor:
    """Select values from ``self`` at ``indices`` along ``dim``."""
    from lucid._ops.composite.shape import take_along_dim as _tad

    return _tad(self, indices, dim=dim)


# ── in-place forms with no engine kernel ─────────────────────────────────────


def _write(
    self: Tensor, name: str, result_of: Callable[[Tensor], Tensor], *args: object
) -> Tensor:
    """Make ``result_of(self)`` the values of ``self`` in place; return ``self``.

    ``args`` are the op's other arguments: the tensors among them may carry
    gradient, which decides whether the write records.
    """
    from lucid._ops.composite._shared import _is_tensor
    from lucid._ops.composite.indexing import _write_inplace

    return _write_inplace(self, name, result_of, *(a for a in args if _is_tensor(a)))


def _clamp_inplace(
    self: Tensor,
    min: TensorOrScalar | None,
    max: TensorOrScalar | None,
    name: str,
) -> Tensor:
    """``self`` clamped to ``[min, max]`` in place, by :meth:`Tensor.clamp`'s rules.

    The bounds promote ``self``'s dtype as they do out of place
    (:func:`lucid._ops._adapters._clip_dtype`), and a dtype ``self`` cannot
    hold is refused before anything runs, with the error ``add_`` gives.
    Number bounds that leave the dtype alone take the engine's ``clip_``
    where ``clamp`` takes its ``clip``; anything else is ``clamp``'s result,
    written through :func:`_write`.  One helper for ``clamp_``, ``clip_``,
    ``clamp_min_`` and ``clamp_max_``, so each public spelling is one
    in-place write (one Metal write-back) and names itself in its errors.
    """
    impl = self._impl
    dt = _clip_dtype(impl, min, max)
    if dt != impl.dtype:
        if not _promotion.can_cast(dt, impl.dtype):
            raise _cannot_cast_inplace(dt, impl.dtype)
    else:
        is_float = dt in _REAL_FLOATS
        kernel = _clip_scalar_bounds(impl, min, max, is_float, _C_engine.clip_)
        if kernel is not None:
            return _adopt_inplace(self, kernel, name)

    def result_of(base: Tensor) -> Tensor:
        return _wrap(_clip_adapter(base._impl, min, max))

    return _write(self, name, result_of, min, max)


_EXTRA_METHODS = (
    ("masked_fill_", _masked_fill_),
    ("scatter_", _scatter_),
    ("scatter_add_", _scatter_add_),
    ("scatter_reduce", _scatter_reduce),
    ("scatter_reduce_", _scatter_reduce_),
    ("put_", _put_),
    ("masked_scatter", _masked_scatter),
    ("masked_scatter_", _masked_scatter_),
    ("bincount", _bincount),
    ("take_along_dim", _take_along_dim),
)


def _make_method(e: OpEntry, tensor_cls: type) -> object:
    """Build a Tensor method that dispatches to ``e.engine_fn``."""
    if e.n_tensor_args == -1:

        def method_list(
            self: Tensor,
            tensors: list[Tensor],
            *args: object,
            **kwargs: object,
        ) -> Tensor:
            """Variadic-tensor method body — prepends ``self`` to the unwrapped tensor list."""
            all_tensors = [_unwrap(t) for t in [self] + list(tensors)]
            result = e.engine_fn(all_tensors, *args, **kwargs)
            if e.returns_tensor:
                if isinstance(result, (list, tuple)):
                    return type(result)(_wrap(r) for r in result)  # type: ignore[return-value]
                return _wrap(result)
            return cast("Tensor", result)

        method_list.__name__ = e.method_name or e.name
        return method_list
    return _fixed_arity_method(e, tensor_cls)


def _inject_methods(tensor_cls: type) -> None:
    """Attach all registry ops as Tensor methods."""
    globals().setdefault("Tensor", tensor_cls)

    for entry in _REGISTRY:
        if entry.method_name is None or entry.method_name.startswith("__"):
            continue
        setattr(tensor_cls, entry.method_name, _make_method(entry, tensor_cls))

    def eval(self: Tensor) -> Tensor:
        """Force immediate evaluation of this tensor."""
        self._impl.eval()
        return self

    for _name, _fn in (("eval", eval), *_EXTRA_METHODS):
        _fn.__name__ = _name
        setattr(tensor_cls, _name, _fn)
