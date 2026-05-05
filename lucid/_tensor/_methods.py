"""
Auto-inject Tensor methods from the ops registry.

Called once at module import time by tensor.py.
"""

from typing import Any, TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._ops._registry import _REGISTRY, OpEntry

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _to_axes(dim):
    """Convert dim/axis (None | int | list[int]) → list[int] for the engine."""
    if dim is None:
        return []
    if isinstance(dim, (list, tuple)):
        return [int(d) for d in dim]
    return [int(dim)]


def _bessel_correct(result_impl, x_impl, axes_list, correction):
    if correction == 0:
        return result_impl
    n = 1
    if axes_list:
        for ax in axes_list:
            n *= int(x_impl.shape[ax])
    else:
        for s in x_impl.shape:
            n *= int(s)
    if n <= correction:
        return result_impl
    scale = float(n) / float(n - correction)
    scale_t = _C_engine.full(
        list(result_impl.shape), scale, result_impl.dtype, result_impl.device
    )
    return _C_engine.mul(result_impl, scale_t)


def _inject_methods(tensor_cls: type) -> None:
    """Attach all registry ops as Tensor methods."""

    def _make_method(e: OpEntry) -> object:
        if e.n_tensor_args == -1:

            def method_list(
                self: Tensor, tensors: list[Tensor], *args: object
            ) -> Tensor:
                all_tensors = [_unwrap(t) for t in [self] + list(tensors)]
                result = e.engine_fn(all_tensors, *args)
                if e.returns_tensor:
                    if isinstance(result, (list, tuple)):
                        return type(result)(_wrap(r) for r in result)
                    return _wrap(result)
                return result

            method_list.__name__ = e.method_name or e.name
            return method_list
        else:

            def method(self: Tensor, *args: object) -> Tensor:
                # Unwrap any Tensor in extra tensor arg positions
                proc_args: list[object] = []
                for i, a in enumerate(args):
                    if i < (e.n_tensor_args - 1) and isinstance(a, tensor_cls):
                        proc_args.append(_unwrap(a))
                    else:
                        proc_args.append(a)

                if e.inplace:
                    result = e.engine_fn(self._impl, *proc_args)
                    self._impl = result
                    return self
                else:
                    result = e.engine_fn(self._impl, *proc_args)
                    if e.returns_tensor:
                        if isinstance(result, (list, tuple)):
                            return type(result)(_wrap(r) for r in result)
                        return _wrap(result)
                    return result

            method.__name__ = e.method_name or e.name
            return method

    for entry in _REGISTRY:
        if entry.method_name is None:
            continue
        # skip dunders — handled in _dunders.py
        if entry.method_name.startswith("__"):
            continue
        setattr(tensor_cls, entry.method_name, _make_method(entry))

    # ── methods not in registry (Python-level implementations) ───────────

    def view(self: Tensor, *shape: int | tuple[int, ...]) -> Tensor:
        """Return a tensor with the same data but a different shape."""
        s = (
            list(shape[0])
            if len(shape) == 1 and isinstance(shape[0], (list, tuple))
            else list(shape)
        )
        return _wrap(_C_engine.reshape(self._impl, s))

    def t(self: Tensor) -> Tensor:
        """Return the 2D transpose of this tensor."""
        return _wrap(_C_engine.T(self._impl))

    # ── PyTorch-compatible reduction methods (override registry versions) ─────

    def sum(
        self: Tensor, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None
    ) -> Tensor:
        ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
        kd = keepdims if keepdims is not None else keepdim
        return _wrap(_C_engine.sum(self._impl, ax, kd))

    def mean(
        self: Tensor, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None
    ) -> Tensor:
        ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
        kd = keepdims if keepdims is not None else keepdim
        return _wrap(_C_engine.mean(self._impl, ax, kd))

    def prod(
        self: Tensor, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None
    ) -> Tensor:
        ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
        kd = keepdims if keepdims is not None else keepdim
        return _wrap(_C_engine.prod(self._impl, ax, kd))

    def max(
        self: Tensor, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None
    ) -> Tensor:
        ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
        kd = keepdims if keepdims is not None else keepdim
        return _wrap(_C_engine.max(self._impl, ax, kd))

    def min(
        self: Tensor, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None
    ) -> Tensor:
        ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
        kd = keepdims if keepdims is not None else keepdim
        return _wrap(_C_engine.min(self._impl, ax, kd))

    def var(
        self: Tensor,
        dim=None,
        keepdim=False,
        *,
        correction=1,
        unbiased=None,
        axis=None,
        axes=None,
        keepdims=None,
    ) -> Tensor:
        """Variance; correction=1 applies Bessel's correction (PyTorch default)."""
        if unbiased is not None:
            correction = 1 if unbiased else 0
        ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
        kd = keepdims if keepdims is not None else keepdim
        v = _C_engine.var(self._impl, ax, kd)
        return _wrap(_bessel_correct(v, self._impl, ax, correction))

    def std(
        self: Tensor,
        dim=None,
        keepdim=False,
        *,
        correction=1,
        unbiased=None,
        axis=None,
        axes=None,
        keepdims=None,
    ) -> Tensor:
        """Std dev; correction=1 applies Bessel's correction (PyTorch default)."""
        if unbiased is not None:
            correction = 1 if unbiased else 0
        ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
        kd = keepdims if keepdims is not None else keepdim
        v = _C_engine.var(self._impl, ax, kd)
        v = _bessel_correct(v, self._impl, ax, correction)
        return _wrap(_C_engine.sqrt(v))

    def argmax(
        self: Tensor, dim=None, keepdim=False, *, axis=None, keepdims=None
    ) -> Tensor:
        d = dim if dim is not None else axis
        kd = keepdims if keepdims is not None else keepdim
        ax = -1 if d is None else int(d)
        return _wrap(_C_engine.argmax(self._impl, ax, kd))

    def argmin(
        self: Tensor, dim=None, keepdim=False, *, axis=None, keepdims=None
    ) -> Tensor:
        d = dim if dim is not None else axis
        kd = keepdims if keepdims is not None else keepdim
        ax = -1 if d is None else int(d)
        return _wrap(_C_engine.argmin(self._impl, ax, kd))

    def reshape(self: Tensor, *shape) -> Tensor:
        """Reshape; accepts t.reshape(d0, d1) or t.reshape([d0, d1]) or t.reshape(-1)."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            s = [int(d) for d in shape[0]]
        elif len(shape) == 1 and isinstance(shape[0], int):
            s = [int(shape[0])]
        else:
            s = [int(d) for d in shape]
        return _wrap(_C_engine.reshape(self._impl, s))

    def permute(self: Tensor, *dims) -> Tensor:
        """Permute axes; accepts t.permute(d0, d1, d2) or t.permute([d0, d1, d2])."""
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            p = [int(d) for d in dims[0]]
        else:
            p = [int(d) for d in dims]
        return _wrap(_C_engine.permute(self._impl, p))

    def expand(self: Tensor, *sizes) -> Tensor:
        """Expand to shape; accepts t.expand(s0, s1) or t.expand([s0, s1])."""
        if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
            s = [int(d) for d in sizes[0]]
        else:
            s = [int(d) for d in sizes]
        return _wrap(_C_engine.expand(self._impl, s))

    def squeeze(self: Tensor, dim=None) -> Tensor:
        """Remove size-1 dims; non-unit dims silently ignored (PyTorch behaviour)."""
        if dim is None:
            return _wrap(_C_engine.squeeze_all(self._impl))
        impl = self._impl
        if isinstance(dim, (list, tuple)):
            ndim = len(impl.shape)
            result = impl
            for d in sorted([int(d) for d in dim], reverse=True):
                nd = d if d >= 0 else ndim + d
                if 0 <= nd < ndim and int(impl.shape[nd]) == 1:
                    result = _C_engine.squeeze(result, nd)
                    ndim -= 1
            return _wrap(result)
        ndim = len(impl.shape)
        d = int(dim)
        nd = d if d >= 0 else ndim + d
        if nd < 0 or nd >= ndim or int(impl.shape[nd]) != 1:
            return _wrap(impl)
        return _wrap(_C_engine.squeeze(impl, nd))

    def repeat(self: Tensor, *sizes) -> Tensor:
        """Tile copies (PyTorch Tensor.repeat semantics).

        Accepts repeat(2, 3) or repeat((2, 3)) — tiles the tensor sizes[i] times
        along each dimension (wraps short size tuples by prepending 1s).
        """
        if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
            reps = list(sizes[0])
        else:
            reps = list(sizes)
        return _wrap(_C_engine.tile(self._impl, [int(r) for r in reps]))

    def repeat_interleave(self: Tensor, repeats: int, dim: int | None = None) -> Tensor:
        """Repeat each element `repeats` times (torch.repeat_interleave semantics)."""
        axis = 0 if dim is None else int(dim)
        return _wrap(_C_engine.repeat(self._impl, int(repeats), axis))

    def split(self: Tensor, split_size_or_sections, dim: int = 0) -> list[Tensor]:
        """Split into chunks of split_size along dim (PyTorch semantics)."""
        axis_size = int(self._impl.shape[dim])
        if isinstance(split_size_or_sections, int):
            chunk_size = split_size_or_sections
            n = (axis_size + chunk_size - 1) // chunk_size
            parts = _C_engine.split(self._impl, n, int(dim))
        else:
            indices: list[int] = []
            cumsum = 0
            for s in split_size_or_sections[:-1]:
                cumsum += s
                indices.append(cumsum)
            parts = _C_engine.split_at(self._impl, indices, int(dim))
        return [_wrap(p) for p in parts]

    def eval(self: Tensor) -> Tensor:
        """Force immediate evaluation of this tensor.

        On Metal (MLX backend) this flushes the lazy computation graph so that
        the graph does not grow unboundedly across training iterations.
        On CPU this is a no-op.  Returns ``self`` for chaining.
        Implemented entirely in C++ — no Python-level mlx import.
        """
        self._impl.eval()
        return self

    def log_softmax(self: Tensor, axis: int = -1) -> Tensor:
        """Return log-softmax along the given axis."""
        sm = _C_engine.softmax(self._impl, axis)
        return _wrap(_C_engine.log(sm))

    def any(self: Tensor) -> Tensor:
        """Return True if any element is non-zero."""
        return _wrap(_C_engine.any(self._impl))

    def all(self: Tensor) -> Tensor:
        """Return True if all elements are non-zero."""
        return _wrap(_C_engine.all(self._impl))

    for _name, _fn in [
        ("view", view),
        ("t", t),
        ("eval", eval),
        ("reshape", reshape),
        ("permute", permute),
        ("expand", expand),
        ("squeeze", squeeze),
        ("sum", sum),
        ("mean", mean),
        ("prod", prod),
        ("max", max),
        ("min", min),
        ("var", var),
        ("std", std),
        ("argmax", argmax),
        ("argmin", argmin),
        ("repeat", repeat),
        ("repeat_interleave", repeat_interleave),
        ("split", split),
        ("log_softmax", log_softmax),
        ("any", any),
        ("all", all),
    ]:
        setattr(tensor_cls, _name, _fn)
