"""In-place writes through a view of a Metal tensor reach the tensor it views.

On the CPU a view and its base share one buffer, so ``x[0].add_(1)`` changes
``x``.  A Metal tensor is an MLX array, and an MLX array is a value: indexing,
``view``, ``transpose`` and the rest hand back a new array, and an in-place op
on that array replaced it alone.  ``x[0].add_(1)``, ``x[:, 0].zero_()``,
``x.t()[1].mul_(2)``, ``r[1][0] = v``, ``p.data.sub_(lr * g)`` and
``p.grad.mul_(s)`` all left ``x``, ``r`` and ``p`` as they were on Metal.  None
raised, while the CPU and the reference framework wrote through.  A
hand-written optimizer step on Metal trained nothing.

A view taken on Metal now remembers its base and how to write itself back.
Every value-writing in-place method on such a view, ``__setitem__`` included,
writes it back once the op is done:

* basic indexing writes ``base[key] = view``;
* order-preserving reshapes (``view``, ``reshape``, ``flatten``, ``squeeze``
  and the like) write ``base.copy_(view.reshape(base.shape))``;
* ``.data`` and ``detach()`` write ``base.copy_(view)`` outside autograd, and
  ``.grad`` sets its owner's gradient, as the shared storage they stand for
  would;
* every other view (``transpose``, ``permute``, ``T``, ``narrow``, ``split``,
  ``diagonal``, ...) replays itself on an index tensor to learn where each
  element came from, and writes there.

The write-back is itself an in-place write to the base, so a view of a view
writes all the way up.  Some writes are refused, as the CPU and the reference
refuse them: a view whose elements overlap (``expand``, ``unfold`` with
overlapping windows), and a view of a leaf that requires grad while autograd
records.

A view op has up to three spellings — the method ``x.squeeze(0)``, the free
function ``lucid.squeeze(x, 0)`` and, for a whole view, the constructor
``lucid.Tensor(x)`` — and each one links its result, from the one table of
view ops below.  Only the method did at first: ``lucid.squeeze(x, 0).add_(1)``
left ``x`` as it was on Metal while ``x.squeeze(0).add_(1)`` wrote through.

One difference remains.  A view taken *before* its base is written does not
see that write on Metal.

CPU tensors are untouched: their views alias the buffer already.
"""

import functools
import inspect
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING

from lucid._C import engine as _C_engine

if TYPE_CHECKING:
    from lucid._ops._registry import OpEntry
    from lucid._tensor.tensor import Tensor
    from lucid._types import _IndexType

_ATTR = "_metal_view"

_DETACHED = 0  # .data / detach(): copy into the base, outside autograd
_KEY = 1  # basic indexing: base[key] = view
_RESHAPE = 2  # order-preserving: base.copy_(view.reshape(base.shape))
_TRACED = 3  # anything else: replay on an index tensor, write by position
_GRAD = 4  # .grad: the owner's gradient is set to the written value

#: View ops whose result reads the base's elements in the base's order.
_RESHAPES = (
    "view",
    "reshape",
    "flatten",
    "unflatten",
    "ravel",
    "squeeze",
    "squeeze_all",
    "unsqueeze",
    "expand_dims",
    "view_as",
    "reshape_as",
)
#: View ops whose result is a rearrangement or a part of the base.
_REARRANGEMENTS = (
    "transpose",
    "swapaxes",
    "permute",
    "movedim",
    "t",
    "narrow",
    "diagonal",
)
#: View ops whose result may read one base element more than once.
_OVERLAPPING = ("expand", "expand_as", "broadcast_to", "as_strided", "unfold")
#: View ops that return several views of disjoint parts.
_PIECES = ("split", "chunk", "unbind")
#: In-place methods that set flags rather than values.
_NOT_VALUE_WRITES = frozenset({"requires_grad_", "detach_", "share_memory_"})
#: Augmented assignment, which Python routes to these in place.
_AUGMENTED = ("__iadd__", "__isub__", "__imul__", "__itruediv__", "__ipow__")


class _Link:
    """How a Metal view writes itself back into the tensor it was taken from."""

    __slots__ = (
        "_base",
        "_weak",
        "kind",
        "key",
        "replay",
        "overlap",
        "positions",
        "shape",
        "stride",
        "stride_known",
    )

    def __init__(
        self,
        base: Tensor,
        kind: int,
        *,
        key: _IndexType = None,
        replay: (
            tuple[str, tuple[object, ...], dict[str, object], int | None] | None
        ) = None,
        overlap: bool = False,
        weak: bool = False,
        stride: tuple[int, ...] | None = None,
        stride_known: bool = False,
    ) -> None:
        # A detached alias may outlive its base by a long way — a loss
        # detached into a history list — and a strong reference would keep
        # the base's whole autograd graph alive with it.  If the base is
        # gone, nothing is left to see a write.
        self._weak = weak
        self._base: object = weakref.ref(base) if weak else base
        self.kind = kind
        self.key = key
        self.replay = replay
        self.overlap = overlap
        self.positions: Tensor | None = None
        # Kept so positions, stride checks and an overlap refusal do not need
        # the base, which may be gone.
        self.shape: tuple[int, ...] = tuple(base.shape)
        self.stride: tuple[int, ...] | None = stride
        self.stride_known: bool = stride_known or stride is not None

    @property
    def base(self) -> Tensor | None:
        """The tensor this view was taken from, or ``None`` once it is gone."""
        if self._weak:
            ref: Callable[[], Tensor | None] = self._base  # type: ignore[assignment]
            return ref()
        return self._base  # type: ignore[return-value]


#: :class:`lucid.Tensor`, bound by :func:`install` — an import here would
#: run on every view taken.
_TENSOR: type[Tensor]
#: Compared by value: every view op on every device asks, and the enum's own
#: ``==`` costs as much again as reading ``device`` (~0.23 µs each).
_GPU = _C_engine.Device.GPU.value


def _on_metal(t: Tensor) -> bool:
    return t._impl.device.value == _GPU


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    stride = [1] * len(shape)
    for d in range(len(shape) - 2, -1, -1):
        stride[d] = stride[d + 1] * max(shape[d + 1], 1)
    return tuple(stride)


def _is_contiguous_stride(shape: tuple[int, ...], stride: tuple[int, ...]) -> bool:
    expected = 1
    for d in range(len(shape) - 1, -1, -1):
        if shape[d] == 0:
            return True
        if shape[d] != 1:
            if stride[d] != expected:
                return False
            expected *= shape[d]
    return True


def _squeezes_only(
    in_shape: tuple[int, ...],
    in_stride: tuple[int, ...],
    out_shape: tuple[int, ...],
) -> tuple[int, ...] | None:
    sizes = [s for s in in_shape if s != 1]
    kept = [st for s, st in zip(in_shape, in_stride, strict=True) if s != 1]
    want = [s for s in out_shape if s != 1]
    if want != sizes:
        return None
    stride = [0] * len(out_shape)
    k = len(kept)
    next_st = 1
    for d in range(len(out_shape) - 1, -1, -1):
        if out_shape[d] != 1:
            k -= 1
            stride[d] = kept[k]
        else:
            stride[d] = next_st
        next_st = stride[d] * out_shape[d]
    return tuple(stride)


def _fill_chunk(
    stride: list[int],
    out_shape: tuple[int, ...],
    view_d: int,
    in_numel: int,
    chunk_stride: int,
) -> tuple[int, int]:
    view_numel = 1
    while view_d >= 0 and (view_numel < in_numel or out_shape[view_d] == 1):
        stride[view_d] = view_numel * chunk_stride
        view_numel *= out_shape[view_d]
        view_d -= 1
    return view_d, view_numel


def _chunk_strides(
    in_shape: tuple[int, ...],
    in_stride: tuple[int, ...],
    out_shape: tuple[int, ...],
) -> tuple[int, ...] | None:
    if not in_shape:
        return (1,) * len(out_shape)
    if 0 in in_shape:
        return in_stride if in_shape == out_shape else _contiguous_stride(out_shape)
    stride = [0] * len(out_shape)
    view_d = len(out_shape) - 1
    chunk_stride = in_stride[-1]
    in_numel = 1
    for k in range(len(in_shape) - 1, -1, -1):
        if in_shape[k] == 1 and k > 0 and in_numel == 1:
            chunk_stride = in_stride[k - 1]
            continue
        in_numel *= in_shape[k]
        if k > 0 and (
            in_shape[k - 1] == 1 or in_stride[k - 1] == in_numel * chunk_stride
        ):
            continue
        view_d, view_numel = _fill_chunk(
            stride, out_shape, view_d, in_numel, chunk_stride
        )
        if view_numel != in_numel:
            return None
        if k > 0:
            chunk_stride = in_stride[k - 1]
            in_numel = 1
    return tuple(stride) if view_d == -1 else None


def _view_strides(
    in_shape: tuple[int, ...],
    in_stride: tuple[int, ...],
    out_shape: tuple[int, ...],
    squeeze: bool,
) -> tuple[int, ...] | None:
    if squeeze:
        return _squeezes_only(in_shape, in_stride, out_shape)
    if _is_contiguous_stride(in_shape, in_stride):
        return _contiguous_stride(out_shape)
    return _chunk_strides(in_shape, in_stride, out_shape)


def _probe_stride(link: _Link, in_stride: tuple[int, ...]) -> tuple[int, ...] | None:
    from lucid._dispatch import _wrap
    from lucid.compile import _host_check

    with _host_check():
        span = (
            sum(
                (s - 1) * max(st, 0)
                for s, st in zip(link.shape, in_stride, strict=True)
                if s > 0
            )
            + 1
        )
        seed = _C_engine.empty([span], _C_engine.I8, _C_engine.CPU)
        probe = _wrap(
            _C_engine.TensorImpl._make_view(seed, list(link.shape), list(in_stride), 0)
        )
        if link.kind == _KEY:
            traced = probe[link.key]
        else:
            assert link.replay is not None
            name, args, kwargs, piece = link.replay
            attr = getattr(probe, name)
            traced = attr(*args, **kwargs) if callable(attr) else attr
            if piece is not None:
                traced = traced[piece]
        if 0 in link.shape or traced._impl.is_aliased():
            return traced.stride()
        return None


def _stride(t: Tensor) -> tuple[int, ...] | None:
    link: _Link | None = t.__dict__.get(_ATTR)
    if link is None:
        return _contiguous_stride(t.shape)
    if link.stride_known:
        return link.stride
    base = link.base
    in_stride = (
        _stride(base)
        if base is not None and _ATTR in base.__dict__
        else _contiguous_stride(link.shape)
    )
    if in_stride is None:
        out_stride = None
    elif link.kind in (_DETACHED, _GRAD):
        out_stride = in_stride
    elif link.kind == _RESHAPE:
        out_stride = _view_strides(link.shape, in_stride, t.shape, False)
    else:
        out_stride = _probe_stride(link, in_stride)
    link.stride = out_stride
    link.stride_known = True
    return out_stride


def _link(view: object, base: Tensor, kind: int, **kwargs: object) -> None:
    if not isinstance(view, _TENSOR) or view is base or view._impl is base._impl:
        return
    # A base that is not itself a view is held weakly.  Holding it strongly
    # kept it alive as long as any view of it: ``q, k, v = qkv.chunk(3)``
    # held all of ``qkv`` until backward had used q, k and v, a whole
    # projection per layer.  If the base is gone, nobody can see a write to
    # it.  A base that *is* a view — the ``view(-1)`` in
    # ``buf.view(-1)[5].add_(1)``, which nothing else references — is held
    # strongly, or the chain would break before the write reached ``buf``.
    if _ATTR not in base.__dict__:
        kwargs["weak"] = True
    view.__dict__[_ATTR] = _Link(base, kind, **kwargs)  # type: ignore[arg-type]


def _is_basic(key: object) -> bool:
    parts = key if isinstance(key, tuple) else (key,)
    for part in parts:
        if part is None or part is Ellipsis:
            continue
        if isinstance(part, bool) or not isinstance(part, (int, slice)):
            return False
        if isinstance(part, slice) and not all(
            v is None or (isinstance(v, int) and not isinstance(v, bool))
            for v in (part.start, part.stop, part.step)
        ):
            return False
    return True


def _positions(link: _Link) -> Tensor:
    """Where each element of the view sits in its base, flattened, on the CPU."""
    if link.positions is None:
        import lucid

        assert link.replay is not None
        numel = 1
        for n in link.shape:
            numel *= n
        index = lucid.arange(numel, dtype=lucid.int64).reshape(link.shape)
        # The same view, taken on the index: each element names its source.
        name, args, kwargs, piece = link.replay
        attr = getattr(index, name)
        traced = attr(*args, **kwargs) if callable(attr) else attr
        if piece is not None:
            traced = traced[piece]
        link.positions = traced.reshape(-1)
    return link.positions


def _overlaps(link: _Link) -> bool:
    import lucid

    positions = _positions(link)
    if positions.numel() < 2:
        return False
    ordered = lucid.sort(positions)
    return bool((ordered[1:] == ordered[:-1]).any().item())


def _check(view: Tensor) -> None:
    """Refuse, before anything is written, what the write-back could not honour."""
    link: _Link | None = view.__dict__.get(_ATTR)
    detached = False
    while link is not None:
        if link.overlap and _overlaps(link):
            raise RuntimeError(
                "an in-place write into a tensor whose elements overlap (an expanded "
                "view, or unfold windows that overlap) is ambiguous — clone() first"
            )
        base = link.base
        if base is None:
            return
        detached = detached or link.kind in (_DETACHED, _GRAD)
        if (
            not detached
            and _C_engine.grad_enabled()
            and base.requires_grad
            and base.is_leaf
        ):
            raise RuntimeError(
                "a view of a leaf tensor that requires grad cannot be modified in "
                "place — wrap the call in no_grad, or use the out-of-place form"
            )
        link = base.__dict__.get(_ATTR)


def write_back(view: Tensor) -> None:
    """Write ``view``'s values into the tensor it is a Metal view of, if it is one.

    Parameters
    ----------
    view : Tensor
        A tensor that has just been written in place.  Nothing happens unless
        it was taken as a view of a Metal tensor.
    """
    link: _Link | None = view.__dict__.get(_ATTR)
    if link is None:
        return
    base = link.base
    if base is None:
        return
    import lucid

    if link.kind == _DETACHED:
        with lucid.no_grad():
            base.copy_(view)
    elif link.kind == _GRAD:
        with lucid.no_grad():
            base.grad = view
    elif link.kind == _KEY:
        base[link.key] = view
    elif link.kind == _RESHAPE:
        base.copy_(view.reshape(tuple(base.shape)))
    else:
        flat = base.reshape(-1)
        flat[_positions(link).to(base.device)] = view.reshape(-1)
        if _ATTR not in flat.__dict__:
            base.copy_(flat.reshape(tuple(base.shape)))


def _writer(original: Callable[..., object]) -> Callable[..., object]:
    @functools.wraps(original)
    def method(self: Tensor, *args: object, **kwargs: object) -> object:
        if _ATTR not in self.__dict__:
            return original(self, *args, **kwargs)
        _check(self)
        out = original(self, *args, **kwargs)
        write_back(self)
        return out

    return method


def _metal_input(args: tuple[object, ...]) -> Tensor | None:
    """The tensor a view op was taken from, if it is a Metal tensor.

    It comes first in every spelling: ``self`` of the method, ``input`` of
    the free function.  A free function may also be handed something that
    is not a tensor at all, and then there is nothing to link.
    """
    if args:
        base = args[0]
        if isinstance(base, _TENSOR) and base._impl.device.value == _GPU:
            return base
    return None


def _link_reshape(out: Tensor, base: Tensor, name: str) -> None:
    if _ATTR not in base.__dict__:
        _link(out, base, _RESHAPE, stride=_contiguous_stride(out.shape))
        return
    base_stride = _stride(base)
    squeeze = name in ("squeeze", "squeeze_all", "unsqueeze", "expand_dims")
    new_stride = (
        _view_strides(base.shape, base_stride, out.shape, squeeze)
        if base_stride is not None
        else None
    )
    if new_stride is not None:
        _link(out, base, _RESHAPE, stride=new_stride)
    elif name in ("view", "view_as"):
        raise _C_engine.LucidError(
            "view: view size is not compatible with input tensor's "
            "size and stride (at least one dimension spans across "
            "two contiguous subspaces) — use .reshape(...) instead"
        )


def _viewer(
    original: Callable[..., object], kind: int, name: str, overlap: bool = False
) -> Callable[..., object]:
    """Wrap one spelling of a view op (method or free function) to link its result.

    ``name`` is the method a traced view replays on an index tensor.  A free
    function wrapped here has the same parameters after its input as the
    method of that name, so ``lucid.name(x, *args)`` replays as
    ``index.name(*args)``.
    """

    @functools.wraps(original)
    def view(*args: object, **kwargs: object) -> object:
        out = original(*args, **kwargs)
        # _metal_input, inlined: this runs on every view op on every device.
        base = args[0] if args else None
        if isinstance(base, _TENSOR) and base._impl.device.value == _GPU:
            if kind == _TRACED:
                _link(
                    out,
                    base,
                    kind,
                    replay=(name, args[1:], kwargs, None),
                    overlap=overlap,
                )
            elif kind == _RESHAPE and isinstance(out, _TENSOR):
                _link_reshape(out, base, name)
            else:
                _link(out, base, kind)
        return out

    return view


def _pieces(original: Callable[..., object], name: str) -> Callable[..., object]:
    """:func:`_viewer` for a view op that returns several views of disjoint parts."""

    @functools.wraps(original)
    def view(*args: object, **kwargs: object) -> object:
        out = original(*args, **kwargs)
        base = args[0] if args else None
        if (
            isinstance(base, _TENSOR)
            and base._impl.device.value == _GPU
            and isinstance(out, (tuple, list))
        ):
            for i, piece in enumerate(out):
                _link(
                    piece,
                    base,
                    _TRACED,
                    replay=(name, args[1:], kwargs, i),
                )
        return out

    return view


def _constructor(original: Callable[..., object]) -> Callable[..., object]:
    """Wrap ``Tensor.__init__`` so ``lucid.Tensor(x)`` of a Metal ``x`` is linked.

    ``Tensor(x)`` is a whole view of ``x`` — the CPU shares the storage — and
    an in-place write into it reaches ``x`` as a write into ``x.view(shape)``
    does.  Anything that is not such a view (Python data, a copy to another
    dtype) is left alone.
    """

    @functools.wraps(original)
    def __init__(self: Tensor, *args: object, **kwargs: object) -> None:
        original(self, *args, **kwargs)
        base = _metal_input(args if args else (kwargs.get("data"),))
        if base is not None:
            impl, source = self._impl, base._impl
            if (
                impl.device.value == _GPU
                and impl.dtype == source.dtype
                and impl.shape == source.shape
            ):
                _link(self, base, _RESHAPE)

    return __init__


def _makers() -> dict[str, Callable[[Callable[..., object]], Callable[..., object]]]:
    """Every view op by name, with the wrapper that links its result.

    The one table both spellings of a view op are wrapped from, so a method
    and its free function cannot disagree about whether — or how — a write
    goes back.
    """
    makers: dict[str, Callable[[Callable[..., object]], Callable[..., object]]] = {}
    for name in _RESHAPES:
        makers[name] = functools.partial(_viewer, kind=_RESHAPE, name=name)
    for name in _REARRANGEMENTS:
        makers[name] = functools.partial(_viewer, kind=_TRACED, name=name)
    for name in _OVERLAPPING:
        makers[name] = functools.partial(_viewer, kind=_TRACED, name=name, overlap=True)
    for name in _PIECES:
        makers[name] = functools.partial(_pieces, name=name)
    makers["detach"] = functools.partial(_viewer, kind=_DETACHED, name="detach")
    return makers


def _same_positions(free: OpEntry, method: OpEntry) -> bool:
    """Whether ``free`` and ``method`` read the same arguments by position."""
    from lucid._ops import _signature_for_entry

    free_kinds = [p.kind for p in _signature_for_entry(free).parameters.values()]
    method_kinds = [p.kind for p in _signature_for_entry(method).parameters.values()]
    return (
        free.n_tensor_args == method.n_tensor_args
        and free_kinds == method_kinds
        and inspect.Parameter.KEYWORD_ONLY not in free_kinds
    )


def _by_position(
    linked: Callable[..., object], signature: inspect.Signature
) -> Callable[..., object]:
    """``linked``, called with every argument in position, as its replay is."""

    @functools.wraps(linked)
    def call(*args: object, **kwargs: object) -> object:
        if kwargs:
            bound = signature.bind(*args, **kwargs)
            return linked(*bound.args, **bound.kwargs)
        return linked(*args)

    return call


def _install_free_functions(
    makers: dict[str, Callable[[Callable[..., object]], Callable[..., object]]],
) -> None:
    """Wrap ``lucid.<name>`` for every view op whose method is wrapped too.

    Only a free function generated from the same registry entry as the
    method of its name is wrapped: that is what makes ``lucid.name(x, *a)``
    and ``x.name(*a)`` the same call, which a traced view's replay relies
    on.  Free functions written by hand (``lucid.swapaxes``, ``lucid.t``,
    ``lucid.hsplit`` ...) are built from these and are linked through them.
    """
    import sys

    import lucid._ops as ops
    from lucid._ops._registry import _REGISTRY

    by_method: dict[str, OpEntry] = {}
    by_free: dict[str, OpEntry] = {}
    for entry in _REGISTRY:
        if entry.method_name is not None:
            by_method[entry.method_name] = entry  # the last one is the method
        if entry.free_fn_name is not None:
            by_free.setdefault(entry.free_fn_name, entry)  # the first one is free
    top = sys.modules.get("lucid")
    for name, make in makers.items():
        shared = by_free.get(name)
        method = by_method.get(name)
        if shared is None or method is None:
            continue
        generated = ops.__dict__.get(name)
        if generated is None:
            continue
        if method is shared:
            linked = make(generated)
        elif _same_positions(shared, method):
            # The reference names a parameter of the method otherwise than
            # the free function's (``split_size`` / ``split_size_or_sections``):
            # the replay is recorded by position, which the two share.
            linked = _by_position(make(generated), inspect.signature(generated))
        else:
            continue
        setattr(ops, name, linked)
        # ``lucid.<name>`` is copied from ``lucid._ops`` the first time any op
        # is looked up there; if that already happened, replace the copy.
        if top is not None and top.__dict__.get(name) is generated:
            setattr(top, name, linked)


def _getitem(original: Callable[..., object]) -> Callable[..., object]:
    @functools.wraps(original)
    def method(self: Tensor, key: object) -> object:
        out = original(self, key)
        if _on_metal(self) and _is_basic(key):
            parts = key if isinstance(key, tuple) else (key,)
            if any(part is None for part in parts):
                _link(out, self, _TRACED, replay=("__getitem__", (key,), {}, None))
            else:
                _link(out, self, _KEY, key=key)
        return out

    return method


def _property(prop: property, kind: int) -> property:
    getter = prop.fget
    assert getter is not None

    def fget(self: Tensor) -> object:
        out = getter(self)
        if out is not None and _on_metal(self):
            name = getter.__name__
            if kind == _TRACED:
                _link(out, self, kind, replay=(name, (), {}, None))
            else:
                _link(out, self, kind)
        return out

    functools.update_wrapper(fget, getter)
    return property(fget, prop.fset, prop.fdel, prop.__doc__)


def install(cls: type) -> None:
    """Wrap every spelling of each view op and every in-place method, once at import.

    The view-producing methods of ``cls``, its constructor, and the
    ``lucid.<name>`` free functions of the same view ops are wrapped to link
    their Metal results; the value-writing in-place methods to write back.

    Parameters
    ----------
    cls : type
        :class:`lucid.Tensor`.
    """
    global _TENSOR
    _TENSOR = cls
    for name in dir(cls):
        if (
            name.endswith("_")
            and not name.startswith("_")
            and name not in _NOT_VALUE_WRITES
        ):
            attr = getattr(cls, name, None)
            if callable(attr):
                setattr(cls, name, _writer(attr))
    for name in (*_AUGMENTED, "__setitem__"):
        if hasattr(cls, name):
            setattr(cls, name, _writer(getattr(cls, name)))
    setattr(cls, "__getitem__", _getitem(getattr(cls, "__getitem__")))
    makers = _makers()
    for name, make in makers.items():
        if hasattr(cls, name):
            setattr(cls, name, make(getattr(cls, name)))
    _install_free_functions(makers)
    setattr(cls, "__init__", _constructor(getattr(cls, "__init__")))
    for name, kind in (
        ("T", _TRACED),
        ("mT", _TRACED),
        ("data", _DETACHED),
        ("grad", _GRAD),
    ):
        prop = cls.__dict__.get(name)
        if isinstance(prop, property):
            setattr(cls, name, _property(prop, kind))
