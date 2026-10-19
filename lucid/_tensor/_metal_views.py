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

Only what the CPU reads in place is linked: a slice with a negative step is
gathered into a copy there, so on Metal it is no view either.

The write-back is itself an in-place write to the base, so a view of a view
writes all the way up; past a base whose elements overlap (``expand``,
``unfold`` with overlapping windows), it goes straight to the storage and the
views between are taken again.  Some writes are refused, as the CPU and the
reference refuse them: one into a tensor whose own elements overlap, unless
it is a fill, and one into a view of a leaf that requires grad while
autograd records.

A view also has the layout the reference gives it: ``stride()``,
``storage_offset()`` and ``is_contiguous()`` say where its elements sit in
its base's storage, and ``as_strided`` counts positions in that storage —
``y.t().as_strided((2,), (1,))`` reads ``y``'s first two elements, not the
transpose's.

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
import math
import operator
import weakref
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, SupportsIndex, TypeIs

from lucid._C import engine as _C_engine
from lucid._tensor._view_layout import (
    _LAYOUT_RULES,
    _Dims,
    _Geometry,
    _Layout,
    _contiguous_stride,
    _dense_order,
    _index_rule,
    _is_contiguous_stride,
    _view_strides,
)

if TYPE_CHECKING:
    from lucid._ops._registry import OpEntry
    from lucid._tensor.tensor import Tensor
    from lucid._types import _IndexType

_ATTR = "_metal_view"
_EPOCH = "_metal_epoch"

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
#: ``as_strided`` may too, and is linked by :func:`_as_strided`.
_OVERLAPPING = ("expand", "expand_as", "broadcast_to", "unfold")
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
        "_epoch",
        "kind",
        "key",
        "replay",
        "overlap",
        "positions",
        "shape",
        "layout",
        "skips",
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
        layout: _Layout | None = None,
        positions: Tensor | None = None,
        skips: tuple[weakref.ref[Tensor], ...] = (),
    ) -> None:
        # A detached alias may outlive its base by a long way — a loss
        # detached into a history list — and a strong reference would keep
        # the base's whole autograd graph alive with it.  If the base is
        # gone, nothing is left to see a write.
        self._base: Tensor | weakref.ref[Tensor] = weakref.ref(base) if weak else base
        self._epoch: object = base.__dict__.get(_EPOCH)
        self.kind = kind
        self.key = key
        self.replay = replay
        self.overlap = overlap
        self.positions: Tensor | None = positions
        # Kept so positions, layouts and an overlap refusal do not need the
        # base, which may be gone.
        self.shape: tuple[int, ...] = tuple(base.shape)
        self.layout: _Layout | None = layout
        # The views an ``as_strided`` window was read past, to a tensor of
        # their chain that holds every element of it: a write through the
        # window would reach that tensor and leave them as they were.
        self.skips = skips

    @property
    def base(self) -> Tensor | None:
        """The tensor this view was taken from, or ``None`` once it or its storage is gone."""
        b = self._base() if isinstance(self._base, weakref.ref) else self._base
        if b is not None and b.__dict__.get(_EPOCH) is not self._epoch:
            return None
        return b


#: :class:`lucid.Tensor`, bound by :func:`install` — an import here would
#: run on every view taken.
_TENSOR: type[Tensor]
#: Compared by value: every view op on every device asks, and the enum's own
#: ``==`` costs as much again as reading ``device`` (~0.23 µs each).
_GPU = _C_engine.Device.GPU.value


def _on_metal(t: Tensor) -> bool:
    return t._impl.device.value == _GPU


# ── The layout a traced view would have on the CPU ───────────────────────────


def _derived(link: _Link, source: _Layout, out: _Dims) -> _Layout:
    """A view's layout from its input's, by the rule of the op that took it."""
    if link.kind == _DETACHED:
        return source
    if link.kind == _KEY:
        geometry = _Geometry(link.shape, source.stride, out, source.offset)
        return _index_rule(geometry, link.key)
    # A reshape and an ``as_strided`` window are linked with their layouts.
    assert link.replay is not None
    name, args, kwargs, piece = link.replay
    geometry = _Geometry(link.shape, source.stride, out, source.offset, piece or 0)
    return _LAYOUT_RULES[name](geometry, *args, **kwargs)


def _own_layout(t: Tensor) -> _Layout:
    """The layout the engine gives ``t`` itself.

    A Metal tensor is packed, or a member of one of the engine's own view
    families, which carries its geometry in its root.
    """
    impl = t._impl
    size = _C_engine.dtype_size(impl.dtype)
    stride = tuple(s // size for s in impl.stride)
    return _Layout(stride, impl.storage_offset_bytes() // size)


def _layout(t: Tensor) -> _Layout:
    """Where ``t``'s elements sit in the storage the reference would give it.

    A tensor that is no Metal view has the engine's layout, and so does a
    ``.grad``: the gradient is no view of the tensor it belongs to.  A base
    that is gone is taken to have been packed.
    """
    link: _Link | None = t.__dict__.get(_ATTR)
    if link is None or link.kind == _GRAD:
        return _own_layout(t)
    if link.layout is None:
        base = link.base
        if base is None or _ATTR not in base.__dict__ and base._impl.is_dense():
            source = _Layout(_contiguous_stride(link.shape), 0)
        else:
            source = _layout(base)
        link.layout = _derived(link, source, tuple(t.shape))
    return link.layout


def _link(view: object, base: Tensor, kind: int, **kwargs: object) -> None:
    if not isinstance(view, _TENSOR) or view is base or view._impl is base._impl:
        return
    # A base that is not itself a view is held weakly.  Holding it strongly
    # kept it alive as long as any view of it: ``q, k, v = qkv.chunk(3)``
    # held all of ``qkv`` until backward had used q, k and v, a whole
    # projection per layer.  If the base is gone, nobody can see a write to
    # it.  A base that *is* a view — the ``view(-1)`` in
    # ``buf.view(-1)[5].add_(1)``, which nothing else references — is held
    # strongly, or the chain would break before the write reached ``buf``,
    # and so is a copy only the view reaches (``weak=False``).
    if _ATTR not in base.__dict__ and not (base.requires_grad and base.is_leaf):
        kwargs.setdefault("weak", True)
    view.__dict__[_ATTR] = _Link(base, kind, **kwargs)  # type: ignore[arg-type]


def _is_view_key(key: object) -> bool:
    """Whether ``x[key]`` reads ``x`` in place on the CPU, so Metal links it.

    The CPU's basic indexing reads ``None``, ``...``, an integer (anything
    ``__index__`` makes one, but a bool) and a slice that steps forwards in
    place.  A slice that steps backwards is gathered into a copy, and so is
    every index tensor, list and mask.
    """
    parts = key if isinstance(key, tuple) else (key,)
    return all(_is_basic(part) and not _backwards(part) for part in parts)


def _is_basic(part: object) -> bool:
    return (
        part is None
        or part is Ellipsis
        or isinstance(part, slice)
        or _is_index_scalar(part)
    )


def _is_index_scalar(part: object) -> bool:
    if isinstance(part, int):
        return not isinstance(part, bool)
    if isinstance(part, list) or getattr(part, "ndim", 0) != 0:
        return False  # an index tensor, list or array, which the CPU gathers by
    # The CPU's own reading of a key part decides the rest — a 0-d integer
    # tensor or NumPy integer is an integer to it, a bool a mask — so the two
    # cannot disagree.
    from lucid._tensor._indexing import _token

    token = _token(part)
    return isinstance(token, int) and not isinstance(token, bool)


def _backwards(part: object) -> bool:
    if not isinstance(part, slice) or part.step is None:
        return False
    step = part.step
    if isinstance(step, int):
        return step < 0
    # A step that is no integer is refused by the indexing itself, in its words.
    return _is_integer(step) and operator.index(step) < 0


def _is_integer(value: object) -> TypeIs[SupportsIndex]:
    # ``int`` first: a runtime protocol check costs microseconds.
    return isinstance(value, (int, SupportsIndex))


def _spelled_out(parts: tuple[object, ...], ndim: int) -> list[object]:
    """``parts`` with ``...`` written out as the full slices it stands for."""
    taken = sum(1 for part in parts if part is not None and part is not Ellipsis)
    spelled: list[object] = []
    for part in parts:
        spelled.extend([slice(None)] * (ndim - taken) if part is Ellipsis else [part])
    return spelled


def _split_at_copy(
    key: object, t: Tensor
) -> tuple[tuple[object, ...], tuple[object, ...]] | None:
    """A basic key the CPU reads as a copy and then a view of it, split there.

    The CPU applies a basic key a part at a time, and a slice that steps
    backwards gathers all it has so far into a copy; the parts after the
    last one view that copy.  Returns the key up to that slice and the key
    the rest is of the copy — ``None`` when no slice steps backwards, when
    the key is no basic key, or when the rest views all of the copy.
    """
    parts = key if isinstance(key, tuple) else (key,)
    if not any(_backwards(part) for part in parts) or not all(
        _is_basic(part) for part in parts
    ):
        return None
    spelled = _spelled_out(parts, t.ndim)
    last = max(i for i, part in enumerate(spelled) if _backwards(part))
    head, rest = tuple(spelled[: last + 1]), spelled[last + 1 :]
    if all(isinstance(part, slice) and part == slice(None) for part in rest):
        return None
    kept = sum(1 for part in head if part is None or isinstance(part, slice))
    return head, (*[slice(None)] * kept, *rest)


def _replayed(link: _Link, base: Tensor, shape: _Dims) -> Tensor:
    """The view ``link`` describes, of shape ``shape``, taken again of ``base``."""
    if link.kind == _DETACHED:
        return base
    if link.kind == _KEY:
        return base[link.key]
    if link.kind == _RESHAPE:
        return base.reshape(shape)
    assert link.replay is not None
    name, args, kwargs, piece = link.replay
    attr = getattr(base, name)
    out: Tensor = attr(*args, **kwargs) if callable(attr) else attr
    return out if piece is None else out[piece]


def _positions(link: _Link) -> Tensor:
    """Where each element of the view sits in its base, flattened, on the CPU."""
    if link.positions is None:
        import lucid

        index = lucid.arange(math.prod(link.shape), dtype=lucid.int64)
        # The same view, taken on the index: each element names its source.
        traced = _replayed(link, index.reshape(link.shape), ())
        link.positions = traced.reshape(-1)
    return link.positions


def _self_overlaps(view: Tensor) -> bool:
    """Whether two of ``view``'s elements are one element of its storage.

    Only an ``expand``, overlapping ``unfold`` windows or an ``as_strided``
    window make elements share, so the storage is read only below one.
    """
    import lucid

    link: _Link | None = view.__dict__.get(_ATTR)
    while link is not None and not link.overlap:
        base = link.base
        link = None if base is None else base.__dict__.get(_ATTR)
    if link is None or view.numel() < 2:
        return False
    positions = _window_positions(tuple(view.shape), *_layout(view))
    ordered = lucid.sort(positions)
    return bool((ordered[1:] == ordered[:-1]).any().item())


#: The writes that leave one value in every element, so elements that share a
#: position agree on it: they may write an overlapping view, as in the engine
#: (``refuse_ambiguous_write``, core/TensorImpl.cpp) and the reference.
_FILLS = frozenset({"fill_", "zero_"})


def _check(view: Tensor, *, allow_overlap: bool = False) -> None:
    """Refuse, before anything is written, what the write-back could not honour."""
    # As the reference refuses: a write into the written tensor's own
    # elements that share a place, unless it leaves one value in all of them.
    # A view of such a tensor that shares none may be written.
    if not allow_overlap and _self_overlaps(view):
        raise RuntimeError(
            "an in-place write into a tensor whose elements overlap (an expanded "
            "view, or unfold windows that overlap) is ambiguous — clone() first"
        )
    link: _Link | None = view.__dict__.get(_ATTR)
    detached = False
    while link is not None:
        if any(ref() is not None for ref in link.skips):
            raise RuntimeError(
                "as_strided: an in-place write into a Metal window that reads "
                "elements outside the view it was taken from, while that view "
                "is alive, is not supported — take the window from the view's "
                "base, or write into a clone()"
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

    if _self_overlaps(base):
        _write_past_overlaps(view)
    elif link.kind == _DETACHED:
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


def _write_past_overlaps(view: Tensor) -> None:
    """Write ``view`` into its storage's owner, past a base whose elements overlap.

    Written back a link at a time, the overlapping base would scatter all its
    elements, those that share a place with ``view``'s included, and the
    stale ones would win.  ``view``'s own elements go straight to their place
    in the storage instead, and each tensor between is taken again from it.
    """
    import lucid

    frames, _ = _frames(view)
    owner = frames[-1]
    picked = _picked(owner, (tuple(view.shape), *_layout(view)))
    if picked is None:
        return  # the owner is gone, and with it every place to write to
    flat = owner.reshape(-1)
    flat[picked.to(owner.device)] = view.reshape(-1)
    if _ATTR not in flat.__dict__:
        owner.copy_(flat.reshape(tuple(owner.shape)))
    with lucid.no_grad():
        for k in range(len(frames) - 2, 0, -1):
            link = frames[k].__dict__[_ATTR]
            again = _replayed(link, frames[k + 1], tuple(frames[k].shape))
            frames[k]._impl.copy_from(again._impl)


def _writer(original: Callable[..., object]) -> Callable[..., object]:
    allow_overlap = getattr(original, "__name__", "") in _FILLS

    @functools.wraps(original)
    def method(self: Tensor, *args: object, **kwargs: object) -> object:
        if _ATTR not in self.__dict__:
            return original(self, *args, **kwargs)
        _check(self, allow_overlap=allow_overlap)
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
    shape = tuple(out.shape)
    # Any shape views a packed base, row-major.  An empty result keeps its
    # base's strides, and a member of an engine view family has a geometry
    # of its own, which a reshape may not step through.
    if _ATTR not in base.__dict__ and 0 not in shape and base._impl.is_dense():
        _link(out, base, _RESHAPE, layout=_Layout(_contiguous_stride(shape), 0))
        return
    source = _layout(base)
    squeeze = name in ("squeeze", "squeeze_all", "unsqueeze", "expand_dims")
    new_stride = _view_strides(tuple(base.shape), source.stride, shape, squeeze)
    if new_stride is not None:
        _link(out, base, _RESHAPE, layout=_Layout(new_stride, source.offset))
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


# ── as_strided: a window onto the storage ────────────────────────────────────


def _window(
    size: object, stride: object, storage_offset: object = None
) -> tuple[_Dims, _Dims, int | None]:
    """``as_strided``'s window, from the arguments after its input, in elements.

    Python binds the arguments, so a missing, extra or unknown one is its own
    ``TypeError``; each is then checked as the reference checks it —
    integers only, so ``size=(2.0,)`` and ``size=(True,)`` are refused — on
    every device, before anything is read.  A negative ``storage_offset``
    stands for the input's own, as the engine reads it.
    """
    if storage_offset is not None and not _is_count(storage_offset):
        raise TypeError(
            "as_strided(): argument 'storage_offset' must be int or None, not "
            f"{type(storage_offset).__name__}"
        )
    at = None if storage_offset is None else operator.index(storage_offset)
    return (
        _integers("size", size),
        _integers("stride", stride),
        None if at is None or at < 0 else at,
    )


# Python names the op in the errors it raises binding the arguments.
_window.__name__ = _window.__qualname__ = "as_strided"


def _is_count(value: object) -> TypeIs[SupportsIndex]:
    # A bool is no count here, as the reference has it.
    return _is_integer(value) and not isinstance(value, bool)


def _integers(name: str, value: object) -> _Dims:
    if not isinstance(value, Sequence):
        raise TypeError(
            f"as_strided(): argument {name!r} must be a sequence of ints, not "
            f"{type(value).__name__}"
        )
    for at, item in enumerate(value):
        if not _is_count(item):
            raise TypeError(
                f"as_strided(): argument {name!r} must be a sequence of ints, but "
                f"found an element of type {type(item).__name__} at position {at}"
            )
    return tuple(operator.index(item) for item in value)


def _frames(t: Tensor) -> tuple[list[Tensor], int]:
    """The live tensors ``t``'s storage is reached through, ``t`` first, and its size.

    The chain ends at the storage's owner — a tensor that is no view, or a
    ``.grad`` — or, once the owner is gone, at the last tensor still alive.
    """
    chain = [t]
    link: _Link | None = t.__dict__.get(_ATTR)
    while link is not None and link.kind != _GRAD:
        base = link.base
        if base is None:
            return chain, math.prod(link.shape)
        chain.append(base)
        link = base.__dict__.get(_ATTR)
    owner = chain[-1]
    if not owner.numel():
        return chain, 0
    # As far as the owner reaches: all of a packed tensor, and as much of
    # its root as a member of an engine view family reads.
    layout = _layout(owner)
    reach = sum((n - 1) * s for n, s in zip(owner.shape, layout.stride, strict=True))
    return chain, layout.offset + reach + 1


def _check_window(size: _Dims, stride: _Dims, offset: int, capacity: int) -> int | None:
    """Refuse what the engine refuses; the window's last position, ``None`` if empty.

    The engine checks a window against its input's own elements, and a Metal
    view's storage may have gone with its owner, so the storage is checked
    here, in the engine's words.
    """
    if len(size) != len(stride):
        raise _C_engine.LucidError("as_strided: size and stride need the same length")
    if any(n < 0 for n in size) or any(s < 0 for s in stride):
        raise _C_engine.InvalidArgument(
            "as_strided: size and stride must be non-negative"
        )
    if 0 in size:
        return None
    last = offset + sum((n - 1) * s for n, s in zip(size, stride, strict=True))
    if last >= capacity:
        raise _C_engine.LucidError(
            "as_strided: the view reaches past the end of the storage"
        )
    return last


def _holds(frame: Tensor, first: int, last: int | None) -> tuple[list[int], int] | None:
    """How ``frame`` holds storage positions ``first`` to ``last`` as a dense block.

    The axis order that lays ``frame`` out in storage order, and how far
    into it ``first`` is; ``None`` when ``frame`` is no dense block or does
    not hold them all.  An empty window (``last`` is ``None``) reads nothing,
    and any tensor holds it.
    """
    if last is None:
        return list(range(frame.ndim)), 0
    layout = _layout(frame)
    order = _dense_order(tuple(frame.shape), layout.stride)
    start = first - layout.offset
    if order is None or start < 0 or last - layout.offset >= frame.numel():
        return None
    return order, start


def _window_positions(size: _Dims, stride: _Dims, start: int) -> Tensor:
    """``start + sum(i * stride)`` over the indices ``i`` of ``size``, on the CPU."""
    import lucid

    at = lucid.full(size, start, dtype=lucid.int64)
    for d, (n, s) in enumerate(zip(size, stride, strict=True)):
        steps = lucid.arange(n, dtype=lucid.int64) * s
        at = at + steps.reshape([n if e == d else 1 for e in range(len(size))])
    return at.reshape(-1)


#: ``as_strided``'s window in the storage's elements: size, stride, offset.
type _Window = tuple[_Dims, _Dims, int]


def _picked(frame: Tensor, window: _Window) -> Tensor | None:
    """Which of ``frame``'s elements each of the window's is, by flat index.

    ``None`` when the window reads an element ``frame`` does not hold.
    Found by sorting ``frame``'s positions in the storage: a strided view
    (``y[:, ::2]``, a diagonal) is no dense block, though it may hold every
    element of a window built from its own strides.
    """
    import lucid

    if not frame.numel():
        return None
    layout = _layout(frame)
    shape = tuple(frame.shape)
    size, stride, offset = window
    # Positions outside the span ``frame`` reaches cannot be among its own.
    reach = sum((n - 1) * s for n, s in zip(shape, layout.stride, strict=True))
    span = sum((n - 1) * s for n, s in zip(size, stride, strict=True))
    if offset < layout.offset or offset + span > layout.offset + reach:
        return None
    own = _window_positions(shape, layout.stride, layout.offset)
    order = lucid.argsort(own)
    ordered = own[order]
    wanted = _window_positions(*window)
    at = lucid.searchsorted(ordered, wanted).clamp(max=ordered.numel() - 1)
    if not bool((ordered[at] == wanted).all().item()):
        return None
    return order[at]


def _read_block(
    original: Callable[..., object],
    frame: Tensor,
    held: tuple[list[int], int],
    window: _Window,
    skips: tuple[weakref.ref[Tensor], ...] = (),
) -> object:
    """The window, taken by ``as_strided`` of the dense block ``frame`` is."""
    size, stride, offset = window
    order, start = held
    block = frame if order == sorted(order) else frame.permute(order)
    out = original(block, size, stride, start)
    _link(
        out,
        block,
        _TRACED,
        replay=("as_strided", window, {}, None),
        overlap=True,
        layout=_Layout(stride, offset),
        positions=_window_positions(size, stride, start),
        skips=skips,
    )
    return out


def _read_picked(
    frame: Tensor,
    picked: Tensor,
    window: _Window,
    skips: tuple[weakref.ref[Tensor], ...] = (),
) -> object:
    """The window, gathered from ``frame``'s elements at ``picked``."""
    size, stride, offset = window
    out = frame.reshape(-1)[picked.reshape(size).to(frame.device)]
    _link(
        out,
        frame,
        _TRACED,
        replay=("as_strided", window, {}, None),
        overlap=True,
        layout=_Layout(stride, offset),
        positions=picked,
        skips=skips,
    )
    return out


def _strided_window(
    original: Callable[..., object],
    v: Tensor,
    window: tuple[_Dims, _Dims, int | None],
) -> object:
    """``v.as_strided(...)`` of a Metal view, read from the storage ``v`` views.

    The elements come from ``v`` itself when it holds them all, so a write
    through the window reaches ``v`` and a gradient flows into it; otherwise
    from the nearest tensor of ``v``'s chain that holds them, outside
    autograd.  A gradient that way would skip ``v``, and is refused as the
    CPU refuses it; so is a write while ``v`` or a view between lives, as it
    would keep its old values and write them back over the window's.
    """
    import lucid

    size, stride, given = window
    offset = _layout(v).offset if given is None else given
    whole: _Window = (size, stride, offset)
    frames, capacity = _frames(v)
    last = _check_window(size, stride, offset, capacity)
    held = _holds(v, offset, last)
    if held is not None:
        return _read_block(original, v, held, whole)
    picked = _picked(v, whole)
    if picked is not None:
        return _read_picked(v, picked, whole)
    if _C_engine.grad_enabled() and v.requires_grad:
        raise NotImplementedError(
            "as_strided: a gradient through a Metal view that does not hold "
            "every element of the window — call contiguous() first and take "
            "the window of that"
        )
    with lucid.no_grad():
        for k in range(1, len(frames)):
            held = _holds(frames[k], offset, last)
            if held is not None:
                skips = tuple(weakref.ref(f) for f in frames[:k])
                return _read_block(original, frames[k], held, whole, skips)
        # The storage's owner is gone: what is left of it is the last tensor alive.
        picked = _picked(frames[-1], whole) if len(frames) > 1 else None
        if picked is not None:
            skips = tuple(weakref.ref(f) for f in frames[:-1])
            return _read_picked(frames[-1], picked, whole, skips)
    raise RuntimeError(
        "as_strided: the window reads elements of a Metal tensor that has been "
        "freed — keep the tensor this view was taken from alive"
    )


def _as_strided(original: Callable[..., object]) -> Callable[..., object]:
    """Wrap one spelling of ``as_strided`` so a Metal window reads its storage.

    ``size``, ``stride`` and ``storage_offset`` count elements of the storage
    the input views, which for a view is not its own elements in order.  The
    window is linked with the call as the CPU would make it, which is what
    ``lucid.save`` replays.
    """

    @functools.wraps(original)
    def view(*args: object, **kwargs: object) -> object:
        base = args[0] if args else None
        if not isinstance(base, _TENSOR):
            return original(*args, **kwargs)
        window = _window(*args[1:], **kwargs)
        if base._impl.device.value != _GPU:
            return original(*args, **kwargs)
        if _ATTR in base.__dict__:
            return _strided_window(original, base, window)
        out = original(*args, **kwargs)
        size, stride, given = window
        offset = 0 if given is None else given
        _link(
            out,
            base,
            _TRACED,
            replay=("as_strided", (size, stride, offset), {}, None),
            overlap=True,
            layout=_Layout(stride, offset),
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
                _link(self, base, _RESHAPE, layout=_layout(base))

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
    makers["as_strided"] = _as_strided
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
        if self._impl.device.value != _GPU:
            return original(self, key)
        if type(key) is int or type(key) is slice and not _backwards(key):
            out = original(self, key)  # the common keys, read in place
            _link(out, self, _KEY, key=key)
            return out
        split = _split_at_copy(key, self)
        if split is not None:
            # The CPU's view of its copy keeps the copy, which nothing else
            # holds, alive: a window of the view may read the rest of it.
            head, rest = split
            copy = original(self, head)
            out = original(copy, rest)
            if isinstance(copy, _TENSOR):
                _link_key(out, copy, rest, weak=False)
            return out
        out = original(self, key)
        if _is_view_key(key):
            _link_key(out, self, key)
        return out

    return method


def _link_key(out: object, base: Tensor, key: object, **kwargs: object) -> None:
    """Link ``base[key]`` — replayed when ``None`` inserts an axis, else by key."""
    parts = key if isinstance(key, tuple) else (key,)
    if any(part is None for part in parts):
        kwargs["replay"] = ("__getitem__", (key,), {}, None)
        _link(out, base, _TRACED, **kwargs)
    else:
        _link(out, base, _KEY, key=key, **kwargs)


# ── What a Metal view reports of its layout ──────────────────────────────────


def _stride_of(t: Tensor, dim: SupportsIndex | None = None) -> _Dims | int:
    stride = _layout(t).stride
    return stride if dim is None else stride[operator.index(dim)]


def _storage_offset_of(t: Tensor) -> int:
    return _layout(t).offset


def _is_contiguous_of(t: Tensor) -> bool:
    return _is_contiguous_stride(tuple(t.shape), _layout(t).stride)


#: The layout queries a Metal view answers from :func:`_layout` — its place in
#: its base's storage, as the reference gives it, not its own packed buffer.
_REPORTS: dict[str, Callable[..., object]] = {
    "stride": _stride_of,
    "storage_offset": _storage_offset_of,
    "is_contiguous": _is_contiguous_of,
}


def _reporter(
    original: Callable[..., object], report: Callable[..., object]
) -> Callable[..., object]:
    # A query of a tensor that is no Metal view goes straight on, unpacked:
    # ``stride()`` is read in loops.
    if report is _stride_of:

        @functools.wraps(original)
        def stride(self: Tensor, dim: SupportsIndex | None = None) -> object:
            if _ATTR in self.__dict__:
                return report(self, dim)
            return original(self) if dim is None else original(self, dim)

        return stride

    @functools.wraps(original)
    def query(self: Tensor) -> object:
        if _ATTR in self.__dict__:
            return report(self)
        return original(self)

    return query


def _rebind_view(target: Tensor, source: object) -> None:
    """Propagate ``source``'s Metal view link to ``target`` after rebinding ``_impl``."""
    link = source.__dict__.get(_ATTR) if isinstance(source, _TENSOR) else None
    if link is not None:
        target.__dict__[_ATTR] = link
    else:
        target.__dict__.pop(_ATTR, None)


def _detach_in_place(original: Callable[..., object]) -> Callable[..., object]:
    """Wrap ``Tensor.detach_`` so a Metal view stays linked outside autograd."""

    @functools.wraps(original)
    def detach_(self: Tensor) -> object:
        old_link: _Link | None = self.__dict__.get(_ATTR)
        if old_link is None or old_link.kind == _DETACHED:
            return original(self)
        old_impl = self._impl
        out = original(self)
        holder: Tensor = _TENSOR.__new_from_impl__(old_impl)
        holder.__dict__[_ATTR] = old_link
        _link(self, holder, _DETACHED)
        return out

    return detach_


def _property(prop: property, kind: int) -> property:
    getter = prop.fget
    assert getter is not None
    original_set = prop.fset

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
    if original_set is not None and kind == _DETACHED:

        def fset(self: Tensor, value: object) -> None:
            was_metal = _on_metal(self)
            original_set(self, value)
            if was_metal or _on_metal(self):
                self.__dict__[_EPOCH] = object()
            _rebind_view(self, value)

        functools.update_wrapper(fset, original_set)
        return property(fget, fset, prop.fdel, prop.__doc__)
    return property(fget, original_set, prop.fdel, prop.__doc__)


def _install_writers(cls: type) -> None:
    """Wrap each value-writing in-place method of ``cls`` to write back."""
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
    _install_writers(cls)
    setattr(cls, "__getitem__", _getitem(getattr(cls, "__getitem__")))
    makers = _makers()
    makers["detach_"] = _detach_in_place
    for name, make in makers.items():
        if hasattr(cls, name):
            setattr(cls, name, make(getattr(cls, name)))
    _install_free_functions(makers)
    for name, report in _REPORTS.items():
        setattr(cls, name, _reporter(getattr(cls, name), report))
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
