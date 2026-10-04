"""
Index ranges: the one place that decides whether an integer index lies
inside the axis it indexes.

Every ``nn.functional`` op that reads with an integer index is checked
here, before the gather or kernel that would read with it. That covers a
class target of a class-index loss, a ``one_hot`` class, an embedding row,
and ``ctc_loss``'s labels, ``blank`` and sequence lengths. Each used to have
a policy of its own:

- the class-index losses checked after their gather;
- the embedding table was read on the host on every device;
- ``one_hot`` and ``ctc_loss`` did not check at all.

As a result, ``one_hot`` quietly gave a zero row for a class past the end,
and ``ctc_loss`` with an ``input_lengths`` past the input read beyond it
and returned a number.

Policy
------
**CPU.** An index outside its axis raises ``IndexError`` before any gather
or kernel runs.

- The index is on the host already, so reading it waits on no device.
- The read is the engine's own comparison and reduction over the index
  (:func:`_refuse`). A Python scan of the values costs about 25 ns an
  entry, which at a segmentation target's millions of entries takes
  longer than the loss itself. A refusal therefore dispatches the guard's
  reads (the clamp, the comparison, ``all``, and any mask they need) and
  none of the consumer's kernels.
- The guard is not a value: it raises or passes, and nothing downstream
  depends on what it read. So it runs outside any active compile trace,
  and a compiled step is validated when it is traced.

**Metal.** The index is never read back, because a host read in every
call would stall every training step. An out-of-range index is isolated
inside the graph instead, and never reaches memory outside its buffer:

- The class-index losses clamp before every gather and make the sample's
  factor NaN (:func:`_poison`). A bad label then makes the loss NaN,
  rather than being scored as class ``0`` or ``C - 1``.
- The engine's index kernels clamp and mask the rest: a float gather
  reads NaN, an integer gather reads 0, a scatter drops the update, and
  ``one_hot`` gives a zero row.

Two consumers read on Metal anyway, each for its own reason:

- ``ctc_loss`` runs its recursions on the CPU, so its labels and lengths
  make the round trip regardless.
- ``embedding`` and ``embedding_bag`` keep their host read until the
  engine isolates the table gather. Until then, an index far past the
  table reads past it.

``ctc_loss`` is stricter than the reference in two places:

- The reference accepts a label outside ``[0, C)`` and reads past the
  class axis. The loss it gives is meaningless, so such a label is refused
  here.
- An ``input_lengths`` entry of 0 is refused for now. The reference takes
  an empty sequence (its loss is 0 for an empty target, inf otherwise),
  but Lucid's kernel would write outside its buffer for one. This is
  temporary, until the kernel handles empty sequences.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple, cast, overload

import lucid as _lucid
from lucid._C import engine as _C_engine
from lucid._dtype import dtype as _dtype
from lucid._unsupported import unsupported_if

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _host_read[T](read: Callable[[], T]) -> T:
    """Run ``read``, a guard's read of an index, outside any active compile
    trace.

    A read inside the trace would mark it unsupported (a traced value read
    on the host) and send every step that holds a guard to eager.
    """
    tracer = _C_engine.compile.current_tracer()
    _C_engine.compile.set_current_tracer(None)
    try:
        return read()
    finally:
        _C_engine.compile.set_current_tracer(tracer)


def _as_index(index: Tensor) -> Tensor:
    """``index`` as an integer tensor.

    A float or bool index is truncated to ``int64``, as its consumer's own
    cast would do, and is checked in that form. An integer index is
    returned untouched, so it is checked at its own width: an ``int64``
    value past ``int32``'s range is refused rather than wrapped into the
    range by a later cast (the one exception is a Metal class target; see
    :func:`_class_targets`).
    """
    if index.dtype in (_lucid.int8, _lucid.int16, _lucid.int32, _lucid.int64):
        return index
    return index.to(dtype=_lucid.int64)


def _in_range(
    index: Tensor, extent: int, counted: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """``index`` clamped into ``[0, extent)``, and where that changed nothing.

    Returns ``(clamped, usable)``. ``clamped`` is what a gather can read
    with safely. ``usable`` is ``False`` exactly at an entry outside the
    axis, unless ``counted`` is ``False`` there: an ``ignore_index``
    sentinel and the padding after a label list are allowed to be anything.
    """
    if extent <= 0:
        # No entry fits an empty axis, and clip would be handed an upper
        # bound below its lower one.
        clamped = _lucid.zeros_like(index)
        usable = _lucid.zeros_like(index, dtype=_lucid.bool_)
    else:
        clamped = _lucid.clip(index, 0, extent - 1)
        usable = clamped == index
    if counted is not None:
        usable = usable | ~counted
    return clamped, usable


def _guard(
    index: Tensor, extent: int, counted: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """``(index, usable)`` for a guard whose clamp nothing reads: ``index``
    as an integer tensor, and where it lies inside ``[0, extent)``.

    Run it through :func:`_host_read`, so that a compile trace or an
    export records none of it.
    """
    index = _as_index(index)
    return index, _in_range(index, extent, counted)[1]


def _flat(values: object) -> list[object]:
    """``Tensor.tolist()``'s nested lists, or its scalar, as one flat list."""
    out: list[object] = []
    stack: list[object] = [values]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            out.append(item)
    return out


def _refuse(index: Tensor, usable: Tensor, message: Callable[[object], str]) -> None:
    """The guard's one read: raise ``IndexError(message(v))`` for the first
    entry ``v`` of ``index`` that is not ``usable``, or return.

    Every range check of an index tensor ends here; only ``ctc_loss``'s
    lengths, which are host metadata, are read elsewhere
    (:func:`_check_lengths`). The pass is one ``all`` over the mask the
    clamp already needed. The failure path finds the first offending entry
    with ``tolist``, which dispatches no op.
    """
    if _host_read(lambda: bool(usable.all().item())):
        return
    first = _host_read(
        lambda: next(
            value
            for value, ok in zip(_flat(index.tolist()), _flat(usable.tolist()))
            if not ok
        )
    )
    raise IndexError(message(first))


# ── the class-index losses ──────────────────────────────────────────────────


class _ClassTargets(NamedTuple):
    """The class targets of a class-index loss, checked
    (:func:`_class_targets`)."""

    #: The targets clamped into the class range, as ``int32``: what every
    #: gather reads with.
    safe: Tensor
    #: ``False`` at the entries the loss does not read as a class, or
    #: ``None`` when it reads every entry.
    counted: Tensor | None
    #: Metal only: ``False`` at each out-of-range counted target, for
    #: :func:`_poison`.  ``None`` on the CPU, where such a target raised.
    usable: Tensor | None


def _class_targets(
    target: Tensor,
    num_classes: int,
    op: str,
    *,
    ignore_index: int | None = None,
    counted: Tensor | None = None,
) -> _ClassTargets:
    """Check the class targets of a class-index loss, before its gather.

    The guard of ``cross_entropy``, ``nll_loss``, ``multi_margin_loss`` and
    ``multilabel_margin_loss``.  The entries a loss does not read as a
    class may hold anything: the ``ignore_index`` positions (pass
    ``ignore_index``), or the padding after a label list (pass the
    ``counted`` mask).  Any other target outside ``[0, num_classes)``
    raises ``IndexError("<op>: Target N is out of bounds.")`` for a CPU
    ``target``, and is flagged in ``usable`` for a Metal one.

    A CPU target is checked at its own width, so an ``int64`` target past
    ``int32``'s range is refused rather than wrapped into the range.  A
    Metal target is checked and gathered as ``int32``, the width these
    losses always used there, and an ``int64`` target that wide wraps.  It
    must not be compared at its own width: an ``int64`` comparison aborts
    the graph compiler when the ``int64`` came from an ``argmax`` inside a
    compiled step, because the graph holds that result as ``int32``
    (LCD-270).
    """
    index = _as_index(target)
    on_cpu = target.device == "cpu"
    if not on_cpu:
        index = index.to(dtype=_lucid.int32)
    if ignore_index is not None:
        counted = index != ignore_index
    clamped, usable = _in_range(index, num_classes, counted)
    if on_cpu:
        _refuse(index, usable, lambda v: f"{op}: Target {v} is out of bounds.")
        return _ClassTargets(clamped.to(dtype=_lucid.int32), counted, None)
    return _ClassTargets(clamped, counted, usable)


@overload
def _poison(
    scale: Tensor, usable: Tensor | None, like: Tensor, dtype: _dtype
) -> Tensor: ...
@overload
def _poison(
    scale: None, usable: Tensor | None, like: Tensor, dtype: _dtype
) -> Tensor | None: ...
def _poison(
    scale: Tensor | None,
    usable: Tensor | None,
    like: Tensor,
    dtype: _dtype,
) -> Tensor | None:
    """The per-sample factor ``scale``, NaN at each out-of-range target.

    ``usable`` comes from :func:`_class_targets`. ``None``, which is the CPU
    case where the guard has raised already, returns ``scale`` unchanged.
    ``scale`` of ``None`` means all ones, of the shape of ``like`` and the
    dtype ``dtype``.
    """
    if usable is None:
        return scale
    ones = scale if scale is not None else _lucid.ones_like(like, dtype=dtype)
    return _lucid.where(usable, ones, _lucid.full_like(ones, float("nan")))


# ── tables and class axes ───────────────────────────────────────────────────


def _check_indices(
    index: Tensor,
    extent: int,
    op: str,
    *,
    what: str,
    axis: str,
    read_on_metal: bool = False,
) -> None:
    """Refuse an entry of ``index`` outside ``[0, extent)`` with
    ``IndexError``, before the consumer reads with it.

    The message is ``"<op>: <what> <v> is out of range for <axis> (valid
    range [0, extent - 1])"``. A Metal ``index`` is checked only with
    ``read_on_metal``, for the two consumers the module docstring names;
    any other Metal consumer is isolated in the graph and not read back.
    An empty ``index`` has nothing to check.
    """
    if index.numel() == 0:
        return
    if index.device != "cpu" and not read_on_metal:
        return
    index, usable = _host_read(lambda: _guard(index, extent))
    _refuse(
        index,
        usable,
        lambda v: (
            f"{op}: {what} {v} is out of range for {axis} "
            f"(valid range [0, {extent - 1}])"
        ),
    )


#: The integer dtypes an embedding table can be indexed with.
_TABLE_INDEX_DTYPES = (
    _lucid.int8,
    _lucid.int16,
    _lucid.int32,
    _lucid.int64,
    _lucid.bool_,
)


def _check_table(x: Tensor, weight: Tensor, op: str) -> None:
    """Check the indices ``x`` into the rows of an embedding table
    ``weight``, for ``embedding`` and ``embedding_bag``.

    A float ``x`` raises ``TypeError``. It would pass the range check
    trivially, since its values sit in ``[0, 1)``, and the engine gather
    would then read its bits as integers: ``F.embedding(float_tensor,
    weight)`` used to end in a segmentation fault. An index outside
    ``[0, num_embeddings)`` raises ``IndexError``, on both devices for now
    (see the module docstring).
    """
    if x.dtype not in _TABLE_INDEX_DTYPES:
        raise TypeError(
            f"{op}: indices must be an integer tensor, got {x.dtype}; "
            f"cast with .to(lucid.int64) first"
        )
    rows = int(weight.shape[0])
    _check_indices(
        x,
        rows,
        op,
        what="index",
        axis=f"a table with {rows} entries",
        read_on_metal=True,
    )


# ── ctc_loss ────────────────────────────────────────────────────────────────


def _check_lengths(
    lengths: Tensor, name: str, batch: int, longest: int | None, op: str
) -> list[int]:
    """One length per sequence, each in ``[0, longest]`` (``longest`` of
    ``None`` bounds it below only). Returns the lengths as Python ints.

    A wrong count raises ``ValueError``, because the kernel reads one
    length per sequence. A length outside its range raises ``IndexError``,
    because the kernel would read that far along the axis.
    """
    shape = tuple(int(d) for d in lengths.shape)
    if shape != (batch,):
        raise ValueError(
            f"{op}: {name} must hold one length per sequence, shape "
            f"({batch},), got {shape}"
        )
    values = [int(cast(int, v)) for v in _flat(_host_read(lengths.tolist))]
    for v in values:
        if v < 0 or (longest is not None and v > longest):
            bound = "be non-negative" if longest is None else f"lie in [0, {longest}]"
            raise IndexError(f"{op}: each of {name} must {bound}, got {v}")
    return values


def _check_ctc(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
    blank: int,
    op: str,
) -> list[int]:
    """Check every index ``ctc_loss`` reads with, before its kernel.

    - ``log_probs`` must be ``(T, N, C)``.
    - ``blank`` must lie in ``[0, C)``.
    - ``input_lengths`` must hold one length per sequence, each in
      ``[1, T]``.  A length of 0 is valid in the reference and raises
      ``NotImplementedError`` here for now: the kernel cannot run an
      empty sequence yet.
    - ``targets`` must be padded ``(N, S)`` with each ``target_lengths``
      entry in ``[0, S]``, or concatenated ``(sum(target_lengths),)``.
    - Every label the kernel reads must lie in ``[0, C)``. The padding past
      a sequence's length is not read, so it can hold anything.

    These are read on the host on both devices; the module docstring gives
    the reason. A wrong shape or count raises ``ValueError``. An index
    outside its axis raises ``IndexError``.

    Returns
    -------
    list of int
        ``target_lengths`` as Python ints, which the caller uses to
        gather a padded target into one list.
    """
    if log_probs.ndim != 3:
        raise ValueError(
            f"{op}: log_probs must be (T, N, C), or (T, C) for one sequence, "
            f"got shape {tuple(int(d) for d in log_probs.shape)}"
        )
    time, batch, classes = (int(d) for d in log_probs.shape)
    if not 0 <= blank < classes:
        raise IndexError(
            f"{op}: blank {blank} is out of range for {classes} classes "
            f"(valid range [0, {classes - 1}])"
        )
    if targets.ndim not in (1, 2):
        raise ValueError(
            f"{op}: targets must be padded (N, S) or concatenated (sum of "
            f"target_lengths,), got shape {tuple(int(d) for d in targets.shape)}"
        )
    padded = targets.ndim == 2
    if padded and int(targets.shape[0]) != batch:
        raise ValueError(
            f"{op}: padded targets must hold one row per sequence ({batch}), "
            f"got {int(targets.shape[0])}"
        )
    frames = _check_lengths(input_lengths, "input_lengths", batch, time, op)
    # Temporary, and stricter than the reference: an empty sequence is
    # valid (its loss is 0 for an empty target, inf otherwise), but the
    # kernel writes before its buffer for one.  Lift once it handles them.
    unsupported_if(
        0 in frames,
        op,
        "input_lengths",
        0,
        detail="An empty input sequence is not supported yet (LCD-257).",
    )
    width = int(targets.shape[1]) if padded else None
    lengths = _check_lengths(target_lengths, "target_lengths", batch, width, op)
    if not padded and sum(lengths) != int(targets.numel()):
        raise ValueError(
            f"{op}: concatenated targets hold {int(targets.numel())} labels, "
            f"but target_lengths add up to {sum(lengths)}"
        )
    if targets.numel() == 0:
        return lengths
    counted: Tensor | None = None
    if padded and width is not None:
        # Row b reads its first target_lengths[b] entries; the rest is
        # padding, which may hold anything.
        def real() -> Tensor:
            positions = _lucid.arange(width, device=targets.device)
            sizes = _lucid.tensor(lengths, device=targets.device)
            return positions.reshape([1, width]) < sizes.reshape([batch, 1])

        counted = _host_read(real)
    labels, usable = _host_read(lambda: _guard(targets, classes, counted))
    _refuse(
        labels,
        usable,
        lambda v: (
            f"{op}: target {v} is out of range for {classes} classes "
            f"(valid range [0, {classes - 1}])"
        ),
    )
    return lengths
