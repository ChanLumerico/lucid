"""An integer index outside the axis it indexes is refused before any op
reads with it, and all such indices follow one rule (LCD-234).

Every ``nn.functional`` op that reads with a class or table index checks it
through ``lucid/nn/functional/_index_checks.py``. That covers a class-index
loss's targets, a ``one_hot`` class, an embedding row, and ``ctc_loss``'s
labels, ``blank`` and lengths. The ops used to follow three different
policies:

- the class losses checked after their gather;
- the embedding table was read on the host on every device;
- ``one_hot`` and ``ctc_loss`` did not check at all.

As a result, ``one_hot([0, 7], 3)`` gave a zero row for the 7, and
``ctc_loss`` with an ``input_lengths`` of 50 against 5 frames read past
the input and returned -0.379.

The rule this file holds every consumer to, in the table ``ROWS``:

- **CPU**: ``IndexError`` before the consumer's kernel runs. A refusal
  dispatches only the guard's reads of the index (``GUARD_READS``) and
  waits on no device.
- **Metal**: the index is not read back. The class losses' sample is NaN
  (``POISON``) and ``one_hot`` gives a zero row (``ZERO_ROW``), with 0
  host syncs. ``ctc_loss`` reads on the host anyway (``READ``), because
  its kernel round-trips through the CPU. So, for now, do ``embedding`` and
  ``embedding_bag``: until the engine isolates the table gather (LCD-228),
  an index far past the table would read past it.

Under ``-m parity`` the reference is checked to refuse the same rows,
except for the two places where Lucid is deliberately stricter (marked
``stricter``).
"""

import ast
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import lucid
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid.nn.functional import _index_checks
from lucid.nn.functional.sparse import check_embedding_indices
from lucid.test._fixtures.devices import metal_available

_needs_metal = pytest.mark.skipif(not metal_available(), reason="needs Metal")

N, C, T = 2, 4, 5

#: The only ops a refusal may dispatch, all of them reads of the index
#: itself:
#:
#: - the clamp and the comparison against it, and ``all`` over the result;
#: - the masks of the entries a consumer does not read as an index (an
#:   ``ignore_index`` sentinel, the pad after a label list, the padding of
#:   a ``ctc_loss`` row);
#: - the views and casts that shape them.
#:
#: Anything else is the consumer's own work, and must not have started.
GUARD_READS = frozenset(
    {
        "all",
        "arange",
        "astype",
        "bitwise_or",
        "broadcast_to",
        "clip",
        "cumprod",
        "equal",
        "full",
        "greater_equal",
        "invert",
        "less",
        "not_equal",
        "reshape",
        "unsqueeze",
    }
)

#: Metal policies.
POISON = "poison"  # the bad sample's loss is NaN, nothing read back
ZERO_ROW = "zero-row"  # one_hot's row is all zeros, nothing read back
READ = "read"  # read on the host and refused, as on the CPU

Kwargs = dict[str, object]


def _scores(device: str) -> lucid.Tensor:
    return lucid.tensor([[0.1, 0.2, 0.4, 0.8], [0.9, -0.3, 0.5, 0.0]], device=device)


def _log_probs(device: str) -> lucid.Tensor:
    lucid.manual_seed(0)
    return F.log_softmax(lucid.randn(T, N, C), dim=2).to(device)


def _table(device: str) -> lucid.Tensor:
    return lucid.arange(12, dtype=lucid.float32).reshape(C, 3).to(device)


def _ids(values: object, device: str) -> lucid.Tensor:
    return lucid.tensor(values, dtype=lucid.int64, device=device)


@dataclass(frozen=True)
class Row:
    """One out-of-range index given to one consumer."""

    id: str
    consumer: str
    call: Callable[[str], Kwargs]
    match: str
    metal: str
    #: The consumer's own kernels; none of them may run in a refusal.
    kernels: frozenset[str]
    #: For POISON / ZERO_ROW: the sample or row the bad index is in.
    bad: int = 1
    #: Lucid refuses what the reference computes; documented in the owner.
    stricter: bool = False
    #: Extra kwargs that only the per-sample (Metal) check passes.
    per_sample: Kwargs = field(default_factory=dict)


def _class_rows(name: str, kernels: frozenset[str]) -> list[Row]:
    rows = []
    for bad in (C, 9, -1):
        rows.append(
            Row(
                id=f"{name}-target{bad}",
                consumer=name,
                call=lambda d, b=bad: {"x": _scores(d), "target": _ids([1, b], d)},
                match=f"{name}: Target {bad} is out of bounds",
                metal=POISON,
                kernels=kernels,
                per_sample={"reduction": "none"},
            )
        )
    return rows


ROWS: list[Row] = [
    *_class_rows("cross_entropy", frozenset({"log_softmax", "gather"})),
    *_class_rows("nll_loss", frozenset({"gather"})),
    *_class_rows("multi_margin_loss", frozenset({"gather", "relu"})),
    Row(
        id="cross_entropy-kd-target",
        consumer="cross_entropy",
        call=lambda d: {
            "x": _scores(d).reshape(N, C, 1),
            "target": _ids([[1], [7]], d),
        },
        match="cross_entropy: Target 7 is out of bounds",
        metal=POISON,
        kernels=frozenset({"log_softmax", "gather"}),
        per_sample={"reduction": "none"},
    ),
    Row(
        id="multilabel_margin_loss-listed",
        consumer="multilabel_margin_loss",
        call=lambda d: {
            "x": _scores(d),
            "target": _ids([[1, 0, -1, 0], [2, 9, -1, 7]], d),
        },
        match="multilabel_margin_loss: Target 9 is out of bounds",
        metal=POISON,
        kernels=frozenset({"scatter_add", "relu"}),
        per_sample={"reduction": "none"},
    ),
    *[
        Row(
            id=f"one_hot-class{bad}",
            consumer="one_hot",
            call=lambda d, b=bad: {"tensor": _ids([0, b], d), "num_classes": 3},
            match=f"one_hot: class {bad} is out of range for num_classes=3",
            metal=ZERO_ROW,
            kernels=frozenset({"one_hot"}),
        )
        for bad in (3, 7, -1)
    ],
    *[
        Row(
            id=f"embedding-index{bad}",
            consumer="embedding",
            call=lambda d, b=bad: {"x": _ids([[0, b]], d), "weight": _table(d)},
            match=f"embedding: index {bad} is out of range for a table with 4",
            metal=READ,
            kernels=frozenset({"embedding"}),
        )
        for bad in (C, -1, 10**6)
    ],
    *[
        Row(
            id=f"embedding_bag-{form}-index{bad}",
            consumer="embedding_bag",
            call=(
                (lambda d, b=bad: {"x": _ids([[0, b]], d), "weight": _table(d)})
                if form == "rows"
                else (
                    lambda d, b=bad: {
                        "x": _ids([0, b, 1], d),
                        "weight": _table(d),
                        "offsets": _ids([0, 2], d),
                    }
                )
            ),
            match=f"embedding_bag: index {bad} is out of range for a table with 4",
            metal=READ,
            kernels=frozenset({"embedding_bag", "embedding"}),
        )
        for bad in (C, -1)
        for form in ("rows", "offsets")
    ],
    # ctc_loss: (T, N, C) = (5, 2, 4).
    *[
        Row(
            id=f"ctc_loss-{form}-target{bad}",
            consumer="ctc_loss",
            call=(
                (
                    lambda d, b=bad: {
                        "log_probs": _log_probs(d),
                        "targets": _ids([[1, b], [2, 3]], d),
                        "input_lengths": [T, T],
                        "target_lengths": [2, 2],
                    }
                )
                if form == "padded"
                else (
                    lambda d, b=bad: {
                        "log_probs": _log_probs(d),
                        "targets": _ids([1, b, 2, 3], d),
                        "input_lengths": [T, T],
                        "target_lengths": [2, 2],
                    }
                )
            ),
            match=f"ctc_loss: target {bad} is out of range for 4 classes",
            metal=READ,
            kernels=frozenset({"gather", "maximum", "div"}),
            stricter=True,
        )
        for bad in (C, 9, -1)
        for form in ("padded", "concatenated")
    ],
    *[
        Row(
            id=f"ctc_loss-blank{bad}",
            consumer="ctc_loss",
            call=lambda d, b=bad: {
                "log_probs": _log_probs(d),
                "targets": _ids([[1, 2], [2, 3]], d),
                "input_lengths": [T, T],
                "target_lengths": [2, 2],
                "blank": b,
            },
            match=f"ctc_loss: blank {bad} is out of range for 4 classes",
            metal=READ,
            kernels=frozenset({"gather", "maximum", "div"}),
        )
        for bad in (C, 10, -1)
    ],
    *[
        Row(
            id=f"ctc_loss-input_lengths{bad}",
            consumer="ctc_loss",
            call=lambda d, b=bad: {
                "log_probs": _log_probs(d),
                "targets": _ids([[1, 2], [2, 3]], d),
                "input_lengths": [b, T],
                "target_lengths": [2, 2],
            },
            match=rf"ctc_loss: each of input_lengths must lie in \[0, {T}\], got {bad}",
            metal=READ,
            kernels=frozenset({"gather", "maximum", "div"}),
        )
        for bad in (T + 1, 50, -1)
    ],
    *[
        Row(
            id=f"ctc_loss-target_lengths{bad}",
            consumer="ctc_loss",
            call=lambda d, b=bad: {
                "log_probs": _log_probs(d),
                "targets": _ids([[1, 2], [2, 3]], d),
                "input_lengths": [T, T],
                "target_lengths": [b, 2],
            },
            match=rf"ctc_loss: each of target_lengths must lie in \[0, 2\], got {bad}",
            metal=READ,
            kernels=frozenset({"gather", "maximum", "div"}),
        )
        for bad in (3, -1)
    ],
    Row(
        id="ctc_loss-concatenated-target_lengths-1",
        consumer="ctc_loss",
        call=lambda d: {
            "log_probs": _log_probs(d),
            "targets": _ids([1, 2, 2, 3], d),
            "input_lengths": [T, T],
            "target_lengths": [-1, 5],
        },
        match="ctc_loss: each of target_lengths must be non-negative, got -1",
        metal=READ,
        kernels=frozenset({"gather", "maximum", "div"}),
    ),
]

#: Every op that reads with a class or table index, and so has a row.
CONSUMERS = frozenset(row.consumer for row in ROWS)


def _ids_of(rows: list[Row]) -> list[str]:
    return [row.id for row in rows]


def test_every_row_has_a_unique_id() -> None:
    assert len(set(_ids_of(ROWS))) == len(ROWS)


# ── CPU: refused before the consumer's kernel ──────────────────────────────


@pytest.mark.parametrize("row", ROWS, ids=_ids_of(ROWS))
def test_the_cpu_refuses_before_any_kernel(row: Row) -> None:
    kwargs = row.call("cpu")
    fn = getattr(F, row.consumer)
    syncs = _C_engine.host_sync_count()
    with lucid.profiler.profile() as prof:
        with pytest.raises(IndexError, match=row.match):
            fn(**kwargs)
    dispatched = {event.name for event in prof.events()}
    assert dispatched <= GUARD_READS, f"not a guard read: {dispatched - GUARD_READS}"
    assert not dispatched & row.kernels
    assert _C_engine.host_sync_count() == syncs


@pytest.mark.parametrize("consumer", ["cross_entropy", "nll_loss", "multi_margin_loss"])
def test_a_cpu_int64_target_past_int32_is_refused_not_wrapped(consumer: str) -> None:
    # 2**40 cast to int32 is 0, and used to be scored as class 0.  A CPU
    # target is checked at its own width.  (Metal checks at the int32 its
    # gather reads with; the owner's docstring says why.)
    x, target = _scores("cpu"), _ids([1, 2**40], "cpu")
    with lucid.profiler.profile() as prof:
        with pytest.raises(IndexError, match=f"Target {2**40} is out of bounds"):
            getattr(F, consumer)(x, target)
    assert {event.name for event in prof.events()} <= GUARD_READS


# ── Metal: one documented policy ───────────────────────────────────────────


def _flat(values: object) -> list[float]:
    if isinstance(values, list):
        return [v for item in values for v in _flat(item)]
    return [float(values)]  # type: ignore[arg-type]


@_needs_metal
@pytest.mark.parametrize("row", ROWS, ids=_ids_of(ROWS))
def test_metal_follows_the_documented_policy(row: Row) -> None:
    fn = getattr(F, row.consumer)
    if row.metal == READ:
        with pytest.raises(IndexError, match=row.match):
            fn(**row.call("metal"))
        return
    kwargs, again = row.call("metal"), row.call("metal")
    syncs = _C_engine.host_sync_count()
    out = fn(**kwargs, **row.per_sample)
    assert _C_engine.host_sync_count() == syncs, "read back on Metal"
    if row.metal == POISON:
        values = _flat(out.tolist())
        assert math.isnan(values[row.bad])
        assert all(math.isfinite(v) for i, v in enumerate(values) if i != row.bad)
        syncs = _C_engine.host_sync_count()
        mean = fn(**again)
        assert _C_engine.host_sync_count() == syncs, "read back on Metal"
        assert math.isnan(float(mean.item()))
    else:
        rows = out.tolist()
        assert isinstance(rows, list)
        assert rows[row.bad] == [0, 0, 0]
        assert rows[0] == [1, 0, 0]


# ── what is not refused ────────────────────────────────────────────────────


def test_the_ignore_index_and_the_pad_are_not_indices(device: str) -> None:
    x = _scores(device)
    for fn in (F.cross_entropy, F.nll_loss):
        assert math.isfinite(float(fn(x, _ids([1, -100], device)).item()))
        assert math.isfinite(float(fn(x, _ids([1, 7], device), ignore_index=7).item()))
    # After a label list's first negative entry, nothing is read.
    t = _ids([[1, -1, 9, -7], [2, 3, -1, 99]], device)
    assert math.isfinite(float(F.multilabel_margin_loss(x, t).item()))
    # Past a ctc row's target length, nothing is read either.
    out = F.ctc_loss(
        _log_probs(device), _ids([[1, 99], [2, 3]], device), [T, T], [1, 2]
    )
    assert math.isfinite(float(out.item()))


def test_the_ends_of_every_axis_are_taken(device: str) -> None:
    x = _scores(device)
    for fn in (F.cross_entropy, F.nll_loss, F.multi_margin_loss):
        assert math.isfinite(float(fn(x, _ids([0, C - 1], device)).item()))
    assert F.one_hot(_ids([0, 2], device), 3).tolist() == [[1, 0, 0], [0, 0, 1]]
    assert F.embedding(_ids([0, C - 1], device), _table(device)).shape == (2, 3)
    assert F.embedding_bag(_ids([[0, C - 1]], device), _table(device)).shape == (
        1,
        3,
    )
    full = F.ctc_loss(
        _log_probs(device),
        _ids([[1, C - 1], [2, 3]], device),
        [T, T],
        [2, 2],
        blank=0,
    )
    assert math.isfinite(float(full.item()))
    last_blank = F.ctc_loss(
        _log_probs(device), _ids([[1, 2], [2, 0]], device), [T, 3], [2, 2], blank=3
    )
    assert math.isfinite(float(last_blank.item()))


def test_an_empty_index_has_nothing_to_check(device: str) -> None:
    empty = lucid.zeros(0, dtype=lucid.int64, device=device)
    assert F.one_hot(empty, 3).shape == (0, 3)
    assert F.embedding(empty, _table(device)).shape == (0, 3)


def test_check_embedding_indices_is_the_same_rule() -> None:
    # Kept under its released name; it is the owner's table check.
    table = _table("cpu")
    check_embedding_indices(_ids([0, 3], "cpu"), table, "lookup")
    with pytest.raises(IndexError, match="lookup: index 4 is out of range"):
        check_embedding_indices(_ids([4], "cpu"), table, "lookup")
    with pytest.raises(TypeError, match="lookup: indices must be an integer"):
        check_embedding_indices(lucid.tensor([0.5]), table, "lookup")


# ── ctc_loss: shapes the kernel would read past ────────────────────────────


@pytest.mark.parametrize(
    ("targets", "input_lengths", "target_lengths", "match"),
    [
        ([[1, 2], [2, 3]], [T], [2, 2], "input_lengths must hold one length"),
        ([[1, 2], [2, 3]], [T, T], [2], "target_lengths must hold one length"),
        ([[1, 2]], [T, T], [2, 2], "padded targets must hold one row"),
        ([1, 2, 2, 3], [T, T], [2, 3], "concatenated targets hold 4 labels"),
        ([[[1, 2]], [[2, 3]]], [T, T], [2, 2], "targets must be padded"),
    ],
    ids=["input-count", "target-count", "rows", "concat-sum", "rank-3"],
)
def test_ctc_refuses_a_shape_it_would_read_past(
    device: str,
    targets: object,
    input_lengths: list[int],
    target_lengths: list[int],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        F.ctc_loss(
            _log_probs(device), _ids(targets, device), input_lengths, target_lengths
        )


def test_ctc_refuses_an_empty_input_sequence_for_now(device: str) -> None:
    # Temporary (LCD-257): the reference takes it, but the kernel wrote
    # outside its buffer for one and took the process down.
    with pytest.raises(NotImplementedError, match="LCD-257"):
        F.ctc_loss(_log_probs(device), _ids([[1, 2], [2, 3]], device), [0, T], [0, 2])


# ── one owner ──────────────────────────────────────────────────────────────


_FUNCTIONAL = Path(F.__file__).parent
_OWNER = Path(_index_checks.__file__).name


def _calls_the_owner() -> set[str]:
    """Every public ``nn.functional`` function that calls into the owner."""
    owner_names = {
        name for name in vars(_index_checks) if name.startswith("_check")
    } | {"_class_targets", "check_embedding_indices"}
    found: set[str] = set()
    for path in sorted(_FUNCTIONAL.glob("*.py")):
        if path.name == _OWNER:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                continue
            for call in ast.walk(node):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id in owner_names
                ):
                    found.add(node.name)
    return found


def test_every_consumer_calls_the_owner_and_has_rows() -> None:
    # A new op that reads with an index and calls the owner gets rows in
    # the table; one of the table's ops that stopped calling it is caught.
    assert _calls_the_owner() - {"check_embedding_indices"} == set(CONSUMERS)


def test_only_the_owner_raises_index_error_in_nn_functional() -> None:
    # One rule, one place: an IndexError raised by an op of its own is
    # another policy starting.
    for path in sorted(_FUNCTIONAL.glob("*.py")):
        if path.name == _OWNER:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Raise) and node.exc is not None:
                exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
                assert not (
                    isinstance(exc, ast.Name) and exc.id == "IndexError"
                ), f"{path.name}:{node.lineno} raises IndexError itself"


# ── the reference ──────────────────────────────────────────────────────────


def _ref_kwargs(R: object, row: Row) -> Kwargs:
    out: Kwargs = {}
    for key, value in row.call("cpu").items():
        if isinstance(value, lucid.Tensor):
            out[key] = R.tensor(value.tolist())  # type: ignore[attr-defined]
        else:
            out[key] = value
    return out


@pytest.mark.parity
@pytest.mark.parametrize("row", ROWS, ids=_ids_of(ROWS))
def test_the_reference_refuses_the_same_rows(ref: object, row: Row) -> None:
    R = ref
    kwargs = _ref_kwargs(R, row)
    args: list[object] = []
    if row.consumer in ("embedding", "embedding_bag"):
        args = [kwargs.pop("x"), kwargs.pop("weight")]
    elif "x" in kwargs:
        args = [kwargs.pop("x")]
    elif "tensor" in kwargs:
        args = [kwargs.pop("tensor")]
    fn = getattr(R.nn.functional, row.consumer)  # type: ignore[attr-defined]
    if row.stricter:
        # The reference reads past the class axis and answers; Lucid
        # refuses, as the owner documents.
        fn(*args, **kwargs)
        return
    with pytest.raises(Exception):  # noqa: B017 - its types vary by op
        fn(*args, **kwargs)


@pytest.mark.parity
@pytest.mark.parametrize("blank", [0, C - 1])
def test_ctc_at_the_ends_of_its_axes_matches_the_reference(
    ref: object, blank: int
) -> None:
    R = ref
    lp = _log_probs("cpu")
    targets = [[1, 2], [2, 1]] if blank == 0 else [[0, 2], [2, 1]]
    for il, tl in (([T, T], [2, 2]), ([T, 2], [2, 0]), ([3, T], [2, 1])):
        got = F.ctc_loss(
            lp, _ids(targets, "cpu"), il, tl, blank=blank, reduction="none"
        )
        want = R.nn.functional.ctc_loss(  # type: ignore[attr-defined]
            R.tensor(lp.tolist()),  # type: ignore[attr-defined]
            R.tensor(targets),  # type: ignore[attr-defined]
            il,
            tl,
            blank=blank,
            reduction="none",
        )
        for g, w in zip(got.tolist(), want.tolist()):  # type: ignore[union-attr]
            assert math.isclose(g, w, rel_tol=1e-4, abs_tol=1e-5) or (
                math.isinf(g) and math.isinf(w)
            )


@pytest.mark.parity
@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason="LCD-257: the ctc kernel cannot run an empty input sequence yet",
)
def test_ctc_takes_an_empty_input_sequence_as_the_reference_does(ref: object) -> None:
    R = ref
    lp = _log_probs("cpu")
    for zero_infinity in (False, True):
        got = F.ctc_loss(
            lp,
            _ids([[1, 2], [2, 3]], "cpu"),
            [0, 0],
            [0, 1],
            reduction="none",
            zero_infinity=zero_infinity,
        )
        want = R.nn.functional.ctc_loss(  # type: ignore[attr-defined]
            R.tensor(lp.tolist()),  # type: ignore[attr-defined]
            R.tensor([[1, 2], [2, 3]]),  # type: ignore[attr-defined]
            [0, 0],
            [0, 1],
            reduction="none",
            zero_infinity=zero_infinity,
        )
        assert got.tolist() == want.tolist()
