"""The index and statistics adapters answer as the reference does (LCD-336).

Each one used to re-decide a rule its engine op or the reference already
fixes: ``take`` took a 1-D index only, ``masked_fill`` passed an
unbroadcast mask to a kernel that walks both buffers side by side,
``where(cond)`` and ``nonzero(as_tuple=True)`` had no spelling at all,
``histc`` counted NaN in the first bin and widened a one-point range to
``[v, v + 1]``, ``unique`` took none of its options, and ``scatter`` /
``scatter_add`` narrowed an ``int64`` index to ``int32`` and cast a
``src`` of another dtype.  Every case runs on both devices against the
reference.
"""

from types import ModuleType
from typing import Protocol

import pytest

import lucid
from lucid._C import engine as _C_engine

NAN = float("nan")


class _Array(Protocol):
    """What the comparison reads off a reference tensor."""

    shape: tuple[int, ...]
    dtype: object

    def reshape(self, *shape: int) -> _Array: ...

    def tolist(self) -> object: ...


def _same(got: object, want: _Array | tuple[_Array, ...]) -> None:
    """``got`` (Lucid) holds ``want``'s (reference) values, dtype, shape and nesting."""
    if isinstance(want, tuple):
        assert isinstance(got, tuple) and len(got) == len(want)
        for g, w in zip(got, want, strict=True):
            _same(g, w)
        return
    assert isinstance(got, lucid.Tensor)
    assert tuple(got.shape) == tuple(want.shape)
    assert str(got.dtype).rsplit(".")[-1] == str(want.dtype).rsplit(".")[-1]
    flat = want.reshape(-1).tolist()
    assert got.reshape(-1).tolist() == pytest.approx(flat, nan_ok=True)


@pytest.mark.parametrize(
    "index", [[[1, 2], [3, -1]], 4, [], [[[0]], [[5]]], [0, -6]], ids=str
)
def test_take_answers_in_the_index_shape(
    ref: ModuleType, device: str, index: object
) -> None:
    data = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    got = lucid.take(
        lucid.tensor(data, device=device),
        lucid.tensor(index, dtype=lucid.int64, device=device),
    )
    _same(got, ref.take(ref.tensor(data), ref.tensor(index, dtype=ref.int64)))


@pytest.mark.parametrize(
    "mask_shape", [(2, 1, 1, 4), (), (4,), (3, 1, 4), (2, 3, 4, 4)], ids=str
)
def test_masked_fill_broadcasts_the_mask(
    ref: ModuleType, device: str, mask_shape: tuple[int, ...]
) -> None:
    x = [float(i) for i in range(2 * 3 * 4 * 4)]
    n = 1
    for s in mask_shape:
        n *= s
    bits = [i % 3 == 0 for i in range(n)]
    want = (
        ref.tensor(x)
        .reshape(2, 3, 4, 4)
        .masked_fill(ref.tensor(bits).reshape(mask_shape), -1.0)
    )
    got = lucid.masked_fill(
        lucid.tensor(x, device=device).reshape(2, 3, 4, 4),
        lucid.tensor(bits, device=device).reshape(mask_shape),
        -1.0,
    )
    _same(got, want)


def test_masked_fill_broadcasts_the_input_too(ref: ModuleType, device: str) -> None:
    want = ref.zeros(4).masked_fill(ref.ones(2, 4, dtype=ref.bool), 1.0)
    got = lucid.zeros(4, device=device).masked_fill(
        lucid.ones(2, 4, dtype=lucid.bool, device=device), 1.0
    )
    _same(got, want)


def test_masked_fill_refuses_a_mask_that_does_not_broadcast(device: str) -> None:
    with pytest.raises(ValueError, match="do not broadcast"):
        lucid.masked_fill(
            lucid.zeros(4, device=device),
            lucid.ones(3, dtype=lucid.bool, device=device),
            1.0,
        )


@pytest.mark.parametrize(
    "cond", [[[1, 0], [0, 1]], [0, 0, 0], [[True]], 3.0, 0.0], ids=str
)
def test_where_alone_and_nonzero_as_tuple(
    ref: ModuleType, device: str, cond: object
) -> None:
    c = lucid.tensor(cond, device=device)
    want = ref.nonzero(ref.tensor(cond), as_tuple=True)
    _same(lucid.where(c), ref.where(ref.tensor(cond)))
    _same(lucid.nonzero(c, as_tuple=True), want)
    _same(c.nonzero(as_tuple=True), want)
    _same(lucid.nonzero(c), ref.nonzero(ref.tensor(cond)))


def test_where_with_one_branch_is_refused(device: str) -> None:
    c = lucid.tensor([True, False], device=device)
    with pytest.raises(TypeError, match="cond alone"):
        lucid.where(c, lucid.ones(2, device=device))


@pytest.mark.parametrize(
    ("data", "bins", "lo", "hi"),
    [
        ([NAN, 1.0, 2.0], 2, 0.0, 2.0),
        ([3.0, 3.0], 4, 0.0, 0.0),
        ([3.0], 3, 5.0, 5.0),
        ([1.0, 3.0], 4, 2.0, 2.0),
        ([float("inf"), 1.0, 2.0], 2, 0.0, 2.0),
        ([-float("inf"), NAN, 0.5], 2, 0.0, 1.0),
        ([5.0, -1.0, 1.0], 2, 0.0, 2.0),
        ([], 3, 0.0, 0.0),
        ([0.25, 0.5, 0.75, 1.0], 4, 0.0, 0.0),
    ],
    ids=str,
)
def test_histc(
    ref: ModuleType, device: str, data: list[float], bins: int, lo: float, hi: float
) -> None:
    got = lucid.histc(lucid.tensor(data, device=device), bins, lo, hi)
    _same(got, ref.histc(ref.tensor(data), bins, lo, hi))


@pytest.mark.parametrize(
    ("data", "lo", "hi"),
    [
        ([NAN, 1.0], 0.0, 0.0),
        ([float("inf"), 1.0], 0.0, 0.0),
        ([1.0], 0.0, float("inf")),
    ],
    ids=str,
)
def test_histc_refuses_a_range_that_is_not_finite(
    device: str, data: list[float], lo: float, hi: float
) -> None:
    with pytest.raises(ValueError, match="not finite"):
        lucid.histc(lucid.tensor(data, device=device), 2, lo, hi)


def test_histc_refuses_min_above_max(device: str) -> None:
    with pytest.raises(ValueError, match="at least min"):
        lucid.histc(lucid.tensor([1.0], device=device), 2, 3.0, 1.0)


_UNIQUE_DATA = {
    "ints": [3, 1, 3, 2, 1, 1, 3, 2, 2, 1, 0, 3, 3, 1],
    "matrix": [[1, 2], [1, 2], [0, 5], [1, 2], [0, 4]],
    "floats": [1.0, -0.0, 0.0, -1.0, 2.5, 1.0],
    "scalar": 5,
    "empty": [],
}


@pytest.mark.parametrize("name", list(_UNIQUE_DATA))
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("counts", [False, True])
def test_unique_flat(
    ref: ModuleType, device: str, name: str, inverse: bool, counts: bool
) -> None:
    data = _UNIQUE_DATA[name]
    got = lucid.unique(
        lucid.tensor(data, device=device),
        return_inverse=inverse,
        return_counts=counts,
    )
    want = ref.unique(ref.tensor(data), return_inverse=inverse, return_counts=counts)
    _same(got, want)


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_unique_along_a_dim(ref: ModuleType, device: str, dim: int) -> None:
    data = [[[1, 2], [1, 2]], [[0, 5], [0, 5]], [[1, 2], [1, 2]]]
    got = lucid.unique(
        lucid.tensor(data, device=device),
        dim=dim,
        return_inverse=True,
        return_counts=True,
    )
    want = ref.unique(
        ref.tensor(data), dim=dim, return_inverse=True, return_counts=True
    )
    _same(got, want)


@pytest.mark.parametrize(("shape", "dim"), [((0, 3), 0), ((2, 0, 3), 1)], ids=str)
def test_unique_along_an_empty_dim(
    ref: ModuleType, device: str, shape: tuple[int, ...], dim: int
) -> None:
    got = lucid.unique(
        lucid.zeros(*shape, device=device),
        dim=dim,
        return_inverse=True,
        return_counts=True,
    )
    want = ref.unique(
        ref.zeros(*shape), dim=dim, return_inverse=True, return_counts=True
    )
    _same(got, want)


def test_unique_sorted_false_still_answers(ref: ModuleType, device: str) -> None:
    got = lucid.unique(lucid.tensor([3, 1, 3], device=device), sorted=False)
    assert sorted(got.tolist()) == sorted(ref.unique(ref.tensor([3, 1, 3])).tolist())


@pytest.mark.parametrize("op", ["scatter", "scatter_add"])
def test_scatter_passes_the_index_at_full_width(device: str, op: str) -> None:
    base = lucid.zeros(4, device=device)
    index = lucid.tensor([2**40], dtype=lucid.int64, device=device)
    call = lambda: getattr(base, op)(0, index, lucid.ones(1, device=device))
    if device == "cpu":
        with pytest.raises(IndexError):
            call()
    else:
        # Metal reads no index back: one past the axis writes nothing, and
        # never the slot a narrowed 2**40 named (slot 0).
        assert call().tolist() == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("op", ["scatter", "scatter_add"])
def test_scatter_refuses_a_src_of_another_dtype(ref: ModuleType, op: str) -> None:
    index = [1, 3]
    with pytest.raises(RuntimeError):
        getattr(ref.zeros(4), op)(0, ref.tensor(index), ref.ones(2, dtype=ref.float64))
    with pytest.raises(_C_engine.DtypeMismatch):
        getattr(lucid.zeros(4), op)(
            0, lucid.tensor(index), lucid.ones(2, dtype=lucid.float64)
        )


# The reference warns that ``reduce=`` with a tensor src is on its way out.
@pytest.mark.filterwarnings("ignore:The reduce argument:UserWarning")
def test_scatter_reduce_modes_and_a_number(ref: ModuleType, device: str) -> None:
    index = [[0, 1, 0], [2, 2, 1]]
    src = [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]
    for reduce in ("add", "multiply"):
        want = ref.ones(3, 3).scatter(
            0, ref.tensor(index), ref.tensor(src), reduce=reduce
        )
        got = lucid.ones(3, 3, device=device).scatter(
            0,
            lucid.tensor(index, device=device),
            lucid.tensor(src, device=device),
            reduce=reduce,
        )
        _same(got, want)
    want = ref.zeros(3, 3).scatter(0, ref.tensor(index), 9.0)
    got = lucid.zeros(3, 3, device=device).scatter(
        0, lucid.tensor(index, device=device), 9.0
    )
    _same(got, want)
    with pytest.raises(ValueError, match="'add' or 'multiply'"):
        lucid.zeros(3, device=device).scatter(
            0,
            lucid.tensor([0], device=device),
            lucid.ones(1, device=device),
            reduce="max",
        )


_SCATTERS = {
    "scatter": lambda b, i, s: lucid.scatter(b, 0, i, s),
    "scatter_add": lambda b, i, s: lucid.scatter_add(b, 0, i, s),
    "reduce_add": lambda b, i, s: lucid.scatter(b, 0, i, s, reduce="add"),
    "reduce_multiply": lambda b, i, s: lucid.scatter(b, 0, i, s, reduce="multiply"),
    "scatter_reduce_amax": lambda b, i, s: lucid.scatter_reduce(b, 0, i, s, "amax"),
}


@pytest.mark.parametrize("index_dtype", ["int64", "int32"])
@pytest.mark.parametrize("name", list(_SCATTERS))
def test_an_index_written_after_forward_is_refused_in_backward(
    device: str, name: str, index_dtype: str
) -> None:
    # The index now reaches the node as the caller's own tensor — no
    # narrowing copy in between — so a write into it after forward would
    # move the gradient to other slots.  The node's version guard refuses
    # it instead (LCD-320, LCD-351).
    base = lucid.ones(4, device=device, requires_grad=True)
    src = lucid.tensor([10.0, 20.0], device=device, requires_grad=True)
    index = lucid.tensor([0, 1], dtype=getattr(lucid, index_dtype), device=device)
    out = _SCATTERS[name](base, index, src)
    index.add_(2)
    with pytest.raises(_C_engine.VersionMismatch):
        out.sum().backward()
