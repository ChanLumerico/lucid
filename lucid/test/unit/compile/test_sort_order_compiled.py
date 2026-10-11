"""Compiled ordering ops name the positions eager names (LCD-425).

Eager orders stably — equal values in index order, so ``topk`` names ties
lowest index first (LCD-407) — treats NaN as the largest value (and, for
``argmin``, the smallest), calls -0.0 and +0.0 equal and compares int64
exactly.  MPSGraph's ``topK`` did none of that: ``topk([3, 3, 3, 3], 2)``
named 3 and 1, NaN came back as 0.0, and 2^40 + 3 lost to 2^40.  The
emitters now order through ``OpEmitters/_SortOrder.h``; this matrix — ties,
NaN, signed zeros, infinities, wide int64, every axis, long axes — holds
``topk`` / ``sort`` / ``argsort`` / ``argmax`` / ``argmin`` to the eager
answer exactly.
"""

import pytest

import lucid
from lucid._C import engine as _C_engine

from lucid.test.unit.compile._helpers import COMPILE_DEVICE

NAN, INF = float("nan"), float("inf")
ROWS = [
    [0.0, 5.0, 4.0, 4.0, 1.0, 4.0],
    [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
    [1.0, NAN, 2.0, NAN, -INF, INF],
    [-0.0, 0.0, -1.0, 0.0, -0.0, 7.0],
    [2.0, 2.0, -2.0, -2.0, 2.0, -2.0],
]
FLOATS = {"f32": lucid.float32, "f16": lucid.float16, "bf16": lucid.bfloat16}
INTS = {"i32": lucid.int32, "i64": lucid.int64}


def _rows(dtype: lucid.dtype) -> lucid.Tensor:
    if dtype in INTS.values():
        src = [[0.0 if v != v or abs(v) == INF else v for v in r] for r in ROWS]
        return lucid.tensor(src).to(dtype=dtype).to(COMPILE_DEVICE)
    return lucid.tensor(ROWS).to(dtype=dtype).to(COMPILE_DEVICE)


def _same(fn: object, x: lucid.Tensor) -> None:
    _C_engine.compile.session_cache_clear()
    compiled = lucid.compile(fn)  # type: ignore[arg-type]
    got = compiled(x)
    info = compiled.cache_info()  # type: ignore[attr-defined]
    assert info["entries"] == 1 and not info["eager_only"], info
    want = fn(x)  # type: ignore[operator]
    for g, w in zip(
        got if isinstance(got, tuple) else (got,),
        want if isinstance(want, tuple) else (want,),
        strict=True,
    ):
        g_cpu, w_cpu = g.to("cpu"), w.to("cpu")
        if g_cpu.is_floating_point():
            assert (g_cpu.isnan() == w_cpu.isnan()).all().item()
            g_cpu = lucid.where(g_cpu.isnan(), lucid.zeros_like(g_cpu), g_cpu)
            w_cpu = lucid.where(w_cpu.isnan(), lucid.zeros_like(w_cpu), w_cpu)
            # -0.0 keeps its sign: compare the bits' sign as well.
            assert ((1.0 / g_cpu.float()) == (1.0 / w_cpu.float())).all().item()
        assert g_cpu.tolist() == w_cpu.tolist()


DTYPES = {**FLOATS, **INTS}


@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("k", [1, 2, 3, 5])
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_topk(dtype: str, k: int, dim: int) -> None:
    x = _rows(DTYPES[dtype])
    _same(lambda t: tuple(o * 1 for o in lucid.topk(t, k, dim=dim)), x)


@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_sort_and_argsort(dtype: str, dim: int) -> None:
    x = _rows(DTYPES[dtype])
    _same(lambda t: lucid.sort(t, dim=dim) * 1, x)
    _same(lambda t: lucid.argsort(t, dim=dim) * 1, x)
    _same(lambda t: lucid.argsort(t, dim=dim, descending=True) * 1, x)


@pytest.mark.parametrize("op", ["argmax", "argmin"])
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_argmax_and_argmin(dtype: str, dim: int, op: str) -> None:
    fn = getattr(lucid, op)
    _same(lambda t: fn(t, dim=dim) * 1, _rows(DTYPES[dtype]))


@pytest.mark.parametrize("dtype", list(FLOATS))
def test_compiling_the_order_prints_no_mps_warning(
    dtype: str, capfd: pytest.CaptureFixture[str]
) -> None:
    """The ±0 key once ran a float16 ``equal`` that MPSGraph's
    compare-to-zero pass rejected with a warning on every compile."""
    x = _rows(FLOATS[dtype])
    capfd.readouterr()
    _same(lambda t: lucid.argsort(t) * 1, x)
    _same(lambda t: lucid.sort(t) * 1, x)
    _same(lambda t: tuple(o * 1 for o in lucid.topk(t, 3)), x)
    out, err = capfd.readouterr()
    assert "mps_equal" not in out + err
    assert "ConvertBinaryCompareToZero" not in out + err


def _wide() -> lucid.Tensor:
    base = 1 << 40
    vals = [base + 1, base + 3, base + 2, -base, 1 << 33, base + 3 + (1 << 34), 5, base]
    return lucid.tensor(vals, dtype=lucid.int64).to(COMPILE_DEVICE)


def test_wide_int64_ranks_exactly() -> None:
    _same(lambda t: lucid.topk(t, 8)[1] * 1, _wide())
    _same(lambda t: lucid.argsort(t, descending=True) * 1, _wide())


def test_wide_int64_values_come_back_whole() -> None:
    """The values come back through ``_DataMovement.h`` (LCD-435)."""
    _same(lambda t: lucid.topk(t, 8)[0] * 1, _wide())
    _same(lambda t: lucid.sort(t, descending=True) * 1, _wide())


@pytest.mark.parametrize("n", [5387, 1 << 17])
def test_ties_on_a_long_axis_stay_in_index_order(n: int) -> None:
    """Few distinct values, many ties: the order the sort keeps for them."""
    x = (lucid.arange(n) * 7919 % 5).to(lucid.float32).reshape(1, n).to(COMPILE_DEVICE)
    _same(lambda t: tuple(o * 1 for o in lucid.topk(t, 64)), x)
    _same(lambda t: lucid.argsort(t) * 1, x)
