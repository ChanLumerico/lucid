"""NaN-skipping order statistics, one slice at a time (LCD-376).

``nanmedian`` along a dim used to pick one sorted position for every slice
from the *average* non-NaN count, so a slice with a different count got
another slice's median — silently.  It built that position on the CPU,
which a Metal input refused; it filled NaNs with ``inf``, which an integer
input refused; and it read the count back to the host.  Every slice now
reads its own count, a tensor on the input's device, and the median is
gathered from the input so the gradient lands on the entry ``indices``
names.

``nanquantile`` shared the inf fill and, over the whole tensor, the host
read; ``quantile`` filled an integer ``interpolation="lower"`` result with
NaN.  The cases below hold all three to the reference: values, indices
and gradients, on every dim with and without ``keepdim``, for data with no
NaN, some, an all-NaN slice and ties, in float32, float16 and int64, on
CPU and Metal.
"""

import math
from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

import lucid

pytestmark = pytest.mark.parity

_NAN = math.nan

# Distinct numbers, so the reference's indices are not a tie-break choice.
_DATA: dict[str, list[list[float]]] = {
    "no_nan": [[3, 7, 1, 9], [4, 0, 6, 2], [11, 5, 8, 10]],
    "some_nan": [[3, _NAN, 1, 9], [4, 0, _NAN, _NAN], [_NAN, 5, 8, 10]],
    # Row 0 and column 2 are all NaN; the other slices hold 1 to 3 numbers.
    "all_nan_slice": [
        [_NAN, _NAN, _NAN, _NAN],
        [4, _NAN, _NAN, 2],
        [_NAN, 5, _NAN, 10],
    ],
    # Both even and odd counts per slice, so the lower median is exercised.
    "uneven_counts": [[1, _NAN, _NAN, _NAN], [1, 2, 3, _NAN], [6, 4, 2, 0]],
}
_FLOAT_ONLY = {"some_nan", "all_nan_slice", "uneven_counts"}
_TIES = [[2, 1, 2, 2], [5, 5, 5, 5], [0, 3, 3, 1]]

_DTYPES = ("float32", "float16", "int64")
_DIMS = (None, 0, 1, -1)


def _cases() -> list[object]:
    return [
        pytest.param(name, dtype, id=f"{name}-{dtype}")
        for name in _DATA
        for dtype in _DTYPES
        if not (dtype == "int64" and name in _FLOAT_ONLY)
    ]


def _pair(
    ref: ModuleType, data: object, dtype: str, device: str, grad: bool
) -> tuple[lucid.Tensor, object]:
    x = lucid.tensor(data, dtype=getattr(lucid, dtype), device=device)
    want = ref.tensor(data, dtype=getattr(ref, dtype))
    if grad:
        x.requires_grad_(True)
        want.requires_grad_(True)
    return x, want


def _np(t: object) -> np.ndarray:
    return np.asarray(t.detach().cpu().float().numpy())


def _same(got: object, want: object) -> None:
    assert tuple(got.shape) == tuple(want.shape)
    np.testing.assert_array_equal(_np(got), _np(want))


@pytest.mark.parametrize("keepdim", [False, True], ids=["drop", "keep"])
@pytest.mark.parametrize("dim", _DIMS, ids=lambda d: f"dim{d}")
@pytest.mark.parametrize("name,dtype", _cases())
def test_nanmedian_matches_reference(
    ref: ModuleType, device: str, name: str, dtype: str, dim: int | None, keepdim: bool
) -> None:
    grad = dtype != "int64"
    x, want_x = _pair(ref, _DATA[name], dtype, device, grad)
    if dim is None:
        got, want = lucid.nanmedian(x), ref.nanmedian(want_x)
        _same(got, want)
        outs = (got, want)
    else:
        result = lucid.nanmedian(x, dim, keepdim)
        assert isinstance(result, tuple)
        values, indices = result
        want_values, want_indices = ref.nanmedian(want_x, dim, keepdim)
        _same(values, want_values)
        _same(indices, want_indices)
        assert indices.dtype == lucid.int64
        assert values.device == x.device and indices.device == x.device
        outs = (values, want_values)
    if grad:
        outs[0].sum().backward()
        outs[1].sum().backward()
        assert isinstance(x.grad, lucid.Tensor)
        np.testing.assert_allclose(_np(x.grad), _np(want_x.grad), rtol=1e-3)


@pytest.mark.parametrize("dtype", ("float32", "float16"))
def test_whole_tensor_median_shares_its_gradient_between_ties(
    ref: ModuleType, device: str, dtype: str
) -> None:
    for data in (_TIES, [[math.inf, math.inf, _NAN, 1.0]]):
        x, want_x = _pair(ref, data, dtype, device, True)
        got, want = lucid.nanmedian(x), ref.nanmedian(want_x)
        _same(got, want)
        got.backward()
        want.backward()
        np.testing.assert_allclose(_np(x.grad), _np(want_x.grad), rtol=1e-3)


@pytest.mark.parametrize("dim", (0, 1), ids=lambda d: f"dim{d}")
@pytest.mark.parametrize("dtype", _DTYPES)
def test_a_tied_median_names_an_entry_holding_it(
    ref: ModuleType, device: str, dtype: str, dim: int
) -> None:
    # Which of several equal entries the reference names is its own choice;
    # what holds is the value, and that ``indices`` points at it.
    x, want_x = _pair(ref, _TIES, dtype, device, False)
    values, indices = lucid.nanmedian(x, dim, True)
    _same(values, ref.nanmedian(want_x, dim, True).values)
    _same(lucid.take_along_dim(x, indices, dim), values)


@pytest.mark.parametrize("dtype", ("float32", "int64"))
def test_an_empty_input_matches_reference(
    ref: ModuleType, device: str, dtype: str
) -> None:
    x = lucid.zeros(0, 4, dtype=getattr(lucid, dtype), device=device)
    want_x = ref.zeros(0, 4, dtype=getattr(ref, dtype))
    for dim in (1, -1):
        values, indices = lucid.nanmedian(x, dim)
        want_values, want_indices = ref.nanmedian(want_x, dim)
        _same(values, want_values)
        _same(indices, want_indices)
    # The reference refuses a median along an axis of length 0.
    with pytest.raises(IndexError):
        ref.nanmedian(want_x, 0)
    with pytest.raises(IndexError):
        lucid.nanmedian(x, 0)
    if dtype == "float32":
        assert math.isnan(lucid.nanmedian(x).item())


def test_a_zero_dim_input_is_its_own_median(ref: ModuleType, device: str) -> None:
    x = lucid.tensor(5.0, device=device)
    for dim in (0, -1):
        for keepdim in (False, True):
            values, indices = lucid.nanmedian(x, dim, keepdim)
            want_values, want_indices = ref.nanmedian(ref.tensor(5.0), dim, keepdim)
            _same(values, want_values)
            _same(indices, want_indices)
    with pytest.raises(IndexError):
        lucid.nanmedian(x, 1)


def test_bool_and_out_of_range_dims_are_refused(device: str) -> None:
    with pytest.raises(NotImplementedError):
        lucid.nanmedian(lucid.tensor([True, False], device=device))
    with pytest.raises(IndexError):
        lucid.nanmedian(lucid.zeros(2, 3, device=device), 2)


# ── nanquantile / quantile ────────────────────────────────────────────────────


@pytest.mark.parametrize("dim", _DIMS, ids=lambda d: f"dim{d}")
@pytest.mark.parametrize("name", list(_DATA))
def test_nanquantile_matches_reference(
    ref: ModuleType, device: str, name: str, dim: int | None
) -> None:
    x, want_x = _pair(ref, _DATA[name], "float32", device, True)
    got = lucid.nanquantile(x, 0.4, dim)
    want = ref.nanquantile(want_x, 0.4, dim)
    assert tuple(got.shape) == tuple(want.shape)
    np.testing.assert_allclose(_np(got), _np(want), rtol=1e-6, equal_nan=True)
    got.sum().backward()
    want.sum().backward()
    np.testing.assert_allclose(_np(x.grad), _np(want_x.grad), rtol=1e-6)


@pytest.mark.parametrize("dim", _DIMS, ids=lambda d: f"dim{d}")
def test_integer_quantiles_need_no_nan_handling(
    ref: ModuleType, device: str, dim: int | None
) -> None:
    data = _DATA["no_nan"]
    x = lucid.tensor(data, dtype=lucid.int64, device=device)
    as_float = ref.tensor(data, dtype=ref.float32)
    # The reference refuses an integer quantile; Lucid answers what the same
    # numbers give in floating point, with or without NaN skipping.
    want = ref.quantile(as_float, 0.4, dim)
    for fn in (lucid.quantile, lucid.nanquantile):
        np.testing.assert_allclose(_np(fn(x, 0.4, dim)), _np(want), rtol=1e-6)
    lower = lucid.quantile(x, 0.4, dim, interpolation="lower")
    want_lower = ref.quantile(as_float, 0.4, dim, interpolation="lower")
    assert lower.dtype == lucid.int64
    np.testing.assert_array_equal(_np(lower), _np(want_lower))


# ── no host reads ─────────────────────────────────────────────────────────────

_CALLS: dict[str, Callable[[lucid.Tensor], object]] = {
    "nanmedian_dim": lambda x: lucid.nanmedian(x, 1),
    "nanmedian_all": lambda x: lucid.nanmedian(x),
    "nanquantile_dim": lambda x: lucid.nanquantile(x, 0.5, 1),
    "nanquantile_all": lambda x: lucid.nanquantile(x, 0.5),
    "quantile_all": lambda x: lucid.quantile(x, 0.5),
}


@pytest.mark.parametrize("call", list(_CALLS))
def test_no_value_is_read_back_to_the_host(
    monkeypatch: pytest.MonkeyPatch, device: str, call: str
) -> None:
    x = lucid.tensor(_DATA["all_nan_slice"], device=device).requires_grad_(True)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"{call} read a value back to the host")

    with monkeypatch.context() as guard:
        for method in ("numpy", "item", "tolist", "__float__", "__int__", "__bool__"):
            guard.setattr(lucid.Tensor, method, forbidden)
        _CALLS[call](x)
