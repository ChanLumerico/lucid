"""The overwrite scatters take every dtype (CHA-27).

The CPU overwrite kernel took f32 and f64 only, so ``index_copy`` /
``index_put`` / ``put`` / ``masked_scatter`` refused bool, the integers and
the halves on the CPU — and int64 and complex everywhere, Metal included
(MLX scatters no 8-byte element there, and its CPU fallback hit the same
refusal).  Now the kernel moves bytes by element width, every dtype.
"""

import numpy as np
import pytest

import lucid
from lucid.test._fixtures.devices import device_dtype_params

_ALL_DTYPES = (
    lucid.bool,
    lucid.int8,
    lucid.int16,
    lucid.int32,
    lucid.int64,
    lucid.float16,
    lucid.bfloat16,
    lucid.float32,
    lucid.float64,
    lucid.complex64,
    lucid.complex128,
)

_NP = {
    lucid.bool: np.bool_,
    lucid.int8: np.int8,
    lucid.int16: np.int16,
    lucid.int32: np.int32,
    lucid.int64: np.int64,
    lucid.float16: np.float16,
    lucid.bfloat16: np.float32,  # small integers are exact in bfloat16
    lucid.float32: np.float32,
    lucid.float64: np.float64,
    lucid.complex64: np.complex64,
    lucid.complex128: np.complex128,
}


def _t(values: object, dtype: lucid.dtype, device: str) -> lucid.Tensor:
    t = lucid.tensor(np.asarray(values, dtype=_NP[dtype]), device=device)
    return t.to(dtype) if t.dtype != dtype else t


def _i(values: object, device: str) -> lucid.Tensor:
    return lucid.tensor(np.asarray(values, dtype=np.int64), device=device)


def _same(got: lucid.Tensor, want: np.ndarray) -> None:
    assert got.tolist() == want.tolist()


_BASE = [1, 2, 3, 4, 5, 6]
_POS = [4, 0, 2]
_SRC = [0, 7, 1]


def _expected(dtype: lucid.dtype, pos: list[int] = _POS) -> np.ndarray:
    out = np.asarray(_BASE, dtype=_NP[dtype])
    out[pos] = np.asarray(_SRC, dtype=_NP[dtype])
    return out


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_ALL_DTYPES))
class TestTheOverwritesTakeEveryDtype:
    """``index_copy`` / ``index_put`` / ``put`` refused int, half and bool on
    the CPU (CHA-27), and int64 and complex64 on Metal too."""

    def test_index_copy(self, device: str, dtype: lucid.dtype) -> None:
        out = _t(_BASE, dtype, device).index_copy(0, _i(_POS, device), _t(_SRC, dtype, device))
        _same(out, _expected(dtype))

    def test_index_put(self, device: str, dtype: lucid.dtype) -> None:
        out = lucid.index_put(
            _t(_BASE, dtype, device), (_i(_POS, device),), _t(_SRC, dtype, device)
        )
        _same(out, _expected(dtype))

    def test_put(self, device: str, dtype: lucid.dtype) -> None:
        out = lucid.put(_t(_BASE, dtype, device), _i(_POS, device), _t(_SRC, dtype, device))
        _same(out, _expected(dtype))

    def test_masked_scatter(self, device: str, dtype: lucid.dtype) -> None:
        mask = lucid.tensor([True, False, True, False, True, False], device=device)
        out = lucid.masked_scatter(_t(_BASE, dtype, device), mask, _t(_SRC, dtype, device))
        _same(out, _expected(dtype, [0, 2, 4]))
