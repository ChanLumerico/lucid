"""Unit tests for dtype operations: astype, dtype promotion, dtypes."""

import pytest
import numpy as np
import lucid
from lucid.test._comparison import assert_close
from lucid.test.helpers.numerics import make_tensor


class TestDtypeEnum:
    def test_float32_exists(self):
        assert lucid.float32 is not None

    def test_float64_exists(self):
        assert lucid.float64 is not None

    def test_int32_exists(self):
        assert lucid.int32 is not None

    def test_bool_exists(self):
        assert lucid.bool_ is not None

    def test_float_alias(self):
        t = lucid.zeros(2, dtype=lucid.float32)
        assert t.dtype == lucid.float32


class TestAstype:
    def test_f32_to_f64(self):
        t = lucid.tensor([1.0, 2.0, 3.0], dtype=lucid.float32)
        out = t.to(lucid.float64)
        assert out.dtype == lucid.float64

    def test_f64_to_f32(self):
        t = lucid.tensor([1.0], dtype=lucid.float64)
        out = t.to(lucid.float32)
        assert out.dtype == lucid.float32

    def test_astype_free_fn(self):
        from lucid._C import engine as _C_engine

        t = make_tensor((4,))
        impl = t._impl
        out = _C_engine.astype(impl, _C_engine.F64)
        assert out.dtype == _C_engine.F64

    def test_values_preserved_f32_to_f64(self):
        t = lucid.tensor([1.5, 2.5, 3.5], dtype=lucid.float32)
        out = t.to(lucid.float64)
        assert_close(out, t.to(lucid.float32), atol=1e-5)

    def test_int_to_float(self):
        t = lucid.tensor([1, 2, 3], dtype=lucid.int32)
        out = t.to(lucid.float32)
        assert out.dtype == lucid.float32
        assert float(out[0].item()) == 1.0


class TestDtypeInConstruction:
    @pytest.mark.parametrize("dtype", [lucid.float32, lucid.float64, lucid.int32])
    def test_dtype_preserved(self, dtype):
        t = lucid.zeros(3, dtype=dtype)
        assert t.dtype == dtype

    def test_zeros_default_f32(self):
        t = lucid.zeros(4)
        assert t.dtype == lucid.float32

    def test_ones_default_f32(self):
        t = lucid.ones(4)
        assert t.dtype == lucid.float32
