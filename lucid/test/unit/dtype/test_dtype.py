"""Dtype objects, casting, finfo / iinfo, promotion."""

import numpy as np
import pytest

import lucid
from lucid.test._fixtures.devices import skip_if_unsupported


class TestDtypeObjects:
    def test_distinct_identities(self) -> None:
        # Each dtype should be a distinct value.
        assert lucid.float32 != lucid.float64
        assert lucid.int32 != lucid.int64
        assert lucid.bool_ != lucid.int32

    def test_repr_contains_name(self) -> None:
        assert "float32" in str(lucid.float32)
        assert "int64" in str(lucid.int64)


class TestCasting:
    def test_int_to_float(self, device: str) -> None:
        t = lucid.tensor([1, 2, 3], dtype=lucid.int32, device=device)
        out = t.to(dtype=lucid.float32)
        assert out.dtype == lucid.float32
        np.testing.assert_array_equal(out.numpy(), [1.0, 2.0, 3.0])

    def test_float_to_int_truncates(self, device: str) -> None:
        t = lucid.tensor([1.7, -2.3, 3.9], device=device)
        out = t.to(dtype=lucid.int32)
        np.testing.assert_array_equal(out.numpy(), [1, -2, 3])

    def test_float_to_bool(self, device: str) -> None:
        t = lucid.tensor([0.0, 1.0, -1.0, 0.0], device=device)
        out = t.to(dtype=lucid.bool_)
        np.testing.assert_array_equal(out.numpy(), [False, True, True, False])


class TestFinfo:
    def test_float32_eps(self) -> None:
        info = lucid.finfo(lucid.float32)
        assert info.eps == np.finfo(np.float32).eps

    def test_float32_max(self) -> None:
        info = lucid.finfo(lucid.float32)
        assert info.max == np.finfo(np.float32).max

    def test_float64_eps(self) -> None:
        info = lucid.finfo(lucid.float64)
        assert info.eps == np.finfo(np.float64).eps

    def test_unsupported_raises(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            lucid.finfo(lucid.int32)


class TestIinfo:
    def test_int32_bounds(self) -> None:
        info = lucid.iinfo(lucid.int32)
        assert info.min == -(2 ** 31)
        assert info.max == 2 ** 31 - 1

    def test_int64_bounds(self) -> None:
        info = lucid.iinfo(lucid.int64)
        assert info.min == -(2 ** 63)
        assert info.max == 2 ** 63 - 1


class TestPromotion:
    def test_op_rejects_mixed_dtypes(self, device: str) -> None:
        # Lucid's binary kernels do *not* auto-promote — adding f32 to
        # i32 raises with a clear message.  Users cast explicitly.
        a = lucid.tensor([1.5], device=device)
        b = lucid.tensor([2], dtype=lucid.int32, device=device)
        with pytest.raises(Exception):
            _ = a + b

    def test_explicit_cast_then_op(self, device: str) -> None:
        a = lucid.tensor([1.5], device=device)
        b = lucid.tensor([2], dtype=lucid.int32, device=device)
        out = a + b.to(dtype=a.dtype)
        assert out.dtype == lucid.float32
        np.testing.assert_allclose(out.numpy(), [3.5])

    def test_promote_types_int_float(self) -> None:
        # ``promote_types(i32, f32)`` lifts the int side into f32.
        assert lucid.promote_types(lucid.int32, lucid.float32) == lucid.float32

    def test_promote_types_widens_int(self) -> None:
        assert lucid.promote_types(lucid.int32, lucid.int64) == lucid.int64

    def test_promote_types_widens_float(self) -> None:
        assert lucid.promote_types(lucid.float32, lucid.float64) == lucid.float64

    def test_can_cast_safe(self) -> None:
        # int32 fits in float64 — safe cast.
        assert lucid.can_cast(lucid.int32, lucid.float64)
        # float64 → int32 is not a safe cast (loses precision).
        assert not lucid.can_cast(lucid.float64, lucid.int32)


class TestDtypeAliases:
    def test_half_is_float16(self) -> None:
        assert lucid.half == lucid.float16

    def test_double_is_float64(self) -> None:
        assert lucid.double == lucid.float64

    def test_short_is_int16(self) -> None:
        assert lucid.short == lucid.int16

    def test_long_is_int64(self) -> None:
        assert lucid.long == lucid.int64
