"""Unit tests for the P2-B complex viewing surface — engine ops
(``real`` / ``imag`` / ``complex`` / ``conj``), composites
(``angle`` / ``polar`` / ``view_as_real`` / ``view_as_complex``),
plus the C64 backend extensions (``full(C64)`` / ``ones(C64)`` /
``mul(C64, C64)``).
"""

import math

import numpy as np
import pytest

import lucid


def _c(re_list: list[float], im_list: list[float]) -> lucid.Tensor:
    """Build a small complex tensor from two Python lists."""
    return lucid.complex(lucid.tensor(re_list), lucid.tensor(im_list))


class TestComplexBuilder:
    def test_complex_combines_real_imag(self) -> None:
        c = _c([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        np.testing.assert_array_equal(c.numpy(), [1 + 4j, 2 + 5j, 3 + 6j])
        assert c.dtype is lucid.complex64

    def test_complex_shape_mismatch_raises(self) -> None:
        with pytest.raises(Exception):
            lucid.complex(lucid.tensor([1.0, 2.0]), lucid.tensor([3.0]))

    def test_complex_2d(self) -> None:
        re = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        im = lucid.tensor([[5.0, 6.0], [7.0, 8.0]])
        c = lucid.complex(re, im)
        assert c.shape == (2, 2)
        np.testing.assert_array_equal(
            c.numpy(),
            [[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]],
        )


class TestRealImag:
    def test_real_extracts(self) -> None:
        c = _c([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        np.testing.assert_array_equal(lucid.real(c).numpy(), [1.0, 2.0, 3.0])
        assert lucid.real(c).dtype is lucid.float32

    def test_imag_extracts(self) -> None:
        c = _c([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        np.testing.assert_array_equal(lucid.imag(c).numpy(), [4.0, 5.0, 6.0])
        assert lucid.imag(c).dtype is lucid.float32

    def test_real_imag_round_trip(self) -> None:
        c = _c([1.0, -2.0, 3.5], [-0.5, 1.5, 0.0])
        round_trip = lucid.complex(lucid.real(c), lucid.imag(c))
        np.testing.assert_allclose(round_trip.numpy(), c.numpy())

    def test_real_rejects_real_input(self) -> None:
        with pytest.raises(Exception):
            lucid.real(lucid.tensor([1.0, 2.0]))


class TestConj:
    def test_conj_negates_imag(self) -> None:
        c = _c([1.0, 2.0], [3.0, -4.0])
        np.testing.assert_array_equal(lucid.conj(c).numpy(), [1 - 3j, 2 + 4j])

    def test_conj_real_is_identity(self) -> None:
        x = lucid.tensor([1.0, -2.0, 3.0])
        np.testing.assert_array_equal(lucid.conj(x).numpy(), x.numpy())

    def test_conj_double_is_identity(self) -> None:
        c = _c([1.0, 2.0], [3.0, 4.0])
        np.testing.assert_allclose(lucid.conj(lucid.conj(c)).numpy(), c.numpy())


class TestC64BackendExtensions:
    def test_full_c64_via_top_level(self) -> None:
        # The standard ``Tensor * float`` path uses ``full(C64)`` under
        # the hood — this exercises that call path.
        c = _c([1.0, 2.0], [3.0, 4.0])
        result = c * 0.5
        np.testing.assert_allclose(result.numpy(), [0.5 + 1.5j, 1.0 + 2.0j])

    def test_mul_c64_x_c64(self) -> None:
        a = _c([1.0, 0.0], [0.0, 1.0])  # [1, j]
        b = _c([0.0, 1.0], [1.0, 0.0])  # [j, 1]
        # (1)(j) = j, (j)(1) = j
        np.testing.assert_allclose((a * b).numpy(), [0.0 + 1.0j, 0.0 + 1.0j])

    def test_full_c64_engine(self) -> None:
        # Direct check of full(C64, value) — fills with (value, 0).
        from lucid._C import engine
        from lucid._dispatch import _wrap

        out = _wrap(engine.full([3], 2.5, engine.C64, engine.CPU))
        np.testing.assert_array_equal(out.numpy(), [2.5 + 0j, 2.5 + 0j, 2.5 + 0j])


class TestAngle:
    def test_angle_real_positive(self) -> None:
        # angle(1 + 0j) == 0
        c = _c([1.0], [0.0])
        assert abs(lucid.angle(c).item()) < 1e-6

    def test_angle_imag(self) -> None:
        # angle(0 + 1j) == π/2
        c = _c([0.0], [1.0])
        assert abs(lucid.angle(c).item() - math.pi / 2) < 1e-5

    def test_angle_negative_real(self) -> None:
        # angle(-1 + 0j) == π
        c = _c([-1.0], [0.0])
        assert abs(lucid.angle(c).item() - math.pi) < 1e-5


class TestPolar:
    def test_polar_unit_circle(self) -> None:
        # polar(1, 0) == 1 + 0j
        c = lucid.polar(lucid.tensor([1.0]), lucid.tensor([0.0]))
        np.testing.assert_allclose(c.numpy(), [1 + 0j], atol=1e-6)

    def test_polar_round_trip(self) -> None:
        # polar(|c|, angle(c)) ≈ c (for non-zero c)
        c = _c([3.0, -1.0], [4.0, 1.0])  # |c|=[5, sqrt(2)]
        magnitude = lucid.sqrt(lucid.real(c) ** 2 + lucid.imag(c) ** 2)
        c_back = lucid.polar(magnitude, lucid.angle(c))
        np.testing.assert_allclose(c_back.numpy(), c.numpy(), atol=1e-5)


class TestViewAsReal:
    def test_view_as_real_shape(self) -> None:
        c = _c([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        v = lucid.view_as_real(c)
        assert v.shape == (3, 2)
        np.testing.assert_array_equal(
            v.numpy(),
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
        )

    def test_view_as_real_dtype(self) -> None:
        c = _c([1.0], [2.0])
        assert lucid.view_as_real(c).dtype is lucid.float32

    def test_view_as_complex_round_trip(self) -> None:
        c = _c([1.0, 2.0], [3.0, 4.0])
        v = lucid.view_as_real(c)
        c_back = lucid.view_as_complex(v)
        np.testing.assert_allclose(c_back.numpy(), c.numpy())

    def test_view_as_complex_rejects_bad_shape(self) -> None:
        with pytest.raises(ValueError):
            lucid.view_as_complex(lucid.tensor([1.0, 2.0, 3.0]))  # last dim != 2


class TestSurfacePolicy:
    def test_complex_ops_at_top_level(self) -> None:
        # All four engine ops + four composites are accessible at top level.
        for name in (
            "real",
            "imag",
            "complex",
            "conj",
            "angle",
            "polar",
            "view_as_real",
            "view_as_complex",
        ):
            assert hasattr(lucid, name), f"lucid.{name} should be exposed"
            assert callable(getattr(lucid, name))
