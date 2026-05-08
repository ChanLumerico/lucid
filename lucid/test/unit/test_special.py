"""Unit tests for ``lucid.special``.

Each function is checked against a closed-form expectation or a known
reference value from the literature (Euler-Mascheroni, Apéry's constant,
``ζ(2) = π² / 6``, etc.).  No reference framework dependency — these
tests stand alone.
"""

import math

import numpy as np
import pytest

import lucid
import lucid.special as sp

_ATOL_DEFAULT = 1e-4
_ATOL_LOOSE = 5e-3  # for Abramowitz polynomial-based ops (i1, etc.)


class TestErfcx:
    def test_at_zero(self) -> None:
        assert abs(sp.erfcx(lucid.tensor([0.0])).item() - 1.0) < _ATOL_DEFAULT

    def test_positive(self) -> None:
        # erfcx(1) = e · erfc(1) ≈ 2.71828 · 0.15730 ≈ 0.42758
        assert abs(sp.erfcx(lucid.tensor([1.0])).item() - 0.42758) < _ATOL_DEFAULT


class TestI0eI1I1e:
    def test_i0e_at_zero(self) -> None:
        # I0(0) = 1, exp(0) = 1 → i0e(0) = 1
        assert abs(sp.i0e(lucid.tensor([0.0])).item() - 1.0) < _ATOL_DEFAULT

    def test_i1_at_zero(self) -> None:
        # I1(0) = 0
        assert abs(sp.i1(lucid.tensor([0.0])).item()) < _ATOL_DEFAULT

    def test_i1_is_odd(self) -> None:
        x = lucid.tensor([0.5, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            sp.i1(x).numpy(), -sp.i1(-x).numpy(), atol=_ATOL_LOOSE
        )

    def test_i1_known_value(self) -> None:
        # I1(1) ≈ 0.565159
        assert abs(sp.i1(lucid.tensor([1.0])).item() - 0.565159) < _ATOL_LOOSE

    def test_i1e_decay(self) -> None:
        # i1e(x) = exp(-|x|) · I1(x); for large x this stays bounded
        # (i1 itself blows up exponentially).
        out = sp.i1e(lucid.tensor([5.0])).item()
        assert 0.0 < out < 1.0


class TestNdtr:
    def test_at_zero(self) -> None:
        assert abs(sp.ndtr(lucid.tensor([0.0])).item() - 0.5) < 1e-6

    def test_at_1_96(self) -> None:
        # 97.5% one-sided z-score
        assert abs(sp.ndtr(lucid.tensor([1.96])).item() - 0.975) < 1e-3

    def test_at_minus_inf_like(self) -> None:
        # Φ(-5) ≈ 2.87e-7 — numerically tiny but finite.
        out = sp.ndtr(lucid.tensor([-5.0])).item()
        assert 0.0 < out < 1e-5

    def test_log_ndtr_left_tail(self) -> None:
        # Direct log(ndtr(-5)) underflows; log_ndtr should give a finite
        # value close to -15.07.
        out = sp.log_ndtr(lucid.tensor([-5.0])).item()
        assert -16.0 < out < -14.5
        assert math.isfinite(out)


class TestNdtri:
    def test_at_half(self) -> None:
        assert abs(sp.ndtri(lucid.tensor([0.5])).item()) < 1e-3

    def test_at_975(self) -> None:
        assert abs(sp.ndtri(lucid.tensor([0.975])).item() - 1.959964) < 1e-3

    def test_inverse_round_trip(self) -> None:
        # ndtri(ndtr(x)) ≈ x for x in a moderate range.
        x = lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        round_trip = sp.ndtri(sp.ndtr(x)).numpy()
        np.testing.assert_allclose(round_trip, x.numpy(), atol=1e-3)


class TestXlog1py:
    def test_zero_zero(self) -> None:
        # 0 · log1p(0) = 0 by convention.
        assert sp.xlog1py(lucid.tensor([0.0]), lucid.tensor([0.0])).item() == 0.0

    def test_2_1(self) -> None:
        # 2 · log(2)
        assert (
            abs(
                sp.xlog1py(lucid.tensor([2.0]), lucid.tensor([1.0])).item()
                - 2.0 * math.log(2.0)
            )
            < 1e-6
        )

    def test_x_zero_propagates(self) -> None:
        out = sp.xlog1py(lucid.tensor([0.0]), lucid.tensor([5.0])).item()
        assert out == 0.0


class TestEntr:
    def test_entr_zero(self) -> None:
        # 0·log(0) = 0 by convention.
        assert sp.entr(lucid.tensor([0.0])).item() == 0.0

    def test_entr_one(self) -> None:
        # 1 · log(1) = 0.
        assert abs(sp.entr(lucid.tensor([1.0])).item()) < 1e-6

    def test_entr_half(self) -> None:
        # entr(0.5) = -0.5 · log(0.5) = 0.5 · log(2) ≈ 0.34657
        assert abs(sp.entr(lucid.tensor([0.5])).item() - 0.5 * math.log(2.0)) < 1e-6

    def test_entr_negative_is_nan(self) -> None:
        out = sp.entr(lucid.tensor([-1.0])).item()
        assert math.isnan(out)


class TestMultigammaln:
    def test_p_one_eq_lgamma(self) -> None:
        # Γ_1(a) = Γ(a), so multigammaln(a, 1) = lgamma(a).
        a = lucid.tensor([3.5])
        assert abs(sp.multigammaln(a, 1).item() - lucid.lgamma(a).item()) < 1e-5

    def test_p_three_known(self) -> None:
        # multigammaln(5, 3) ≈ 9.1406 (cross-check value from SciPy).
        assert abs(sp.multigammaln(lucid.tensor([5.0]), 3).item() - 9.1406) < 1e-3

    def test_p_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            sp.multigammaln(lucid.tensor([1.0]), 0)


class TestPolygamma:
    def test_n0_eq_digamma(self) -> None:
        # polygamma(0, 1) = ψ(1) = -γ ≈ -0.5772
        assert (
            abs(sp.polygamma(0, lucid.tensor([1.0])).item() - (-0.5772156649))
            < _ATOL_DEFAULT
        )

    def test_n1_at_one_is_pi2_over_6(self) -> None:
        # ψ¹(1) = ζ(2) = π²/6 ≈ 1.6449
        assert (
            abs(sp.polygamma(1, lucid.tensor([1.0])).item() - math.pi**2 / 6.0) < 1e-4
        )

    def test_n1_at_two(self) -> None:
        # ψ¹(2) = ζ(2) - 1 ≈ 0.6449
        assert (
            abs(sp.polygamma(1, lucid.tensor([2.0])).item() - (math.pi**2 / 6.0 - 1.0))
            < 1e-4
        )

    def test_n2_negative(self) -> None:
        # ψ²(x) is negative for x > 0; ψ²(1) = -2·ζ(3) ≈ -2.404.
        v: float = float(sp.polygamma(2, lucid.tensor([1.0])).item())
        assert v < 0
        assert abs(v + 2.0 * 1.20205690315959428) < 1e-3  # 2·ζ(3).

    def test_n3_positive(self) -> None:
        # ψ³(x) > 0 for x > 0; ψ³(1) = 6·ζ(4) = π⁴/15 ≈ 6.494.
        v: float = float(sp.polygamma(3, lucid.tensor([1.0])).item())
        assert v > 0
        assert abs(v - math.pi**4 / 15.0) < 1e-3

    def test_n_ge_4_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            sp.polygamma(4, lucid.tensor([1.0]))


class TestSphericalBessel:
    def test_j0_at_zero(self) -> None:
        # sin(0)/0 = 1 by continuous extension.
        assert abs(sp.spherical_bessel_j0(lucid.tensor([0.0])).item() - 1.0) < 1e-6

    def test_j0_at_pi(self) -> None:
        # sin(π)/π = 0
        assert abs(sp.spherical_bessel_j0(lucid.tensor([math.pi])).item()) < 1e-5

    def test_j0_at_one(self) -> None:
        # sin(1)/1 = sin(1) ≈ 0.84147
        assert (
            abs(sp.spherical_bessel_j0(lucid.tensor([1.0])).item() - math.sin(1.0))
            < 1e-6
        )


class TestNamespacePolicy:
    def test_only_via_special(self) -> None:
        # New special functions live under ``lucid.special.<name>`` only;
        # no top-level shortcut.
        for name in (
            "erfcx",
            "i0e",
            "i1",
            "i1e",
            "ndtr",
            "ndtri",
            "log_ndtr",
            "xlog1py",
            "entr",
            "multigammaln",
            "polygamma",
            "spherical_bessel_j0",
        ):
            assert not hasattr(lucid, name), (
                f"lucid.{name} should not exist — H8 forbids special "
                f"shortcuts; use lucid.special.{name}"
            )

    def test_existing_topfeature_kept_at_top(self) -> None:
        # Conversely, the ones that *were* already at top-level (and have
        # been there since the engine layer was built) remain there and
        # are NOT re-exported under lucid.special.
        for name in ("erf", "erfc", "erfinv", "sinc", "lgamma", "digamma", "i0"):
            assert hasattr(lucid, name)
            assert not hasattr(sp, name), (
                f"lucid.special.{name} should NOT exist — already canonical "
                f"at top level (H8: single canonical path)."
            )
