"""``lucid.special`` — special-math sub-package."""

import math

import numpy as np
import pytest

import lucid


# Top-level erf/erfc tested in unit/ops/unary; this file covers the
# special-only entries that don't have a top-level alias.


class TestErfcx:
    def test_at_zero(self) -> None:
        # erfcx(0) = exp(0)·erfc(0) = 1.0.
        assert abs(lucid.special.erfcx(lucid.tensor([0.0])).item() - 1.0) < 1e-5


class TestI0e:
    def test_at_zero(self) -> None:
        # i0e(0) = exp(0)·I_0(0) = 1.
        assert abs(lucid.special.i0e(lucid.tensor([0.0])).item() - 1.0) < 1e-5


class TestI1:
    def test_at_zero(self) -> None:
        # I_1(0) = 0.
        assert abs(lucid.special.i1(lucid.tensor([0.0])).item()) < 1e-6


class TestI1e:
    def test_at_zero(self) -> None:
        assert abs(lucid.special.i1e(lucid.tensor([0.0])).item()) < 1e-6


class TestNdtr:
    def test_at_zero(self) -> None:
        # Φ(0) = 0.5.
        assert abs(lucid.special.ndtr(lucid.tensor([0.0])).item() - 0.5) < 1e-5

    def test_extreme(self) -> None:
        assert lucid.special.ndtr(lucid.tensor([5.0])).item() > 0.99
        assert lucid.special.ndtr(lucid.tensor([-5.0])).item() < 0.01


class TestNdtri:
    def test_inverse_of_half(self) -> None:
        # ndtri(0.5) = 0.
        assert abs(lucid.special.ndtri(lucid.tensor([0.5])).item()) < 1e-5

    def test_round_trip(self) -> None:
        x = lucid.tensor([-1.5, -0.5, 0.5, 1.5])
        round_trip = lucid.special.ndtri(lucid.special.ndtr(x)).numpy()
        np.testing.assert_allclose(round_trip, x.numpy(), atol=1e-3)


class TestLogNdtr:
    def test_at_zero(self) -> None:
        # log Φ(0) = log 0.5 = -log 2.
        v = lucid.special.log_ndtr(lucid.tensor([0.0])).item()
        assert abs(v - (-math.log(2.0))) < 1e-5


class TestXlog1py:
    def test_zero_zero(self) -> None:
        # 0·log(1+0) = 0.
        out = lucid.special.xlog1py(lucid.tensor([0.0]), lucid.tensor([0.0])).item()
        assert out == 0.0

    def test_known(self) -> None:
        # 2·log(1+1) = 2·log 2.
        out = lucid.special.xlog1py(lucid.tensor([2.0]), lucid.tensor([1.0])).item()
        assert abs(out - 2.0 * math.log(2.0)) < 1e-5


class TestEntr:
    def test_at_one(self) -> None:
        # entr(1) = -1·log(1) = 0.
        assert abs(lucid.special.entr(lucid.tensor([1.0])).item()) < 1e-6

    def test_at_half(self) -> None:
        # entr(0.5) = -0.5·log(0.5) = 0.5·log 2.
        v = lucid.special.entr(lucid.tensor([0.5])).item()
        assert abs(v - 0.5 * math.log(2.0)) < 1e-5


class TestPolygamma:
    def test_n0_eq_digamma(self) -> None:
        x = lucid.tensor([1.0, 2.0, 5.0])
        np.testing.assert_allclose(
            lucid.special.polygamma(0, x).numpy(),
            lucid.digamma(x).numpy(),
            atol=1e-5,
        )

    def test_n1_at_one_pi2_over_6(self) -> None:
        # ψ¹(1) = π²/6.
        v = lucid.special.polygamma(1, lucid.tensor([1.0])).item()
        assert abs(v - math.pi ** 2 / 6.0) < 1e-3

    def test_n_ge_4_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            lucid.special.polygamma(4, lucid.tensor([1.0]))


class TestSphericalBesselJ0:
    def test_at_zero(self) -> None:
        # j₀(0) = 1 by continuous extension.
        assert abs(lucid.special.spherical_bessel_j0(lucid.tensor([0.0])).item() - 1.0) < 1e-5

    def test_at_pi(self) -> None:
        # sin(π)/π = 0.
        v = lucid.special.spherical_bessel_j0(lucid.tensor([math.pi])).item()
        assert abs(v) < 1e-5


class TestMultigammaln:
    def test_p1_eq_lgamma(self) -> None:
        # multigammaln(a, p=1) = lgamma(a).
        x = lucid.tensor([2.0, 3.0, 4.0])
        np.testing.assert_allclose(
            lucid.special.multigammaln(x, 1).numpy(),
            lucid.lgamma(x).numpy(),
            atol=1e-5,
        )
