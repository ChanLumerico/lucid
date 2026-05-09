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
        assert abs(v - math.pi**2 / 6.0) < 1e-3

    def test_n_ge_4_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            lucid.special.polygamma(4, lucid.tensor([1.0]))


class TestSphericalBesselJ0:
    def test_at_zero(self) -> None:
        # j₀(0) = 1 by continuous extension.
        assert (
            abs(lucid.special.spherical_bessel_j0(lucid.tensor([0.0])).item() - 1.0)
            < 1e-5
        )

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


# ── Orthogonal polynomials ──────────────────────────────────────────────


class TestOrthogonalPolynomials:
    def test_chebyshev_t_known(self) -> None:
        x = lucid.tensor([0.5, 0.0, 1.0])
        # T_0=1, T_1=x, T_2=2x²-1, T_3=4x³-3x.
        np.testing.assert_allclose(
            lucid.special.chebyshev_polynomial_t(x, 0).numpy(), [1.0, 1.0, 1.0]
        )
        np.testing.assert_allclose(
            lucid.special.chebyshev_polynomial_t(x, 2).numpy(), [-0.5, -1.0, 1.0]
        )
        np.testing.assert_allclose(
            lucid.special.chebyshev_polynomial_t(x, 3).numpy(), [-1.0, 0.0, 1.0]
        )

    def test_chebyshev_u_known(self) -> None:
        # U_2(x) = 4x²-1; at 0.5 → 0.
        np.testing.assert_allclose(
            lucid.special.chebyshev_polynomial_u(lucid.tensor([0.5]), 2).numpy(),
            [0.0],
            atol=1e-6,
        )

    def test_shifted_chebyshev_at_one(self) -> None:
        # T*_n(1) = T_n(1) = 1 for all n.
        x = lucid.tensor([1.0])
        for n in range(4):
            np.testing.assert_allclose(
                lucid.special.shifted_chebyshev_polynomial_t(x, n).numpy(),
                [1.0],
                atol=1e-6,
            )

    def test_hermite_h_known(self) -> None:
        # H_2(x)=4x²-2; H_3(x)=8x³-12x.  At x=1 → 2 and -4.
        x = lucid.tensor([1.0])
        np.testing.assert_allclose(
            lucid.special.hermite_polynomial_h(x, 2).numpy(), [2.0]
        )
        np.testing.assert_allclose(
            lucid.special.hermite_polynomial_h(x, 3).numpy(), [-4.0]
        )

    def test_hermite_he_known(self) -> None:
        # He_2=x²-1, He_3=x³-3x at x=2.
        x = lucid.tensor([2.0])
        np.testing.assert_allclose(
            lucid.special.hermite_polynomial_he(x, 2).numpy(), [3.0]
        )
        np.testing.assert_allclose(
            lucid.special.hermite_polynomial_he(x, 3).numpy(), [2.0]
        )

    def test_legendre_p_known(self) -> None:
        # P_2(x) = (3x²-1)/2; at 0.5 → -0.125.
        np.testing.assert_allclose(
            lucid.special.legendre_polynomial_p(lucid.tensor([0.5]), 2).numpy(),
            [-0.125],
            atol=1e-6,
        )

    def test_laguerre_l_known(self) -> None:
        # L_0=1, L_1=1-x, L_2=(x²-4x+2)/2.  At x=1 → 1, 0, -0.5.
        x = lucid.tensor([1.0])
        for n, expected in [(0, 1.0), (1, 0.0), (2, -0.5)]:
            np.testing.assert_allclose(
                lucid.special.laguerre_polynomial_l(x, n).numpy(),
                [expected],
                atol=1e-6,
            )


# ── Bessel ──────────────────────────────────────────────────────────────


class TestBessel:
    def test_j0_known(self) -> None:
        # scipy.special.j0 reference values.
        x = lucid.tensor([0.0, 1.0, 5.0, 2.4048])
        out = lucid.special.bessel_j0(x).numpy()
        np.testing.assert_allclose(out, [1.0, 0.7651977, -0.1775968, 0.0], atol=1e-4)

    def test_j1_known(self) -> None:
        x = lucid.tensor([0.0, 1.0, -1.0, 5.0])
        out = lucid.special.bessel_j1(x).numpy()
        np.testing.assert_allclose(
            out, [0.0, 0.4400506, -0.4400506, -0.3275791], atol=1e-4
        )

    def test_y0_y1_known(self) -> None:
        x = lucid.tensor([1.0, 5.0, 10.0])
        np.testing.assert_allclose(
            lucid.special.bessel_y0(x).numpy(),
            [0.08825696, -0.30851763, 0.05567117],
            atol=1e-4,
        )
        np.testing.assert_allclose(
            lucid.special.bessel_y1(x).numpy(),
            [-0.78121282, 0.14786314, 0.24901543],
            atol=1e-4,
        )

    def test_modified_k0_k1_known(self) -> None:
        x = lucid.tensor([0.5, 1.0, 2.0])
        np.testing.assert_allclose(
            lucid.special.modified_bessel_k0(x).numpy(),
            [0.92441907, 0.42102443, 0.11389387],
            atol=1e-4,
        )
        np.testing.assert_allclose(
            lucid.special.modified_bessel_k1(x).numpy(),
            [1.65644112, 0.60190724, 0.13986588],
            atol=1e-4,
        )

    def test_scaled_modified_k_consistent(self) -> None:
        x_np = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32)
        x = lucid.tensor(x_np)
        unscaled = lucid.special.modified_bessel_k0(x).numpy()
        scaled = lucid.special.scaled_modified_bessel_k0(x).numpy()
        np.testing.assert_allclose(scaled, unscaled * np.exp(x_np), atol=1e-3)


# ── Hurwitz zeta ────────────────────────────────────────────────────────


class TestZeta:
    def test_riemann_special_values(self) -> None:
        import math

        z2 = lucid.special.zeta(lucid.tensor([2.0]), lucid.tensor([1.0])).item()
        z4 = lucid.special.zeta(lucid.tensor([4.0]), lucid.tensor([1.0])).item()
        assert abs(z2 - math.pi**2 / 6.0) < 1e-3
        assert abs(z4 - math.pi**4 / 90.0) < 1e-3

    def test_hurwitz_shift_identity(self) -> None:
        # ζ(s, q+1) = ζ(s, q) - q^{-s}.
        s = lucid.tensor([2.0])
        z_at_1 = lucid.special.zeta(s, lucid.tensor([1.0])).item()
        z_at_2 = lucid.special.zeta(s, lucid.tensor([2.0])).item()
        assert abs((z_at_1 - 1.0) - z_at_2) < 1e-3
