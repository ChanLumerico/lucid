"""Parity: ``lucid.special`` vs reference ``special``."""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.special as ls
from lucid.test._helpers.compare import assert_close


@pytest.mark.parity
class TestOrthogonalPolynomialParity:
    _xs = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)

    def _run(
        self,
        ref: Any,
        lucid_fn,  # type: ignore[no-untyped-def]
        ref_fn,  # type: ignore[no-untyped-def]
        n: int,
        *,
        atol: float = 1e-4,
    ) -> None:
        x_np = self._xs
        l = lucid_fn(lucid.tensor(x_np.copy()), n)
        r = ref_fn(ref.tensor(x_np.copy()), n)
        assert_close(l, r, atol=atol)

    def test_chebyshev_t_n0(self, ref: Any) -> None:
        self._run(ref, ls.chebyshev_polynomial_t, ref.special.chebyshev_polynomial_t, 0)

    def test_chebyshev_t_n1(self, ref: Any) -> None:
        self._run(ref, ls.chebyshev_polynomial_t, ref.special.chebyshev_polynomial_t, 1)

    def test_chebyshev_t_n4(self, ref: Any) -> None:
        self._run(ref, ls.chebyshev_polynomial_t, ref.special.chebyshev_polynomial_t, 4)

    def test_chebyshev_u_n2(self, ref: Any) -> None:
        self._run(ref, ls.chebyshev_polynomial_u, ref.special.chebyshev_polynomial_u, 2)

    def test_chebyshev_u_n3(self, ref: Any) -> None:
        self._run(ref, ls.chebyshev_polynomial_u, ref.special.chebyshev_polynomial_u, 3)

    def test_chebyshev_v_n2(self, ref: Any) -> None:
        self._run(ref, ls.chebyshev_polynomial_v, ref.special.chebyshev_polynomial_v, 2)

    def test_chebyshev_w_n2(self, ref: Any) -> None:
        self._run(ref, ls.chebyshev_polynomial_w, ref.special.chebyshev_polynomial_w, 2)

    def test_shifted_chebyshev_t_n3(self, ref: Any) -> None:
        xs = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        l = ls.shifted_chebyshev_polynomial_t(lucid.tensor(xs.copy()), 3)
        r = ref.special.shifted_chebyshev_polynomial_t(ref.tensor(xs.copy()), 3)
        assert_close(l, r, atol=1e-4)

    def test_shifted_chebyshev_u_n2(self, ref: Any) -> None:
        xs = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        l = ls.shifted_chebyshev_polynomial_u(lucid.tensor(xs.copy()), 2)
        r = ref.special.shifted_chebyshev_polynomial_u(ref.tensor(xs.copy()), 2)
        assert_close(l, r, atol=1e-4)

    def test_shifted_chebyshev_v_n2(self, ref: Any) -> None:
        xs = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        l = ls.shifted_chebyshev_polynomial_v(lucid.tensor(xs.copy()), 2)
        r = ref.special.shifted_chebyshev_polynomial_v(ref.tensor(xs.copy()), 2)
        assert_close(l, r, atol=1e-4)

    def test_shifted_chebyshev_w_n2(self, ref: Any) -> None:
        xs = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        l = ls.shifted_chebyshev_polynomial_w(lucid.tensor(xs.copy()), 2)
        r = ref.special.shifted_chebyshev_polynomial_w(ref.tensor(xs.copy()), 2)
        assert_close(l, r, atol=1e-4)

    def test_hermite_h_n2(self, ref: Any) -> None:
        self._run(ref, ls.hermite_polynomial_h, ref.special.hermite_polynomial_h, 2)

    def test_hermite_h_n4(self, ref: Any) -> None:
        self._run(ref, ls.hermite_polynomial_h, ref.special.hermite_polynomial_h, 4)

    def test_hermite_he_n3(self, ref: Any) -> None:
        self._run(ref, ls.hermite_polynomial_he, ref.special.hermite_polynomial_he, 3)

    def test_legendre_p_n2(self, ref: Any) -> None:
        self._run(ref, ls.legendre_polynomial_p, ref.special.legendre_polynomial_p, 2)

    def test_legendre_p_n4(self, ref: Any) -> None:
        self._run(ref, ls.legendre_polynomial_p, ref.special.legendre_polynomial_p, 4)

    def test_laguerre_l_n2(self, ref: Any) -> None:
        xs = np.array([0.0, 0.5, 1.0, 2.0, 3.0], dtype=np.float32)
        l = ls.laguerre_polynomial_l(lucid.tensor(xs.copy()), 2)
        r = ref.special.laguerre_polynomial_l(ref.tensor(xs.copy()), 2)
        assert_close(l, r, atol=1e-4)

    def test_laguerre_l_n3(self, ref: Any) -> None:
        xs = np.array([0.0, 0.5, 1.0, 2.0, 3.0], dtype=np.float32)
        l = ls.laguerre_polynomial_l(lucid.tensor(xs.copy()), 3)
        r = ref.special.laguerre_polynomial_l(ref.tensor(xs.copy()), 3)
        assert_close(l, r, atol=1e-4)


@pytest.mark.parity
class TestBesselParity:
    _xs_pos = np.array([0.5, 1.0, 2.0, 5.0, 10.0], dtype=np.float32)
    _xs_signed = np.array([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=np.float32)

    def test_bessel_j0(self, ref: Any) -> None:
        xs = self._xs_pos
        assert_close(
            ls.bessel_j0(lucid.tensor(xs.copy())),
            ref.special.bessel_j0(ref.tensor(xs.copy())),
            atol=1e-4,
        )

    def test_bessel_j0_signed(self, ref: Any) -> None:
        xs = self._xs_signed
        assert_close(
            ls.bessel_j0(lucid.tensor(xs.copy())),
            ref.special.bessel_j0(ref.tensor(xs.copy())),
            atol=1e-4,
        )

    def test_bessel_j1(self, ref: Any) -> None:
        xs = self._xs_signed
        assert_close(
            ls.bessel_j1(lucid.tensor(xs.copy())),
            ref.special.bessel_j1(ref.tensor(xs.copy())),
            atol=1e-4,
        )

    def test_bessel_y0(self, ref: Any) -> None:
        xs = self._xs_pos
        assert_close(
            ls.bessel_y0(lucid.tensor(xs.copy())),
            ref.special.bessel_y0(ref.tensor(xs.copy())),
            atol=1e-4,
        )

    def test_bessel_y1(self, ref: Any) -> None:
        xs = self._xs_pos
        assert_close(
            ls.bessel_y1(lucid.tensor(xs.copy())),
            ref.special.bessel_y1(ref.tensor(xs.copy())),
            atol=1e-4,
        )

    def test_modified_bessel_k0(self, ref: Any) -> None:
        xs = self._xs_pos
        assert_close(
            ls.modified_bessel_k0(lucid.tensor(xs.copy())),
            ref.special.modified_bessel_k0(ref.tensor(xs.copy())),
            atol=1e-4,
        )

    def test_modified_bessel_k1(self, ref: Any) -> None:
        xs = self._xs_pos
        assert_close(
            ls.modified_bessel_k1(lucid.tensor(xs.copy())),
            ref.special.modified_bessel_k1(ref.tensor(xs.copy())),
            atol=1e-4,
        )

    def test_scaled_modified_bessel_k0(self, ref: Any) -> None:
        xs = self._xs_pos
        assert_close(
            ls.scaled_modified_bessel_k0(lucid.tensor(xs.copy())),
            ref.special.scaled_modified_bessel_k0(ref.tensor(xs.copy())),
            atol=1e-4,
        )

    def test_scaled_modified_bessel_k1(self, ref: Any) -> None:
        xs = self._xs_pos
        assert_close(
            ls.scaled_modified_bessel_k1(lucid.tensor(xs.copy())),
            ref.special.scaled_modified_bessel_k1(ref.tensor(xs.copy())),
            atol=1e-4,
        )


@pytest.mark.parity
class TestZetaParity:
    def test_riemann_special_values(self, ref: Any) -> None:
        s_np = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        q_np = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        assert_close(
            ls.zeta(lucid.tensor(s_np.copy()), lucid.tensor(q_np.copy())),
            ref.special.zeta(ref.tensor(s_np.copy()), ref.tensor(q_np.copy())),
            atol=1e-3,
        )

    def test_hurwitz_shift(self, ref: Any) -> None:
        s_np = np.array([2.0, 3.0], dtype=np.float32)
        q_np = np.array([2.0, 2.0], dtype=np.float32)
        assert_close(
            ls.zeta(lucid.tensor(s_np.copy()), lucid.tensor(q_np.copy())),
            ref.special.zeta(ref.tensor(s_np.copy()), ref.tensor(q_np.copy())),
            atol=1e-3,
        )
