"""Unit tests for lucid.func functional transforms (grad, vjp, jvp, jacrev, jacfwd, hessian)."""

import pytest
import lucid
import lucid.func as func


# ── func.grad ─────────────────────────────────────────────────────────────────


class TestFuncGrad:
    def test_scalar_quadratic(self) -> None:
        f = lambda x: (x ** 2).sum()
        df = func.grad(f)
        x = lucid.tensor([1.0, 2.0, 3.0])
        g = df(x)
        assert lucid.allclose(g, x * 2)

    def test_argnums_single(self) -> None:
        f = lambda x, y: (x * y).sum()
        df = func.grad(f, argnums=1)
        x = lucid.tensor([2.0, 3.0])
        y = lucid.tensor([1.0, 4.0])
        g = df(x, y)
        assert lucid.allclose(g, x)

    def test_argnums_tuple(self) -> None:
        f = lambda x, y: (x * y).sum()
        df = func.grad(f, argnums=(0, 1))
        x = lucid.tensor([2.0, 3.0])
        y = lucid.tensor([1.0, 4.0])
        gx, gy = df(x, y)
        assert lucid.allclose(gx, y)
        assert lucid.allclose(gy, x)

    def test_has_aux(self) -> None:
        f = lambda x: ((x ** 2).sum(), x * 10)
        df = func.grad(f, has_aux=True)
        x = lucid.tensor([1.0, 2.0])
        g, aux = df(x)
        assert lucid.allclose(g, x * 2)
        assert lucid.allclose(aux, x * 10)

    def test_second_derivative_via_hessian(self) -> None:
        """Second derivative via func.hessian (the correct composable path)."""
        f = lambda x: (x ** 3).sum()
        H = func.hessian(f)(lucid.tensor([2.0]))
        # d²/dx²(x³) = 6x → at x=2: 12
        assert abs(float(H.item()) - 12.0) < 1e-2


# ── func.grad_and_value ───────────────────────────────────────────────────────


class TestGradAndValue:
    def test_returns_grad_and_value(self) -> None:
        f = lambda x: (x ** 2).sum()
        gv = func.grad_and_value(f)
        x = lucid.tensor([1.0, 2.0, 3.0])
        g, v = gv(x)
        assert lucid.allclose(g, x * 2)
        assert abs(float(v.item()) - 14.0) < 1e-5


# ── func.vjp ─────────────────────────────────────────────────────────────────


class TestFuncVJP:
    def test_vjp_quadratic(self) -> None:
        f = lambda x: x ** 2
        x = lucid.tensor([1.0, 2.0, 3.0])
        y, vjp_fn = func.vjp(f, x)
        (g,) = vjp_fn(lucid.ones_like(y))
        assert lucid.allclose(g, x * 2)

    def test_vjp_output_shape(self) -> None:
        f = lambda x: x @ x.T
        x = lucid.randn(3, 4)
        y, vjp_fn = func.vjp(f, x)
        assert list(y.shape) == [3, 3]
        (g,) = vjp_fn(lucid.ones(3, 3))
        assert list(g.shape) == [3, 4]

    def test_vjp_multiple_primals(self) -> None:
        f = lambda x, y: x * y
        x = lucid.tensor([2.0, 3.0])
        y = lucid.tensor([4.0, 5.0])
        out, vjp_fn = func.vjp(f, x, y)
        gx, gy = vjp_fn(lucid.ones_like(out))
        assert lucid.allclose(gx, y)
        assert lucid.allclose(gy, x)

    def test_vjp_has_aux(self) -> None:
        f = lambda x: (x ** 2, x * 3)
        x = lucid.tensor([1.0, 2.0])
        (y, aux), vjp_fn = func.vjp(f, x, has_aux=True)
        assert lucid.allclose(aux, x * 3)
        (g,) = vjp_fn(lucid.ones_like(y))
        assert lucid.allclose(g, x * 2)


# ── func.jvp ─────────────────────────────────────────────────────────────────


class TestFuncJVP:
    def test_jvp_scalar_exact(self) -> None:
        """JVP of x²: d/dt (x+tv)² at t=0 = 2xv."""
        f = lambda x: (x ** 2).sum()
        x = lucid.tensor([1.0, 2.0, 3.0])
        v = lucid.ones(3)
        primals_out, tangents_out = func.jvp(f, (x,), (v,))
        # d/dt sum((x+tv)²) = 2*(x·v) = 2*(1+2+3) = 12
        assert abs(float(tangents_out.item()) - 12.0) < 1e-3

    def test_jvp_linear(self) -> None:
        """JVP of a diagonal scale function: J = diag(scales)."""
        # f(x) = [x[0], 2*x[1]] via elementwise multiply — avoids 1D matmul
        scales = lucid.tensor([1.0, 2.0])
        f = lambda x: x * scales
        x = lucid.tensor([1.0, 1.0])
        v = lucid.tensor([1.0, 1.0])
        primals_out, tangents_out = func.jvp(f, (x,), (v,))
        # J = diag(scales), JVP = scales * v = [1, 2]
        assert lucid.allclose(tangents_out, lucid.tensor([1.0, 2.0]), atol=1e-3)

    def test_jvp_matches_vjp(self) -> None:
        """For diagonal Jacobian: JVP and VJP with same vector should agree."""
        f = lambda x: x * 2
        x = lucid.tensor([1.0, 2.0, 3.0])
        v = lucid.tensor([1.0, 0.0, 0.0])
        _, jvp_out = func.jvp(f, (x,), (v,))
        _, vjp_fn = func.vjp(f, x)
        (vjp_out,) = vjp_fn(v)
        # For f(x)=2x: J = 2I, so Jv = 2v[0]*e_0 = [2, 0, 0] — same as VJP
        assert lucid.allclose(jvp_out, vjp_out, atol=1e-3)

    def test_jvp_strict_raises(self) -> None:
        x = lucid.tensor([1.0, 2.0])
        v = lucid.ones(2)
        # constant output does not depend on x
        f = lambda x: lucid.ones(2)
        with pytest.raises((ValueError, RuntimeError)):
            func.jvp(f, (x,), (v,), strict=True)


# ── func.jacrev ───────────────────────────────────────────────────────────────


class TestJacRev:
    def test_identity_jacobian(self) -> None:
        f = lambda x: x
        x = lucid.tensor([1.0, 2.0, 3.0])
        J = func.jacrev(f)(x)
        assert list(J.shape) == [3, 3]
        assert lucid.allclose(J, lucid.eye(3))

    def test_linear_jacobian(self) -> None:
        # f(x) = x * [1, 3] — diagonal linear map, J = diag([1, 3])
        scales = lucid.tensor([1.0, 3.0])
        f = lambda x: x * scales
        x = lucid.tensor([2.0, 5.0])
        J = func.jacrev(f)(x)
        assert list(J.shape) == [2, 2]
        expected = lucid.tensor([[1.0, 0.0], [0.0, 3.0]])
        assert lucid.allclose(J, expected, atol=1e-4)

    def test_scalar_output_matches_grad(self) -> None:
        f = lambda x: (x ** 2).sum()
        x = lucid.tensor([1.0, 2.0, 3.0])
        J = func.jacrev(f)(x)
        df = func.grad(f)
        g = df(x)
        assert lucid.allclose(J.reshape(-1), g.reshape(-1), atol=1e-4)


# ── func.jacfwd ───────────────────────────────────────────────────────────────


class TestJacFwd:
    def test_identity_jacobian(self) -> None:
        f = lambda x: x
        x = lucid.tensor([1.0, 2.0, 3.0])
        J = func.jacfwd(f)(x)
        assert list(J.shape) == [3, 3]
        assert lucid.allclose(J, lucid.eye(3), atol=1e-3)

    def test_jacfwd_matches_jacrev(self) -> None:
        # Use elementwise scaling — works for any shape, no matmul issues
        scales = lucid.tensor([2.0, 3.0])
        f = lambda x: x * scales
        x = lucid.tensor([1.0, 1.0])
        Jrev = func.jacrev(f)(x)
        Jfwd = func.jacfwd(f)(x)
        assert lucid.allclose(Jrev, Jfwd, atol=1e-3)


# ── func.hessian ──────────────────────────────────────────────────────────────


class TestHessian:
    def test_quadratic_hessian(self) -> None:
        """H of sum(x²) = 2I."""
        f = lambda x: (x ** 2).sum()
        x = lucid.tensor([1.0, 2.0])
        H = func.hessian(f)(x)
        assert list(H.shape) == [2, 2]
        assert lucid.allclose(H, lucid.eye(2) * 2, atol=1e-3)


# ── func.linearize ────────────────────────────────────────────────────────────


class TestLinearize:
    def test_linearize_at_zero(self) -> None:
        """Linearize of exp(x) at x=0: lin_fn(v) ≈ J @ v = v."""
        f = lambda x: lucid.exp(x)
        x = lucid.zeros(3)
        primals_out, lin_fn = func.linearize(f, x)
        v = lucid.tensor([1.0, 0.0, 0.0])
        tangent = lin_fn(v)
        assert lucid.allclose(tangent, v, atol=1e-3)
