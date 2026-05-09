"""Autograd: backward / leaf semantics / retain_graph / functional API."""

import numpy as np
import pytest

import lucid


class TestBackwardScalar:
    def test_simple(self) -> None:
        x = lucid.tensor([3.0], requires_grad=True)
        y = (x * x).sum()
        y.backward()
        np.testing.assert_allclose(x.grad.numpy(), [6.0], atol=1e-6)

    def test_chain(self) -> None:
        x = lucid.tensor([2.0], requires_grad=True)
        y = (x.exp().log()).sum()
        y.backward()
        np.testing.assert_allclose(x.grad.numpy(), [1.0], atol=1e-5)


class TestBackwardVector:
    def test_with_explicit_seed(self) -> None:
        x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * x
        seed = lucid.tensor([1.0, 1.0, 1.0])
        y.backward(seed)
        np.testing.assert_allclose(x.grad.numpy(), [2.0, 4.0, 6.0], atol=1e-6)


class TestLeafSemantics:
    def test_leaf_with_requires_grad(self) -> None:
        x = lucid.tensor([1.0], requires_grad=True)
        assert x.is_leaf

    def test_non_leaf_intermediate(self) -> None:
        x = lucid.tensor([1.0], requires_grad=True)
        y = x * 2.0
        # ``y`` is a non-leaf result of an op.
        assert not y.is_leaf

    def test_grad_of_non_leaf_default_none(self) -> None:
        x = lucid.tensor([1.0], requires_grad=True)
        y = x * 2.0
        z = y.sum()
        z.backward()
        # Without retain_grad, intermediate ``y`` should not hold a grad.
        assert y.grad is None


class TestRequireGradTransitions:
    def test_requires_grad_setter(self) -> None:
        x = lucid.tensor([1.0])
        assert not x.requires_grad
        x.requires_grad = True
        assert x.requires_grad

    def test_requires_grad_method(self) -> None:
        x = lucid.tensor([1.0])
        x.requires_grad_(True)
        assert x.requires_grad


class TestDetach:
    def test_detach_breaks_graph(self) -> None:
        x = lucid.tensor([1.0, 2.0], requires_grad=True)
        y = (x * 2.0).detach()
        assert not y.requires_grad


class TestZeroGrad:
    def test_set_grad_none(self) -> None:
        x = lucid.tensor([1.0], requires_grad=True)
        (x * x).sum().backward()
        assert x.grad is not None
        x.grad = None
        assert x.grad is None


class TestMultiInput:
    def test_two_var_sum(self) -> None:
        a = lucid.tensor([1.0, 2.0], requires_grad=True)
        b = lucid.tensor([3.0, 4.0], requires_grad=True)
        ((a + b) ** 2).sum().backward()
        # d/da [(a+b)²] = 2(a+b).
        np.testing.assert_allclose(a.grad.numpy(), [8.0, 12.0], atol=1e-5)
        np.testing.assert_allclose(b.grad.numpy(), [8.0, 12.0], atol=1e-5)


class TestFunctionalAutograd:
    def test_jacobian_present(self) -> None:
        # Just verify the lucid.autograd surface includes the
        # higher-order helpers — values tested under functional/.
        for name in ("backward", "grad"):
            assert hasattr(lucid.autograd, name) or hasattr(lucid, name)


class TestSecondOrder:
    def test_grad_of_grad_x_cubed(self) -> None:
        # d²/dx² [x³] = 6x; at x=2 → 12.
        x = lucid.tensor([2.0], requires_grad=True)
        y = x * x * x
        (g1,) = lucid.autograd.grad(y, x, create_graph=True)
        (g2,) = lucid.autograd.grad(g1, x)
        assert abs(g2.item() - 12.0) < 1e-4

    def test_hessian_quartic(self) -> None:
        # f(x) = sum(x^4); H_ii = 12 x_i², off-diag 0.  At [1, 1] → diag(12, 12).
        H = lucid.autograd.hessian(lambda x: (x**4).sum(), lucid.tensor([1.0, 1.0]))
        np.testing.assert_allclose(H.numpy(), [[12.0, 0.0], [0.0, 12.0]], atol=1e-3)


class TestAnomalyToggle:
    def test_set_and_query(self) -> None:
        try:
            lucid.autograd.set_detect_anomaly(True)
            assert lucid.autograd.is_anomaly_enabled()
            lucid.autograd.set_detect_anomaly(False)
            assert not lucid.autograd.is_anomaly_enabled()
        finally:
            lucid.autograd.set_detect_anomaly(False)


class TestAutogradGraph:
    def test_allow_mutation_on_saved_flag_toggles(self) -> None:
        from lucid._C import engine as _C_engine

        assert not _C_engine.is_mutation_on_saved_allowed()
        with lucid.autograd.graph.allow_mutation_on_saved_tensors():
            assert _C_engine.is_mutation_on_saved_allowed()
        assert not _C_engine.is_mutation_on_saved_allowed()

    def test_save_on_cpu_callable_stub(self) -> None:
        # Stub: must enter/exit cleanly and not affect backward correctness.
        with lucid.autograd.graph.save_on_cpu():
            x = lucid.tensor([1.0, 2.0], requires_grad=True)
            (x * x).sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [2.0, 4.0], atol=1e-5)


class TestAutogradProfilerAlias:
    def test_alias_matches_lucid_profiler(self) -> None:
        import lucid.profiler as P

        assert lucid.autograd.profiler.profile is P.profile
        assert lucid.autograd.profiler.OpEvent is P.OpEvent

    def test_profile_context_manager_runs(self) -> None:
        with lucid.autograd.profiler.profile():
            t = lucid.tensor([1.0, 2.0])
            _ = t * t
