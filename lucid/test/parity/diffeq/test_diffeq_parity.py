"""Reference parity for ``lucid.diffeq`` fixed-step Runge-Kutta integration.

Two independent references are used.  The first is a hand-written RK loop
built from reference-framework tensor ops — it pins Lucid's fused
``rk_combine`` against the same arithmetic done the unfused way, which is
exactly what the fusion is supposed to be equivalent to.  The second is
``torchdiffeq`` when it happens to be installed, which pins the tableau
coefficients themselves against the established implementation.
"""

from typing import Any, Callable, Sequence

import numpy as np
import pytest

import lucid
import lucid.diffeq as diffeq
from lucid.diffeq._tableau import _METHODS
from lucid.test._helpers.compare import assert_close

METHODS = ["euler", "midpoint", "heun2", "heun3", "rk4"]


def _grid(n: int, t1: float = 1.0) -> list[float]:
    return [i * t1 / n for i in range(n + 1)]


def _ref_odeint(
    ref: Any,
    func: Callable[[Any, Any], Any],
    y0: Any,
    t: Sequence[float],
    method: str,
) -> Any:
    """Unfused reference implementation of the same fixed-step RK loop."""
    tableau = _METHODS[method]
    y = y0
    trajectory = [y0]
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        ks: list[Any] = []
        for stage in range(tableau.stages):
            stage_y = y
            for j, coeff in enumerate(tableau.a[stage]):
                if coeff:
                    stage_y = stage_y + dt * coeff * ks[j]
            stage_t = ref.tensor(t[i] + tableau.c[stage] * dt, dtype=y0.dtype)
            ks.append(func(stage_t, stage_y))
        for j, coeff in enumerate(tableau.b):
            if coeff:
                y = y + dt * coeff * ks[j]
        trajectory.append(y)
    return ref.stack(trajectory, dim=0)


@pytest.mark.parity
class TestOdeintParity:
    @pytest.fixture
    def y0_pair(self, ref: Any) -> tuple[lucid.Tensor, Any]:
        rng = np.random.default_rng(0)
        y0 = rng.standard_normal(size=(3, 2)).astype(np.float64)
        return (
            lucid.tensor(y0.copy(), dtype=lucid.float64),
            ref.tensor(y0.copy(), dtype=ref.float64),
        )

    @pytest.mark.parametrize("method", METHODS)
    def test_trajectory_matches_unfused_loop(
        self, method: str, y0_pair: tuple[lucid.Tensor, Any], ref: Any
    ) -> None:
        # A time-dependent, non-linear RHS so every tableau entry matters.
        grid = _grid(16)
        lucid_traj = diffeq.odeint(
            lambda t, y: -y + lucid.sin(t), y0_pair[0], grid, method=method
        )
        ref_traj = _ref_odeint(
            ref, lambda t, y: -y + ref.sin(t), y0_pair[1], grid, method
        )
        assert_close(lucid_traj, ref_traj, atol=1e-10, rtol=1e-9)

    @pytest.mark.parametrize("method", METHODS)
    def test_final_state_matches_unfused_loop(
        self, method: str, y0_pair: tuple[lucid.Tensor, Any], ref: Any
    ) -> None:
        grid = _grid(16)
        lucid_final = diffeq.odeint(
            lambda t, y: y * y - t,
            y0_pair[0],
            grid,
            method=method,
            return_trajectory=False,
        )
        ref_final = _ref_odeint(ref, lambda t, y: y * y - t, y0_pair[1], grid, method)[
            -1
        ]
        assert_close(lucid_final, ref_final, atol=1e-10, rtol=1e-9)

    @pytest.mark.parametrize("method", METHODS)
    def test_descending_grid_matches_unfused_loop(
        self, method: str, y0_pair: tuple[lucid.Tensor, Any], ref: Any
    ) -> None:
        grid = [1.0 - i / 16 for i in range(17)]
        lucid_traj = diffeq.odeint(lambda t, y: -y, y0_pair[0], grid, method=method)
        ref_traj = _ref_odeint(ref, lambda t, y: -y, y0_pair[1], grid, method)
        assert_close(lucid_traj, ref_traj, atol=1e-10, rtol=1e-9)

    @pytest.mark.parametrize("method", METHODS)
    def test_gradient_matches_unfused_loop(self, method: str, ref: Any) -> None:
        rng = np.random.default_rng(1)
        raw = rng.standard_normal(size=(4,)).astype(np.float64)

        lucid_y0 = lucid.tensor(raw.copy(), dtype=lucid.float64, requires_grad=True)
        ref_y0 = ref.tensor(raw.copy(), dtype=ref.float64, requires_grad=True)
        grid = _grid(12)

        lucid_out = diffeq.odeint(
            lambda t, y: -y + lucid.sin(t),
            lucid_y0,
            grid,
            method=method,
            return_trajectory=False,
        )
        (lucid_out * lucid_out).sum().backward()

        ref_out = _ref_odeint(ref, lambda t, y: -y + ref.sin(t), ref_y0, grid, method)[
            -1
        ]
        (ref_out * ref_out).sum().backward()

        assert lucid_y0.grad is not None
        assert_close(lucid_y0.grad, ref_y0.grad, atol=1e-10, rtol=1e-9)

    @pytest.mark.parametrize("method", METHODS)
    def test_second_order_matches_unfused_loop(self, method: str, ref: Any) -> None:
        # Double backward through the solver.  The reference loop is built
        # from plain arithmetic, so this pins the fused op's graph-recording
        # backward against the unfused chain it stands in for.
        grid = _grid(8)

        lucid_y0 = lucid.tensor([1.0], dtype=lucid.float64, requires_grad=True)
        lucid_k = lucid.tensor([0.5], dtype=lucid.float64, requires_grad=True)
        lucid_out = diffeq.odeint(
            lambda t, y: -lucid_k * y,
            lucid_y0,
            grid,
            method=method,
            return_trajectory=False,
        )
        (lucid_g,) = lucid.autograd.grad(lucid_out.sum(), lucid_y0, create_graph=True)
        (lucid_mixed,) = lucid.autograd.grad(lucid_g.sum(), lucid_k)

        ref_y0 = ref.tensor([1.0], dtype=ref.float64, requires_grad=True)
        ref_k = ref.tensor([0.5], dtype=ref.float64, requires_grad=True)
        ref_out = _ref_odeint(ref, lambda t, y: -ref_k * y, ref_y0, grid, method)[-1]
        (ref_g,) = ref.autograd.grad(ref_out.sum(), ref_y0, create_graph=True)
        (ref_mixed,) = ref.autograd.grad(ref_g.sum(), ref_k)

        assert lucid_mixed is not None
        assert_close(lucid_mixed, ref_mixed, atol=1e-10, rtol=1e-9)


@pytest.mark.parity
class TestTorchdiffeqParity:
    """Cross-check the tableau coefficients against the established library.

    Skipped unless ``torchdiffeq`` is installed — it is not a Lucid test
    dependency, only an opportunistic second opinion.
    """

    @pytest.mark.parametrize("method", ["euler", "midpoint", "rk4"])
    def test_matches_torchdiffeq(self, method: str, ref: Any) -> None:
        torchdiffeq = pytest.importorskip("torchdiffeq")

        rng = np.random.default_rng(2)
        raw = rng.standard_normal(size=(5,)).astype(np.float64)
        n = 16
        grid = _grid(n)

        lucid_out = diffeq.odeint(
            lambda t, y: -y + lucid.sin(t),
            lucid.tensor(raw.copy(), dtype=lucid.float64),
            grid,
            method=method,
        )
        ref_out = torchdiffeq.odeint(
            lambda t, y: -y + ref.sin(t),
            ref.tensor(raw.copy(), dtype=ref.float64),
            ref.tensor(grid, dtype=ref.float64),
            method=method,
            options={"step_size": 1.0 / n},
        )
        assert_close(lucid_out, ref_out, atol=1e-10, rtol=1e-9)
