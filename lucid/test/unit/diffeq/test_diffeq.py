"""``lucid.diffeq`` — Butcher tableaux and fixed-step Runge-Kutta integration."""

import math

import pytest

import lucid
import lucid.diffeq as diffeq
from lucid.diffeq._solvers import _combine
from lucid.diffeq._tableau import _METHODS

METHOD_ORDERS = [
    ("euler", 1),
    ("midpoint", 2),
    ("heun2", 2),
    ("heun3", 3),
    ("rk4", 4),
]


def _grid(n: int, t1: float = 1.0) -> list[float]:
    return [i * t1 / n for i in range(n + 1)]


def _decay(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
    """RHS of ``y' = -y``; solution ``y(t) = y0 * exp(-t)``."""
    return -y


class TestButcherTableau:
    def test_builtin_stage_counts(self) -> None:
        assert diffeq.EULER.stages == 1
        assert diffeq.MIDPOINT.stages == 2
        assert diffeq.HEUN2.stages == 2
        assert diffeq.HEUN3.stages == 3
        assert diffeq.RK4.stages == 4

    def test_builtin_orders(self) -> None:
        tableaux = (diffeq.EULER, diffeq.MIDPOINT, diffeq.HEUN2, diffeq.HEUN3)
        assert [t.order for t in tableaux] == [1, 2, 2, 3]
        assert diffeq.RK4.order == 4

    def test_registry_names_match_the_reference_library(self) -> None:
        # Method-name strings are exposed API, so they track the reference
        # ODE library exactly: it has heun2 / heun3 and no bare "heun".
        assert sorted(_METHODS) == [
            "adaptive_heun",
            "bosh3",
            "dopri5",
            "euler",
            "fehlberg2",
            "heun2",
            "heun3",
            "midpoint",
            "rk4",
            "tsit5",
        ]
        for name, tableau in _METHODS.items():
            assert tableau.name == name

    def test_heun3_coefficients(self) -> None:
        assert diffeq.HEUN3.a == ((), (1 / 3,), (0.0, 2 / 3))
        assert diffeq.HEUN3.b == (0.25, 0.0, 0.75)
        assert diffeq.HEUN3.c == (0.0, 1 / 3, 2 / 3)

    def test_sequences_are_coerced_to_tuples(self) -> None:
        tab = diffeq.ButcherTableau(
            a=[[], [1.0]], b=[0.5, 0.5], c=[0, 1], order=2, name="heun-ish"
        )
        assert tab.a == ((), (1.0,))
        assert tab.b == (0.5, 0.5)
        assert tab.c == (0.0, 1.0)
        # Frozen + fully tuple-ised means the tableau is hashable.
        assert isinstance(hash(tab), int)

    def test_is_frozen(self) -> None:
        with pytest.raises(Exception):
            diffeq.RK4.order = 5  # type: ignore[misc]

    def test_rejects_non_triangular_row(self) -> None:
        with pytest.raises(ValueError, match="strictly lower triangular"):
            diffeq.ButcherTableau(
                a=((0.0,), (1.0,)), b=(0.5, 0.5), c=(0.0, 1.0), order=2, name="bad"
            )

    def test_rejects_weights_not_summing_to_one(self) -> None:
        with pytest.raises(ValueError, match="must sum to 1"):
            diffeq.ButcherTableau(
                a=((), (1.0,)), b=(0.5, 0.4), c=(0.0, 1.0), order=2, name="bad"
            )

    def test_rejects_inconsistent_stage_time(self) -> None:
        with pytest.raises(ValueError, match="consistency violated"):
            diffeq.ButcherTableau(
                a=((), (1.0,)), b=(0.5, 0.5), c=(0.0, 0.5), order=2, name="bad"
            )

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="stage counts disagree"):
            diffeq.ButcherTableau(
                a=((), (1.0,)), b=(1.0,), c=(0.0,), order=1, name="bad"
            )

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one stage"):
            diffeq.ButcherTableau(a=(), b=(), c=(), order=1, name="bad")

    def test_rejects_bad_order_and_name(self) -> None:
        with pytest.raises(ValueError, match="order must be >= 1"):
            diffeq.ButcherTableau(a=((),), b=(1.0,), c=(0.0,), order=0, name="bad")
        with pytest.raises(ValueError, match="non-empty string"):
            diffeq.ButcherTableau(a=((),), b=(1.0,), c=(0.0,), order=1, name="")


class TestOrderOfConvergence:
    """Halving the step must shrink the global error by ``2 ** order``.

    This is the test that catches a mistyped tableau coefficient: a wrong
    entry usually still integrates, just at a lower order than advertised.
    Run in float64 — RK4 at these step counts is already near the float32
    round-off floor, which would swamp the measurement.
    """

    @pytest.mark.parametrize(("method", "order"), METHOD_ORDERS)
    def test_observed_order_matches_theory(self, method: str, order: int) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        errors = []
        for n in (10, 20, 40):
            y = diffeq.odeint(
                _decay, y0, _grid(n), method=method, return_trajectory=False
            )
            errors.append(abs(float(y.item()) - math.exp(-1.0)))

        observed = [math.log2(errors[i] / errors[i + 1]) for i in range(2)]
        for rate in observed:
            assert abs(rate - order) < 0.2, f"{method}: observed {observed}"


class TestAnalyticSolutions:
    def test_exponential_decay(self) -> None:
        y0 = lucid.tensor([2.0], dtype=lucid.float64)
        y = diffeq.odeint(_decay, y0, _grid(64), method="rk4", return_trajectory=False)
        assert abs(float(y.item()) - 2.0 * math.exp(-1.0)) < 1e-8

    def test_exponential_growth_matches_at_every_grid_point(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        grid = _grid(64, t1=2.0)
        traj = diffeq.odeint(lambda t, y: y, y0, grid)
        values = [row[0] for row in traj.tolist()]
        for t, v in zip(grid, values):
            assert abs(v - math.exp(t)) < 1e-6

    def test_harmonic_oscillator_conserves_energy(self) -> None:
        # y = (x, v), y' = (v, -x): a unit circle in phase space, so
        # x^2 + v^2 is invariant.
        def rhs(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
            return lucid.stack([y[1], -y[0]], dim=0)

        y0 = lucid.tensor([1.0, 0.0], dtype=lucid.float64)
        grid = [i * 2 * math.pi / 200 for i in range(201)]
        y = diffeq.odeint(rhs, y0, grid, method="rk4", return_trajectory=False)

        x_end, v_end = y.tolist()
        assert abs(x_end**2 + v_end**2 - 1.0) < 1e-8
        # One full period returns to the start.
        assert abs(x_end - 1.0) < 1e-6
        assert abs(v_end) < 1e-6

    def test_linear_system(self) -> None:
        # y' = A y with A = [[0, 1], [0, 0]] is nilpotent, so the exact
        # solution y(t) = (y0 + t * y1, y1) is a polynomial RK4 integrates
        # without truncation error.
        a = lucid.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=lucid.float64)

        def rhs(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
            return a @ y

        y0 = lucid.tensor([[1.0], [3.0]], dtype=lucid.float64)
        y = diffeq.odeint(rhs, y0, _grid(8, t1=2.0), return_trajectory=False)
        assert y.shape == (2, 1)
        (x_end,), (v_end,) = y.tolist()
        assert abs(x_end - 7.0) < 1e-10
        assert abs(v_end - 3.0) < 1e-10


class TestTrajectory:
    def test_trajectory_shape_and_endpoints(self) -> None:
        y0 = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        grid = _grid(5)
        traj = diffeq.odeint(_decay, y0, grid)

        assert traj.shape == (len(grid), 2, 2)
        # Index 0 is y0 verbatim.
        assert traj[0].tolist() == y0.tolist()

    def test_final_only_shape(self) -> None:
        y0 = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = diffeq.odeint(_decay, y0, _grid(5), return_trajectory=False)
        assert y.shape == y0.shape

    def test_final_state_agrees_with_trajectory_tail(self) -> None:
        y0 = lucid.tensor([1.0, -2.0], dtype=lucid.float64)
        grid = _grid(9)
        traj = diffeq.odeint(_decay, y0, grid)
        final = diffeq.odeint(_decay, y0, grid, return_trajectory=False)
        assert traj[-1].tolist() == pytest.approx(final.tolist())


class TestMethodResolution:
    @pytest.mark.parametrize("name", [m for m, _ in METHOD_ORDERS])
    def test_registered_names_run(self, name: str) -> None:
        y0 = lucid.tensor([1.0])
        y = diffeq.odeint(_decay, y0, _grid(4), method=name, return_trajectory=False)
        assert y.shape == (1,)

    def test_tableau_instance_accepted(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        by_name = diffeq.odeint(
            _decay, y0, _grid(8), method="rk4", return_trajectory=False
        )
        by_tableau = diffeq.odeint(
            _decay, y0, _grid(8), method=diffeq.RK4, return_trajectory=False
        )
        assert by_name.tolist() == pytest.approx(by_tableau.tolist())

    def test_custom_tableau_reproduces_heun2(self) -> None:
        custom = diffeq.ButcherTableau(
            a=((), (1.0,)), b=(0.5, 0.5), c=(0.0, 1.0), order=2, name="custom"
        )
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        got = diffeq.odeint(
            _decay, y0, _grid(8), method=custom, return_trajectory=False
        )
        want = diffeq.odeint(
            _decay, y0, _grid(8), method="heun2", return_trajectory=False
        )
        assert got.tolist() == pytest.approx(want.tolist())

    def test_unknown_method_lists_alternatives(self) -> None:
        y0 = lucid.tensor([1.0])
        with pytest.raises(ValueError, match="unknown method"):
            diffeq.odeint(_decay, y0, _grid(2), method="radauIIA5")

    def test_non_method_type_rejected(self) -> None:
        y0 = lucid.tensor([1.0])
        with pytest.raises(TypeError, match="str or ButcherTableau"):
            diffeq.odeint(_decay, y0, _grid(2), method=4)  # type: ignore[arg-type]


class TestGrid:
    def test_tensor_grid_matches_list_grid(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        grid = _grid(8)
        from_list = diffeq.odeint(_decay, y0, grid, return_trajectory=False)
        from_tensor = diffeq.odeint(
            _decay, y0, lucid.tensor(grid, dtype=lucid.float64), return_trajectory=False
        )
        assert from_list.tolist() == pytest.approx(from_tensor.tolist())

    def test_descending_grid_integrates_backwards(self) -> None:
        # Integrating y' = -y from t=1 back to t=0 recovers y0 * e^{+1}.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        grid = [1.0 - i / 32 for i in range(33)]
        y = diffeq.odeint(_decay, y0, grid, return_trajectory=False)
        assert abs(float(y.item()) - math.exp(1.0)) < 1e-7

    def test_non_uniform_spacing(self) -> None:
        # Spacing is read straight off the grid, so a stretched grid is a
        # legal (if less accurate) integration schedule.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        grid = [0.0, 0.1, 0.15, 0.5, 0.6, 1.0]
        y = diffeq.odeint(_decay, y0, grid, return_trajectory=False)
        assert abs(float(y.item()) - math.exp(-1.0)) < 1e-4

    def test_rejects_too_few_points(self) -> None:
        y0 = lucid.tensor([1.0])
        with pytest.raises(ValueError, match="at least 2 time points"):
            diffeq.odeint(_decay, y0, [0.0])

    def test_rejects_non_monotonic(self) -> None:
        y0 = lucid.tensor([1.0])
        with pytest.raises(ValueError, match="strictly monotonic"):
            diffeq.odeint(_decay, y0, [0.0, 0.5, 0.25, 1.0])

    def test_rejects_repeated_point(self) -> None:
        y0 = lucid.tensor([1.0])
        with pytest.raises(ValueError, match="strictly monotonic"):
            diffeq.odeint(_decay, y0, [0.0, 0.5, 0.5, 1.0])

    def test_rejects_multidimensional_tensor_grid(self) -> None:
        y0 = lucid.tensor([1.0])
        with pytest.raises(ValueError, match="1-D grid"):
            diffeq.odeint(_decay, y0, lucid.tensor([[0.0, 1.0]]))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_rejects_non_finite_point(self, bad: float) -> None:
        # NaN compares False against everything, so without an explicit
        # check it would pass the ordering test and surface only as an
        # all-NaN result many RHS evaluations later.
        y0 = lucid.tensor([1.0])
        with pytest.raises(ValueError, match="finite time points"):
            diffeq.odeint(_decay, y0, [0.0, bad, 1.0])


class TestInputValidation:
    def test_rejects_integer_state(self) -> None:
        y0 = lucid.tensor([1, 2])
        with pytest.raises(ValueError, match="floating dtype"):
            diffeq.odeint(_decay, y0, _grid(2))

    def test_rejects_rhs_shape_drift(self) -> None:
        y0 = lucid.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="func returned shape"):
            diffeq.odeint(lambda t, y: lucid.tensor([1.0]), y0, _grid(2))

    def test_rejects_rhs_device_drift(self, device_gpu_only: str) -> None:
        y0 = lucid.tensor([1.0])
        with pytest.raises(ValueError, match="func returned a tensor on"):
            diffeq.odeint(lambda t, y: (-y).to(device_gpu_only), y0, _grid(2))

    def test_rejects_non_tensor_rhs(self) -> None:
        y0 = lucid.tensor([1.0])
        with pytest.raises(TypeError, match="func must return a Tensor"):
            diffeq.odeint(lambda t, y: 1.0, y0, _grid(2))  # type: ignore[arg-type,return-value]


class TestDtypeAndDevice:
    @pytest.mark.parametrize("dt", [lucid.float32, lucid.float64])
    def test_dtype_is_preserved(self, dt: lucid.dtype) -> None:
        y0 = lucid.tensor([1.0], dtype=dt)
        y = diffeq.odeint(_decay, y0, _grid(4), return_trajectory=False)
        assert y.dtype == dt

    def test_stage_time_matches_state_dtype_and_device(self) -> None:
        seen: list[tuple[object, object, tuple[int, ...]]] = []

        def rhs(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
            seen.append((t.dtype, t.device, t.shape))
            return -y

        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        diffeq.odeint(rhs, y0, _grid(2), method="rk4")
        assert len(seen) == 2 * 4  # 2 steps x 4 stages
        for dtype, device, shape in seen:
            assert dtype == y0.dtype
            assert device == y0.device
            assert shape == ()

    def test_stage_times_follow_the_tableau(self) -> None:
        times: list[float] = []

        def rhs(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
            times.append(float(t.item()))
            return -y

        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        diffeq.odeint(rhs, y0, [0.0, 1.0], method="rk4")
        assert times == pytest.approx([0.0, 0.5, 0.5, 1.0])

    def test_mixed_dtype_rhs_is_promoted(self) -> None:
        # The engine op is strict about dtype, so the solver has to promote
        # exactly as ``Tensor.__add__`` does.  A float16 RHS over a float32
        # state is the autocast case, not a contrived one.
        y0 = lucid.tensor([1.0], dtype=lucid.float32)
        y = diffeq.odeint(
            lambda t, yy: (-yy).to(lucid.float16),
            y0,
            _grid(4),
            return_trajectory=False,
        )
        assert y.dtype == lucid.float32

    def test_trajectory_is_uniform_after_promotion(self) -> None:
        # A float64 RHS lifts the state above y0's dtype; trajectory[0] is
        # still y0, so stacking has to reconcile them.
        y0 = lucid.tensor([1.0], dtype=lucid.float32)
        traj = diffeq.odeint(lambda t, yy: (-yy).to(lucid.float64), y0, _grid(4))
        assert traj.dtype == lucid.float64
        assert traj.shape == (5, 1)

    def test_autocast_on_metal(self, device_gpu_only: str) -> None:
        import lucid.nn as nn

        net = nn.Linear(3, 3).to(device_gpu_only)
        y0 = lucid.ones(1, 3, device=device_gpu_only)
        with lucid.amp.autocast(device_type=device_gpu_only, dtype=lucid.float16):
            y = diffeq.odeint(
                lambda t, yy: net(yy), y0, _grid(4), return_trajectory=False
            )
        assert y.shape == y0.shape
        assert y.dtype == y0.dtype

    def test_runs_on_metal(self, device_gpu_only: str) -> None:
        y0 = lucid.tensor([1.0], device=device_gpu_only)
        y = diffeq.odeint(_decay, y0, _grid(8), return_trajectory=False)
        assert str(y.device) == str(y0.device)
        assert abs(float(y.item()) - math.exp(-1.0)) < 1e-5


class TestGradientFlow:
    def test_grad_through_solver(self) -> None:
        # y' = -k y integrated to t=1 gives y(1) = y0 * exp(-k), so
        # dy/dy0 = exp(-k) exactly.  This only holds if the fused op's
        # backward is wired — a fused forward alone would cut the graph.
        k = 0.5
        y0 = lucid.tensor([1.0], dtype=lucid.float64, requires_grad=True)
        y = diffeq.odeint(
            lambda t, y: -k * y, y0, _grid(64), method="rk4", return_trajectory=False
        )
        y.sum().backward()

        assert y0.grad is not None
        assert float(y0.grad.item()) == pytest.approx(math.exp(-k), abs=1e-8)

    def test_grad_wrt_rhs_parameter(self) -> None:
        # d/dk of y0 * exp(-k) at k=0.5 is -exp(-0.5).
        k = lucid.tensor([0.5], dtype=lucid.float64, requires_grad=True)
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        y = diffeq.odeint(
            lambda t, y: -k * y, y0, _grid(64), method="rk4", return_trajectory=False
        )
        (grad_k,) = lucid.autograd.grad(y.sum(), k)

        assert grad_k is not None
        assert float(grad_k.item()) == pytest.approx(-math.exp(-0.5), abs=1e-8)

    def test_grad_through_trajectory(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64, requires_grad=True)
        traj = diffeq.odeint(_decay, y0, _grid(32))
        traj.sum().backward()

        # Every grid point contributes exp(-t_i), including t_0 = 0.
        expected = sum(math.exp(-t) for t in _grid(32))
        assert y0.grad is not None
        assert float(y0.grad.item()) == pytest.approx(expected, abs=1e-6)

    def test_second_order_grad_through_solver(self) -> None:
        # y(1) = y0 * exp(-k), so d2/(dy0 dk) = -exp(-k).  The fused op has
        # to opt into graph-recording backward for this to work at all —
        # the unfused ``y + dt * c * k`` spelling supports it, so the
        # fusion must not silently cost the caller second-order gradients.
        y0 = lucid.tensor([1.0], dtype=lucid.float64, requires_grad=True)
        k = lucid.tensor([0.5], dtype=lucid.float64, requires_grad=True)
        out = diffeq.odeint(lambda t, y: -k * y, y0, _grid(8), return_trajectory=False)

        (g_y0,) = lucid.autograd.grad(out.sum(), y0, create_graph=True)
        assert g_y0 is not None
        (mixed,) = lucid.autograd.grad(g_y0.sum(), k)

        assert mixed is not None
        assert float(mixed.item()) == pytest.approx(-math.exp(-0.5), abs=1e-6)

    @pytest.mark.parametrize("method", [m for m, _ in METHOD_ORDERS])
    def test_gradcheck_combine(self, method: str) -> None:
        # Finite-difference check of the fused op's backward, for every
        # tableau's final-combination weights.
        tableau = _METHODS[method]
        ks = [
            lucid.tensor([0.5 + i, -1.25 - i], dtype=lucid.float64, requires_grad=True)
            for i in range(tableau.stages)
        ]
        y0 = lucid.tensor([1.0, -2.0], dtype=lucid.float64, requires_grad=True)

        def fn(*args: lucid.Tensor) -> lucid.Tensor:
            # Squared so the gradient varies with the inputs — a linear
            # objective would pass on the scale factors alone.
            out = _combine(args[0], list(args[1:]), tableau.b, 0.3)
            return (out * out).sum()

        assert lucid.autograd.gradcheck(fn, [y0, *ks])


ADAPTIVE_METHODS = ["dopri5", "tsit5", "bosh3", "fehlberg2", "adaptive_heun"]


class TestAdaptiveTableaux:
    @pytest.mark.parametrize("name", ADAPTIVE_METHODS)
    def test_carries_error_and_mid_weights(self, name: str) -> None:
        tab = _METHODS[name]
        assert tab.is_adaptive
        assert tab.b_error is not None and len(tab.b_error) == tab.stages
        assert tab.mid is not None and len(tab.mid) == tab.stages

    @pytest.mark.parametrize("name", ADAPTIVE_METHODS)
    def test_error_weights_sum_to_zero(self, name: str) -> None:
        # Both embedded solutions are consistent, so their weights differ by
        # something that sums to zero.  A transcription slip in the tableau
        # almost always breaks this before it breaks anything else.
        tab = _METHODS[name]
        assert tab.b_error is not None
        assert sum(tab.b_error) == pytest.approx(0.0, abs=1e-12)

    def test_fsal_detection(self) -> None:
        # dopri5 and bosh3 end on a stage that already evaluated the new
        # state, so that derivative is reused as the next step's first stage.
        assert diffeq.DOPRI5.is_fsal
        assert diffeq.BOSH3.is_fsal
        assert not diffeq.ADAPTIVE_HEUN.is_fsal
        assert not diffeq.FEHLBERG2.is_fsal
        # tsit5 ends with a non-zero final weight, so the last stage is
        # not the new state's derivative — the reference library applies
        # the same test and reaches the same conclusion.
        assert not diffeq.TSIT5.is_fsal
        assert not diffeq.RK4.is_adaptive

    def test_rejects_error_weights_that_do_not_cancel(self) -> None:
        with pytest.raises(ValueError, match="b_error must sum to 0"):
            diffeq.ButcherTableau(
                a=((), (1.0,)),
                b=(0.5, 0.5),
                c=(0.0, 1.0),
                order=2,
                name="bad",
                b_error=(0.5, 0.5),
                mid=(0.5, 0.0),
            )

    def test_error_weights_require_mid(self) -> None:
        with pytest.raises(ValueError, match="also needs mid"):
            diffeq.ButcherTableau(
                a=((), (1.0,)),
                b=(0.5, 0.5),
                c=(0.0, 1.0),
                order=2,
                name="bad",
                b_error=(0.5, -0.5),
            )


class TestAdaptiveIntegration:
    @pytest.mark.parametrize("name", ADAPTIVE_METHODS)
    def test_reaches_analytic_solution(self, name: str) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        y = diffeq.odeint(
            _decay,
            y0,
            [0.0, 1.0],
            method=name,
            rtol=1e-10,
            atol=1e-12,
            return_trajectory=False,
        )
        assert float(y.item()) == pytest.approx(math.exp(-1.0), abs=1e-8)

    def test_default_method_is_dopri5(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        implicit = diffeq.odeint(_decay, y0, [0.0, 1.0], return_trajectory=False)
        explicit = diffeq.odeint(
            _decay, y0, [0.0, 1.0], method="dopri5", return_trajectory=False
        )
        assert implicit.tolist() == explicit.tolist()

    def test_tightening_tolerance_reduces_error(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        errors = []
        for tol in (1e-3, 1e-6, 1e-9):
            y = diffeq.odeint(
                _decay, y0, [0.0, 1.0], rtol=tol, atol=tol, return_trajectory=False
            )
            errors.append(abs(float(y.item()) - math.exp(-1.0)))
        assert errors[0] > errors[1] > errors[2]

    def test_output_grid_does_not_change_the_answer(self) -> None:
        # The defining property of an adaptive solve: t is a set of output
        # times, so refining it must not change the value at a shared time.
        # A fixed-step solver would answer differently for each grid.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        coarse = diffeq.odeint(_decay, y0, [0.0, 1.0])
        fine = diffeq.odeint(_decay, y0, _grid(37))
        assert coarse[-1].tolist() == pytest.approx(fine[-1].tolist(), abs=1e-9)

    def test_dense_output_matches_analytic_everywhere(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        grid = _grid(8)
        traj = diffeq.odeint(_decay, y0, grid, rtol=1e-10, atol=1e-12)
        assert traj.shape == (len(grid), 1)
        for t, row in zip(grid, traj.tolist()):
            assert row[0] == pytest.approx(math.exp(-t), abs=1e-9)

    def test_descending_grid(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        y = diffeq.odeint(
            _decay, y0, [1.0, 0.0], rtol=1e-10, atol=1e-12, return_trajectory=False
        )
        assert float(y.item()) == pytest.approx(math.exp(1.0), abs=1e-7)

    def test_harmonic_oscillator_period(self) -> None:
        def rhs(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
            return lucid.stack([y[1], -y[0]], dim=0)

        y0 = lucid.tensor([1.0, 0.0], dtype=lucid.float64)
        y = diffeq.odeint(
            rhs,
            y0,
            [0.0, 2 * math.pi],
            rtol=1e-11,
            atol=1e-13,
            return_trajectory=False,
        )
        x_end, v_end = y.tolist()
        assert x_end == pytest.approx(1.0, abs=1e-7)
        assert v_end == pytest.approx(0.0, abs=1e-7)

    def test_grad_flows_through_adaptive_solve(self) -> None:
        k = 0.5
        y0 = lucid.tensor([1.0], dtype=lucid.float64, requires_grad=True)
        y = diffeq.odeint(
            lambda t, y: -k * y,
            y0,
            [0.0, 1.0],
            rtol=1e-11,
            atol=1e-13,
            return_trajectory=False,
        )
        y.sum().backward()
        assert y0.grad is not None
        assert float(y0.grad.item()) == pytest.approx(math.exp(-k), abs=1e-7)

    def test_mixed_dtype_rhs_is_promoted(self) -> None:
        # Under autocast the derivatives come back below the state's
        # precision; the interpolant is built from derivatives alone in one
        # of its coefficients, so it has to promote as a group.
        y0 = lucid.tensor([1.0], dtype=lucid.float32)
        traj = diffeq.odeint(lambda t, y: (-y).to(lucid.float16), y0, _grid(4))
        assert traj.dtype == lucid.float32
        assert traj.shape == (5, 1)


class TestAdaptiveOptions:
    def test_first_step_and_max_step_are_honoured(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        y = diffeq.odeint(
            _decay,
            y0,
            [0.0, 1.0],
            options={"first_step": 0.01, "max_step": 0.05},
            return_trajectory=False,
        )
        assert float(y.item()) == pytest.approx(math.exp(-1.0), abs=1e-8)

    def test_step_t_is_landed_on(self) -> None:
        seen: list[float] = []

        def rhs(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
            seen.append(float(t.item()))
            return -y

        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        diffeq.odeint(
            rhs,
            y0,
            [0.0, 1.0],
            options={"step_t": [0.3, 0.7]},
            return_trajectory=False,
        )
        assert any(abs(t - 0.3) < 1e-12 for t in seen)
        assert any(abs(t - 0.7) < 1e-12 for t in seen)

    def test_max_num_steps_is_enforced(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(RuntimeError, match="max_num_steps"):
            diffeq.odeint(
                _decay,
                y0,
                [0.0, 1.0],
                rtol=1e-13,
                atol=1e-15,
                options={"max_num_steps": 2},
            )

    def test_rejects_unknown_option(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(ValueError, match="unknown option"):
            diffeq.odeint(_decay, y0, [0.0, 1.0], options={"nope": 1})

    @pytest.mark.parametrize(
        ("bad", "match"),
        [
            ({"ifactor": 0.5}, "ifactor must be"),
            ({"dfactor": 0.0}, "dfactor must lie"),
            ({"max_num_steps": 0}, "max_num_steps must be"),
            ({"min_step": 1.0, "max_step": 0.1}, "must not exceed"),
            ({"first_step": 0.0}, "first_step must be non-zero"),
        ],
    )
    def test_rejects_out_of_range_options(
        self, bad: dict[str, float], match: str
    ) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(ValueError, match=match):
            diffeq.odeint(_decay, y0, [0.0, 1.0], options=bad)

    def test_dtype_and_norm_are_accepted_and_ignored(self) -> None:
        # Both exist upstream to control step-control precision and the error
        # norm; here step control is always host-double and the norm is fused
        # into the kernel, so they are accepted for signature compatibility.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        y = diffeq.odeint(
            _decay,
            y0,
            [0.0, 1.0],
            options={"dtype": lucid.float64, "norm": "rms"},
            return_trajectory=False,
        )
        assert float(y.item()) == pytest.approx(math.exp(-1.0), abs=1e-7)


class TestFixedStepOptions:
    def test_step_size_decouples_the_integration_grid(self) -> None:
        # With step_size the solver walks its own grid and interpolates to t,
        # so a two-point t is as accurate as a fine one.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        y = diffeq.odeint(
            _decay, y0, [0.0, 1.0], method="rk4",
            options={"step_size": 1 / 64, "interp": "cubic"},
            return_trajectory=False,
        )
        assert float(y.item()) == pytest.approx(math.exp(-1.0), abs=1e-9)

    def test_cubic_interpolation_beats_linear_off_grid(self) -> None:
        # Linear interpolation caps accuracy at O(h^2) no matter how good the
        # stepper is; the cubic Hermite recovers the method's own order.  The
        # gap is the whole reason interp is an option.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        grid = [0.0, 0.3, 1.0]

        def worst(interp: str) -> float:
            traj = diffeq.odeint(
                _decay, y0, grid, method="rk4",
                options={"step_size": 1 / 64, "interp": interp},
            )
            return max(
                abs(row[0] - math.exp(-t)) for t, row in zip(grid, traj.tolist())
            )

        assert worst("cubic") < worst("linear") / 100

    def test_grid_constructor_overrides_step_size(self) -> None:
        seen: list[object] = []

        def build(func: object, y0_arg: object, t: object) -> list[float]:
            seen.append(t)
            return [i / 32 for i in range(33)]

        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        y = diffeq.odeint(
            _decay, y0, [0.0, 1.0], method="rk4",
            options={"grid_constructor": build, "step_size": 0.5, "interp": "cubic"},
            return_trajectory=False,
        )
        assert seen  # the constructor really was consulted
        assert float(y.item()) == pytest.approx(math.exp(-1.0), abs=1e-8)

    def test_default_grid_is_the_output_grid(self) -> None:
        # No options at all: t is the integration grid, exactly as before.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        grid = _grid(64)
        implicit = diffeq.odeint(_decay, y0, grid, method="rk4", return_trajectory=False)
        explicit = diffeq.odeint(
            _decay, y0, grid, method="rk4", options={}, return_trajectory=False
        )
        assert implicit.tolist() == explicit.tolist()

    def test_perturb_only_nudges(self) -> None:
        # A one-ulp shift must not move the answer meaningfully; it exists so a
        # discontinuity sitting on a grid point is sampled from one side.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        plain = diffeq.odeint(
            _decay, y0, [0.0, 1.0], method="rk4",
            options={"step_size": 1 / 64}, return_trajectory=False,
        )
        nudged = diffeq.odeint(
            _decay, y0, [0.0, 1.0], method="rk4",
            options={"step_size": 1 / 64, "perturb": True}, return_trajectory=False,
        )
        assert float(nudged.item()) == pytest.approx(float(plain.item()), abs=1e-12)

    def test_trajectory_shape_with_step_size(self) -> None:
        y0 = lucid.tensor([[1.0, 2.0]], dtype=lucid.float64)
        grid = [0.0, 0.25, 0.5, 1.0]
        traj = diffeq.odeint(
            _decay, y0, grid, method="rk4", options={"step_size": 1 / 16}
        )
        assert traj.shape == (len(grid), 1, 2)
        assert traj[0].tolist() == y0.tolist()

    def test_rejects_adaptive_options_on_a_fixed_method(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(ValueError, match="unknown option"):
            diffeq.odeint(
                _decay, y0, _grid(4), method="rk4", options={"first_step": 0.1}
            )

    def test_rejects_fixed_options_on_an_adaptive_method(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(ValueError, match="unknown option"):
            diffeq.odeint(_decay, y0, [0.0, 1.0], options={"step_size": 0.1})

    @pytest.mark.parametrize(
        ("bad", "match"),
        [
            ({"step_size": 0.0}, "finite and non-zero"),
            ({"interp": "quartic"}, "interp must be one of"),
            ({"perturb": 1}, "must be a bool"),
            ({"grid_constructor": 5}, "must be callable"),
        ],
    )
    def test_rejects_out_of_range_options(
        self, bad: dict[str, object], match: str
    ) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(ValueError, match=match):
            diffeq.odeint(_decay, y0, _grid(4), method="rk4", options=bad)

    def test_grid_constructor_must_span_the_interval(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(ValueError, match="must return a grid spanning"):
            diffeq.odeint(
                _decay, y0, [0.0, 1.0], method="rk4",
                options={"grid_constructor": lambda f, y, t: [0.0, 0.5]},
            )
