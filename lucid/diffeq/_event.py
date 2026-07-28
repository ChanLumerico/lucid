"""Integrating until a condition fires rather than to a fixed time.

Some problems do not end at a time you can name in advance — a trajectory ends
when it hits the ground, a simulation stops when a quantity crosses a
threshold.  ``event_fn(t, y)`` expresses that as a scalar whose sign change
marks the moment, and the solver runs until it flips.

Locating the moment precisely is what the interpolants built for dense output
are for: once a step brackets the crossing, the event time is found by
bisecting the polynomial that step already fitted, with no extra
right-hand-side evaluations along the way.
"""

import math
from typing import Callable, Sequence

import lucid
from lucid._tensor.tensor import Tensor
from lucid.diffeq import _adaptive, _fixed, _fused
from lucid.diffeq._tableau import ButcherTableau


__all__: list[str] = []


# Bisection stops once the bracket is this narrow.  Matches the reference
# implementation's default.
_EVENT_TOL = 1e-12

# A solve that never fires has to end somehow; this bounds it.
_MAX_EVENT_STEPS = 100_000


def _sign(value: float) -> float:
    """Sign of a float, with zero kept distinct from either side."""
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _evaluate(
    event_fn: Callable[[Tensor, Tensor], Tensor],
    scalar: Callable[[float], Tensor],
    t: float,
    y: Tensor,
) -> float:
    """Evaluate the event function and bring its scalar value to the host.

    Parameters
    ----------
    event_fn : callable
        ``g(t, y)`` whose sign change marks the event.
    scalar : callable
        Builds the 0-D time tensor the callback expects.
    t : float
        Time to evaluate at.
    y : Tensor
        State at that time.

    Returns
    -------
    float
        The event value.

    Raises
    ------
    ValueError
        If ``event_fn`` returns anything but a single-element tensor.
    """
    out = event_fn(scalar(t), y)
    if not isinstance(out, Tensor):
        raise TypeError(
            f"event_fn must return a Tensor, got {type(out).__name__}"
        )
    if out.numel() != 1:
        raise ValueError(
            f"event_fn must return a single-element tensor, got shape {out.shape}"
        )
    return float(out.item())


def find_event(
    interp: Callable[[float], Tensor],
    sign0: float,
    t0: float,
    t1: float,
    event_fn: Callable[[Tensor, Tensor], Tensor],
    scalar: Callable[[float], Tensor],
    tol: float = _EVENT_TOL,
) -> tuple[float, Tensor]:
    """Bisect a bracketing interval down to the event time.

    Parameters
    ----------
    interp : callable
        ``interp(t) -> y``, the step's polynomial.
    sign0 : float
        Sign of the event function at ``t0``.
    t0, t1 : float
        Ends of a step known to bracket the crossing.
    event_fn : callable
        The event function.
    scalar : callable
        Builds the 0-D time tensor the callback expects.
    tol : float, default=1e-12
        Width the bracket is narrowed to.

    Returns
    -------
    tuple
        The event time and the state there.

    Notes
    -----
    The iteration count is fixed up front from the bracket width, so this
    costs a predictable number of event-function calls and no
    right-hand-side evaluations at all — the state comes from the
    interpolant.
    """
    width = abs(t1 - t0)
    steps = 1 if width <= tol else int(math.ceil(math.log2(width / tol)))

    low, high = t0, t1
    for _ in range(steps):
        mid = (low + high) / 2.0
        if _sign(_evaluate(event_fn, scalar, mid, interp(mid))) == sign0:
            low = mid
        else:
            high = mid

    event_t = (low + high) / 2.0
    return event_t, interp(event_t)


def integrate_until_event(
    func: Callable[[Tensor, Tensor], Tensor],
    y0: Tensor,
    t0: float,
    event_fn: Callable[[Tensor, Tensor], Tensor],
    tableau: ButcherTableau,
    scalar: Callable[[float], Tensor],
    check: Callable[[object, int, int], Tensor],
    *,
    rtol: float,
    atol: float,
    options: dict[str, object] | None,
    direction: float,
) -> tuple[float, Tensor]:
    """Step from ``t0`` until the event function changes sign.

    Parameters
    ----------
    func : callable
        Right-hand side ``f(t, y)``.
    y0 : Tensor
        Initial state at ``t0``.
    t0 : float
        Start time.
    event_fn : callable
        ``g(t, y)``; the solve ends when its sign flips.
    tableau : ButcherTableau
        Adaptive or fixed; a fixed method needs ``options["step_size"]``
        since there is no end time to build a grid from.
    scalar, check : callable
        Time-tensor factory and right-hand-side validator.
    rtol, atol : float
        Tolerances, for an adaptive method.
    options : dict or None
        Solver options.
    direction : float
        ``+1`` to search forwards in time, ``-1`` backwards.

    Returns
    -------
    tuple
        The event time and the state there.

    Raises
    ------
    ValueError
        If a fixed-step method is used without ``step_size``.
    RuntimeError
        If no sign change is found within the step budget.

    Notes
    -----
    The event is bracketed by an accepted step and then located inside it,
    so accuracy comes from the interpolant rather than from making the
    steps small.
    """
    start_value = _evaluate(event_fn, scalar, t0, y0)
    sign0 = _sign(start_value)
    if sign0 == 0.0:
        return t0, y0

    if tableau.is_adaptive:
        opts = _adaptive.parse_options(options)
        stepper = _adaptive.Stepper(
            func, y0, t0, tableau, scalar, check, rtol, atol, opts, direction
        )
        for _ in range(_MAX_EVENT_STEPS):
            step = stepper.advance()
            if _sign(_evaluate(event_fn, scalar, step.t1, step.y1)) == sign0:
                continue
            coeffs = stepper.fit(step)

            def interp_adaptive(
                t: float, _c: list[Tensor] = coeffs, _s: _adaptive.Step = step
            ) -> Tensor:
                return _fused.poly_eval(_c, _s.t0, _s.t1, t)

            return find_event(
                interp_adaptive, sign0, step.t0, step.t1, event_fn, scalar
            )
    else:
        fixed_opts = _fixed.parse_options(options)
        if fixed_opts.step_size is None:
            raise ValueError(
                f"method {tableau.name!r} is a fixed-step method, so an event "
                f"solve needs options={{'step_size': ...}} — there is no end "
                f"time to build a grid from"
            )
        dt = direction * fixed_opts.step_size
        t_cur, y = t0, y0
        f0 = check(func(scalar(t0), y), 0, 0) if fixed_opts.interp == "cubic" else None

        for index in range(_MAX_EVENT_STEPS):
            t_next = t_cur + dt
            ks: list[Tensor] = []
            for stage in range(tableau.stages):
                stage_y = _fused.combine(y, ks, tableau.a[stage], dt)
                stage_t = scalar(t_cur + tableau.c[stage] * dt)
                ks.append(check(func(stage_t, stage_y), index, stage))
            y_next = _fused.combine(y, ks, tableau.b, dt)

            if _sign(_evaluate(event_fn, scalar, t_next, y_next)) != sign0:
                if fixed_opts.interp == "cubic":
                    assert f0 is not None
                    f1 = check(func(scalar(t_next), y_next), index, tableau.stages)
                    coeffs = _fixed._interp_fit_cubic(y, y_next, f0, f1, dt)
                else:
                    coeffs = _fixed._interp_fit_linear(y, y_next)

                def interp_fixed(
                    t: float,
                    _c: list[Tensor] = coeffs,
                    _a: float = t_cur,
                    _b: float = t_next,
                ) -> Tensor:
                    return _fused.poly_eval(_c, _a, _b, t)

                return find_event(
                    interp_fixed, sign0, t_cur, t_next, event_fn, scalar
                )

            if fixed_opts.interp == "cubic":
                f0 = check(func(scalar(t_next), y_next), index, tableau.stages)
            t_cur, y = t_next, y_next

    raise RuntimeError(
        f"event_fn did not change sign within {_MAX_EVENT_STEPS} steps from "
        f"t={t0!r}; check the event condition or the integration direction"
    )


def solution_pair(y0: Tensor, y_event: Tensor) -> Tensor:
    """Stack the initial and event states into the returned trajectory.

    Parameters
    ----------
    y0 : Tensor
        State the solve started from.
    y_event : Tensor
        State when the event fired.

    Returns
    -------
    Tensor
        Shape ``(2, *y0.shape)``.

    Notes
    -----
    An event solve has no output grid to report against — the only two
    times that mean anything are where it started and where it stopped.
    """
    states: Sequence[Tensor] = (y0, y_event)
    target = lucid.promote_types(y0.dtype, y_event.dtype)
    return lucid.stack(
        [s if s.dtype == target else s.to(target) for s in states], dim=0
    )
