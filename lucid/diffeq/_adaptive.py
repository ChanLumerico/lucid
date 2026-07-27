"""Adaptive-step explicit Runge-Kutta integration with dense output.

An adaptive method pairs two embedded solutions of different order.  Their
difference estimates the local error of the step, which a controller turns
into a step size: accept and grow when the estimate is comfortably inside
tolerance, shrink and retry when it is not.  The user's ``t`` grid stops being
the integration grid and becomes a set of *output* times, reached by
interpolating inside whichever step happens to span each one.

This costs one host synchronisation per step — the controller must branch on
the error ratio, so that scalar has to cross to the host. It is the
unavoidable price of adaptivity, and the reason this loop has a different
performance character from the fixed-grid one: the fused error-norm op exists
so the price stays at exactly one sync rather than several.
"""

import math
from dataclasses import dataclass
from typing import Callable, Sequence, SupportsFloat, cast

from lucid._tensor.tensor import Tensor
from lucid.diffeq import _fused
from lucid.diffeq._tableau import ButcherTableau

__all__: list[str] = []


# Upstream's controller defaults, kept identical so a solve reproduces the
# same step sequence.
_SAFETY = 0.9
_IFACTOR = 10.0
_DFACTOR = 0.2
_MAX_NUM_STEPS = 2**31 - 1


@dataclass(frozen=True)
class AdaptiveOptions:
    """Parsed ``options`` dict for an adaptive solver.

    Attributes
    ----------
    min_step, max_step : float
        Bounds on the step magnitude.
    first_step : float or None
        Initial step magnitude; chosen automatically when ``None``.
    step_t, jump_t : tuple of float
        Times the integrator must land on exactly.  ``step_t`` continues
        through; ``jump_t`` marks a discontinuity in the right-hand side, so
        the step after it restarts just past the time.
    safety, ifactor, dfactor : float
        Controller gains — the target margin, and the most a step may grow or
        shrink in one adjustment.
    max_num_steps : int
        Budget of accepted steps before the solve gives up.
    """

    min_step: float = 0.0
    max_step: float = math.inf
    first_step: float | None = None
    step_t: tuple[float, ...] = ()
    jump_t: tuple[float, ...] = ()
    safety: float = _SAFETY
    ifactor: float = _IFACTOR
    dfactor: float = _DFACTOR
    max_num_steps: int = _MAX_NUM_STEPS


_ADAPTIVE_KEYS = frozenset(
    {
        "min_step",
        "max_step",
        "first_step",
        "step_t",
        "jump_t",
        "safety",
        "ifactor",
        "dfactor",
        "max_num_steps",
        "dtype",
        "norm",
    }
)


def parse_options(options: dict[str, object] | None) -> AdaptiveOptions:
    """Validate and convert an ``options`` mapping for an adaptive solver.

    Parameters
    ----------
    options : dict or None
        Caller-supplied options; ``None`` means all defaults.

    Returns
    -------
    AdaptiveOptions
        The parsed options.

    Raises
    ------
    ValueError
        If a key is not recognised, or a value is out of range.

    Notes
    -----
    ``dtype`` is accepted and ignored: step control here runs in host doubles
    unconditionally, which is what that option exists to request.  ``norm`` is
    accepted and ignored too — the fused kernel implements the RMS norm, and
    swapping it would mean giving up the fusion.
    """
    if options is None:
        return AdaptiveOptions()
    unknown = set(options) - _ADAPTIVE_KEYS
    if unknown:
        raise ValueError(
            f"unknown option(s) for an adaptive method: {sorted(unknown)}; "
            f"expected a subset of {sorted(_ADAPTIVE_KEYS)}"
        )

    def _real(key: str, default: float) -> float:
        raw = options.get(key, default)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(
                f"option {key!r} must be a real number, got {type(raw).__name__}"
            )
        return float(raw)

    def _count(key: str, default: int) -> int:
        raw = options.get(key, default)
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"option {key!r} must be an int, got {type(raw).__name__}")
        return raw

    def _times(key: str) -> tuple[float, ...]:
        raw = options.get(key)
        if raw is None:
            return ()
        if isinstance(raw, Tensor):
            values = [float(v) for v in cast(list[SupportsFloat], raw.tolist())]
        elif isinstance(raw, (list, tuple)):
            values = [float(cast(SupportsFloat, v)) for v in raw]
        else:
            raise ValueError(
                f"option {key!r} must be a sequence of times or a 1-D tensor, "
                f"got {type(raw).__name__}"
            )
        return tuple(sorted(values))

    first_raw = options.get("first_step")
    if first_raw is None:
        first = None
    else:
        if isinstance(first_raw, bool) or not isinstance(first_raw, (int, float)):
            raise ValueError(
                f"option 'first_step' must be a real number, "
                f"got {type(first_raw).__name__}"
            )
        first = abs(float(first_raw))
        if first == 0.0:
            raise ValueError("first_step must be non-zero")

    parsed = AdaptiveOptions(
        min_step=abs(_real("min_step", 0.0)),
        max_step=abs(_real("max_step", math.inf)),
        first_step=first,
        step_t=_times("step_t"),
        jump_t=_times("jump_t"),
        safety=_real("safety", _SAFETY),
        ifactor=_real("ifactor", _IFACTOR),
        dfactor=_real("dfactor", _DFACTOR),
        max_num_steps=_count("max_num_steps", _MAX_NUM_STEPS),
    )
    if parsed.min_step > parsed.max_step:
        raise ValueError(
            f"min_step ({parsed.min_step}) must not exceed max_step ({parsed.max_step})"
        )
    if parsed.ifactor < 1.0:
        raise ValueError(f"ifactor must be >= 1, got {parsed.ifactor}")
    if not 0.0 < parsed.dfactor <= 1.0:
        raise ValueError(f"dfactor must lie in (0, 1], got {parsed.dfactor}")
    if parsed.max_num_steps < 1:
        raise ValueError(f"max_num_steps must be >= 1, got {parsed.max_num_steps}")
    return parsed


def optimal_step_size(
    last_step: float,
    ratio: float,
    safety: float,
    ifactor: float,
    dfactor: float,
    order: int,
) -> float:
    """Propose the next step magnitude from the last one and its error ratio.

    Parameters
    ----------
    last_step : float
        Magnitude of the step just attempted.
    ratio : float
        Its error ratio; at most ``1`` means the step was accepted.
    safety, ifactor, dfactor : float
        Controller gains.
    order : int
        Order of the method, the exponent applied to the ratio.

    Returns
    -------
    float
        The proposed magnitude, always positive.

    Notes
    -----
    A step that was accepted is never shrunk — ``dfactor`` is neutralised in
    that case, so a comfortable step can only grow or stay put.
    """
    if ratio == 0.0:
        return last_step * ifactor
    if ratio < 1.0:
        dfactor = 1.0
    # ``math.pow`` rather than ``**``: the ratio is strictly positive here, but
    # the operator is typed to allow a complex result for a negative base.
    factor = min(ifactor, max(safety / math.pow(ratio, 1.0 / order), dfactor))
    return last_step * factor


def select_initial_step(
    func: Callable[[Tensor, Tensor], Tensor],
    t0: float,
    y0: Tensor,
    f0: Tensor,
    order: int,
    rtol: float,
    atol: float,
    direction: float,
    scalar: Callable[[float], Tensor],
) -> float:
    """Pick a starting step magnitude from the local scale of the problem.

    Parameters
    ----------
    func : callable
        The right-hand side.
    t0 : float
        Initial time.
    y0 : Tensor
        Initial state.
    f0 : Tensor
        ``func(t0, y0)``, already computed.
    order : int
        Order of the method.
    rtol, atol : float
        Tolerances.
    direction : float
        ``+1`` integrating forwards, ``-1`` backwards.
    scalar : callable
        Builds the 0-D time tensor the right-hand side expects.

    Returns
    -------
    float
        A positive step magnitude.

    Notes
    -----
    The standard two-probe heuristic: size a trial step from the ratio of the
    state to its derivative, take it, and refine from how much the derivative
    changed.  Costs one extra right-hand-side call at the start of a solve.

    All three norms it needs are the same fused kernel the step controller
    uses, with the estimate standing in for the state or derivative — so this
    adds no new engine surface.
    """
    d0 = _fused.error_ratio(y0, y0, [y0], [1.0], 1.0, rtol, atol)
    d1 = _fused.error_ratio(y0, y0, [f0], [1.0], 1.0, rtol, atol)

    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1

    y1 = _fused.combine(y0, [f0], [1.0], direction * h0)
    f1 = func(scalar(t0 + direction * h0), y1)
    d2 = _fused.error_ratio(y0, y0, [f1, f0], [1.0, -1.0], 1.0, rtol, atol) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / (order + 1))

    return min(100.0 * h0, h1)


def interp_fit(
    y0: Tensor, y1: Tensor, y_mid: Tensor, f0: Tensor, f1: Tensor, dt: float
) -> list[Tensor]:
    r"""Fit the quartic that reproduces a step's endpoints, midpoint and slopes.

    Parameters
    ----------
    y0, y1 : Tensor
        States at the two ends of the step.
    y_mid : Tensor
        State at the midpoint, from the tableau's ``mid`` weights.
    f0, f1 : Tensor
        Derivatives at the two ends.
    dt : float
        Step size.

    Returns
    -------
    list of Tensor
        Five coefficients in descending powers of the normalised time.

    Notes
    -----
    On :math:`x = (t - t_0) / \Delta t` the polynomial
    :math:`p(x) = ax^4 + bx^3 + cx^2 + dx + e` is pinned by five conditions —
    :math:`p(0) = y_0`, :math:`p(1) = y_1`, :math:`p(1/2) = y_{mid}`,
    :math:`p'(0) = \Delta t f_0`, :math:`p'(1) = \Delta t f_1` — and solving
    that system gives the coefficients below.

    The layout is an implementation detail, not part of any public contract,
    so it is simply whatever :func:`interp_evaluate` expects.

    The five inputs are promoted as a group first.  Under ``autocast`` the
    derivatives come back at a lower precision than the state, and a
    coefficient built from derivatives alone would otherwise carry that
    precision into the engine on its own — with nothing to promote against.
    """
    y0, y1, y_mid, f0, f1 = _fused._promote([y0, y1, y_mid, f0, f1])
    return [
        _fused.lincomb(
            [f1, f0, y0, y1, y_mid], [2.0 * dt, -2.0 * dt, -8.0, -8.0, 16.0]
        ),
        _fused.lincomb(
            [f0, f1, y0, y1, y_mid], [5.0 * dt, -3.0 * dt, 18.0, 14.0, -32.0]
        ),
        _fused.lincomb([f1, f0, y0, y1, y_mid], [dt, -4.0 * dt, -11.0, -5.0, 16.0]),
        _fused.lincomb([f0], [dt]),
        y0,
    ]


def interp_evaluate(coeffs: Sequence[Tensor], t0: float, t1: float, t: float) -> Tensor:
    """Evaluate a fitted step polynomial at a time inside the step.

    Parameters
    ----------
    coeffs : sequence of Tensor
        Output of :func:`interp_fit`.
    t0, t1 : float
        Ends of the step the polynomial was fitted on.
    t : float
        Query time, inside ``[t0, t1]`` (either orientation).

    Returns
    -------
    Tensor
        The interpolated state.

    Notes
    -----
    The powers of the normalised time are host floats, so the whole evaluation
    is one affine combination of the coefficient tensors — a single fused
    engine call rather than a chain of scalar multiplies.
    """
    x = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    return _fused.lincomb(list(coeffs), [x**4, x**3, x**2, x, 1.0])


def _clamp(step: float, opts: AdaptiveOptions) -> float:
    """Clamp a proposed step magnitude into the configured bounds."""
    return min(max(step, opts.min_step), opts.max_step)


def _next_forced(times: Sequence[float], t: float, direction: float) -> float | None:
    """Return the next time in ``times`` strictly ahead of ``t``, if any."""
    if direction > 0:
        ahead = [v for v in times if v > t]
        return min(ahead) if ahead else None
    ahead = [v for v in times if v < t]
    return max(ahead) if ahead else None


def integrate(
    func: Callable[[Tensor, Tensor], Tensor],
    y0: Tensor,
    grid: list[float],
    tableau: ButcherTableau,
    scalar: Callable[[float], Tensor],
    check: Callable[[object, int, int], Tensor],
    *,
    rtol: float,
    atol: float,
    options: dict[str, object] | None,
    return_trajectory: bool,
) -> tuple[Tensor, list[Tensor]]:
    """Integrate to every time in ``grid`` with adaptive steps and interpolation.

    Parameters
    ----------
    func : callable
        Right-hand side ``f(t, y)``.
    y0 : Tensor
        Initial state, at ``grid[0]``.
    grid : list of float
        Output times, strictly monotonic.
    tableau : ButcherTableau
        An adaptive tableau, i.e. one carrying ``b_error`` and ``mid``.
    scalar : callable
        Builds the 0-D time tensor the right-hand side expects.
    check : callable
        Validates a right-hand-side result and returns it.
    rtol, atol : float
        Tolerances.
    options : dict or None
        Controller options, see :class:`AdaptiveOptions`.
    return_trajectory : bool
        Whether to collect the state at every output time.

    Returns
    -------
    tuple
        The final state, and the collected trajectory (empty when not
        requested).

    Raises
    ------
    RuntimeError
        If the solve exceeds ``max_num_steps``, or if the step size collapses
        below what the controller can act on.
    """
    opts = parse_options(options)
    direction = 1.0 if grid[-1] > grid[0] else -1.0
    forced = tuple(sorted({*opts.step_t, *opts.jump_t}))

    t_cur = grid[0]
    y = y0
    f0 = check(func(scalar(t_cur), y), 0, 0)

    if opts.first_step is not None:
        step = opts.first_step
    else:
        step = select_initial_step(
            func, t_cur, y0, f0, tableau.order, rtol, atol, direction, scalar
        )
    step = _clamp(step, opts)

    trajectory: list[Tensor] = [y0] if return_trajectory else []
    accepted = 0
    # Polynomial for the most recently accepted step, and the interval it
    # covers.  Fitted lazily — only a step that actually spans an output time
    # pays for the fit.
    span: tuple[float, float] | None = None
    coeffs: list[Tensor] = []

    for target in grid[1:]:
        while direction * (t_cur - target) < 0.0:
            # Only times the caller pinned constrain the step.  Output times
            # deliberately do not: clipping to them would tie the step size to
            # the output grid, which is the very coupling dense output exists
            # to remove — and would make this solver's step sequence differ
            # from the reference implementation's.
            limit = _next_forced(forced, t_cur, direction)
            if limit is not None:
                step = min(step, abs(limit - t_cur))
            if step <= 0.0:
                raise RuntimeError(
                    f"step size collapsed to {step!r} at t={t_cur!r}; the "
                    f"problem may be too stiff for an explicit method"
                )

            t_next = t_cur + direction * step
            ks: list[Tensor] = [f0]
            for stage in range(1, tableau.stages):
                stage_y = _fused.combine(y, ks, tableau.a[stage], direction * step)
                stage_t = scalar(t_cur + tableau.c[stage] * direction * step)
                ks.append(check(func(stage_t, stage_y), accepted, stage))

            y_next = _fused.combine(y, ks, tableau.b, direction * step)
            assert tableau.b_error is not None
            ratio = _fused.error_ratio(
                y, y_next, ks, tableau.b_error, direction * step, rtol, atol
            )

            if ratio > 1.0:
                step = _clamp(
                    optimal_step_size(
                        step,
                        ratio,
                        opts.safety,
                        opts.ifactor,
                        opts.dfactor,
                        tableau.order,
                    ),
                    opts,
                )
                continue

            accepted += 1
            if accepted > opts.max_num_steps:
                raise RuntimeError(
                    f"exceeded max_num_steps ({opts.max_num_steps}) before reaching "
                    f"t={grid[-1]!r}; loosen rtol/atol or raise the budget"
                )

            # The last stage of an FSAL tableau already evaluated the
            # right-hand side at the new state, so it is the next step's first
            # stage for free.  Otherwise the derivative there has to be
            # computed: reusing the final stage instead would seed the next
            # step from the wrong point and cost the method its order.
            if tableau.is_fsal:
                f_next = ks[-1]
            else:
                f_next = check(func(scalar(t_next), y_next), accepted, tableau.stages)

            span = (t_cur, t_next)
            fitted = False
            if direction * (t_next - target) >= 0.0:
                assert tableau.mid is not None
                y_mid = _fused.combine(y, ks, tableau.mid, direction * step)
                coeffs = interp_fit(y, y_next, y_mid, f0, f_next, direction * step)
                fitted = True

            t_cur, y, f0 = t_next, y_next, f_next
            step = _clamp(
                optimal_step_size(
                    step, ratio, opts.safety, opts.ifactor, opts.dfactor, tableau.order
                ),
                opts,
            )
            if fitted:
                break

        if return_trajectory:
            if t_cur == target or span is None:
                trajectory.append(y)
            else:
                trajectory.append(interp_evaluate(coeffs, span[0], span[1], target))

    # ``y`` sits at the last accepted step, which generally overshoots the
    # final output time; the value asked for is the interpolated one.
    if t_cur == grid[-1] or span is None:
        final = y
    else:
        final = interp_evaluate(coeffs, span[0], span[1], grid[-1])
    return final, trajectory
