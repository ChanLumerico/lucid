"""Explicit Runge-Kutta integration of ``dy/dt = f(t, y)``.

The public entry point :func:`odeint` lives here and dispatches on the
tableau: a tableau carrying an embedded error estimate goes to the adaptive
stepper in :mod:`lucid.diffeq._adaptive`, everything else to the fixed-grid
loop below.

The loop stays in Python on purpose.  The right-hand side is a Python callable
— typically a neural network — so driving the loop from C++ would force the
``ops`` layer to call back up into Python, inverting the engine's layer DAG.
What *does* live in C++ is the per-step arithmetic: every stage input and
every state update is the same affine form, fused into a single engine op.
That collapses roughly ten temporaries per RK4 step down to four.

The dominant cost of a solve is still the ``stages x steps`` right-hand-side
evaluations; the fusion trims the bandwidth-bound arithmetic around them, not
the network forwards themselves.
"""

import math
from typing import Callable, Sequence, SupportsFloat, cast

import lucid
from lucid._tensor.tensor import Tensor
from lucid.diffeq import _adaptive, _fused
from lucid.diffeq._tableau import ButcherTableau, _DEFAULT_METHOD, _METHODS

# The fixed-grid loop and the tests both reach for this; it is the fused
# stage combination that every explicit method reduces to.
_combine = _fused.combine

__all__ = ["odeint"]


def _resolve_method(method: str | ButcherTableau | None) -> ButcherTableau:
    """Turn a method name, tableau instance, or ``None`` into a tableau.

    Parameters
    ----------
    method : str or ButcherTableau or None
        A registered method name, a tableau to use verbatim, or ``None`` for
        the default method.

    Returns
    -------
    ButcherTableau
        The resolved tableau.

    Raises
    ------
    ValueError
        If ``method`` is a string that names no registered method.
    TypeError
        If ``method`` is neither a string nor a :class:`ButcherTableau`.
    """
    if method is None:
        method = _DEFAULT_METHOD
    if isinstance(method, ButcherTableau):
        return method
    if isinstance(method, str):
        tableau = _METHODS.get(method)
        if tableau is None:
            known = ", ".join(sorted(_METHODS))
            raise ValueError(f"unknown method {method!r}; expected one of: {known}")
        return tableau
    raise TypeError(
        f"method must be a str or ButcherTableau, got {type(method).__name__}"
    )


def _resolve_grid(t: Tensor | Sequence[float]) -> list[float]:
    """Materialise the integration grid as host floats, exactly once.

    Parameters
    ----------
    t : Tensor or sequence of float
        1-D grid of time points.

    Returns
    -------
    list of float
        The grid as plain Python floats.

    Raises
    ------
    ValueError
        If the grid is not 1-D, holds fewer than two points, contains a
        non-finite value, or is not strictly monotonic.

    Notes
    -----
    A ``Tensor`` grid is read back with a single ``tolist()``.  Reading the
    step size off the device once per step instead would stall the GPU
    queue on every step of every solve.
    """
    if isinstance(t, Tensor):
        if t.ndim != 1:
            raise ValueError(f"t must be a 1-D grid, got shape {t.shape}")
        # ``tolist`` is typed for every rank; the ndim guard above already
        # establishes that this one returns a flat list.
        grid = [float(v) for v in cast(list[SupportsFloat], t.tolist())]
    else:
        grid = [float(v) for v in t]

    if len(grid) < 2:
        raise ValueError(f"t must hold at least 2 time points, got {len(grid)}")

    # Checked before monotonicity: every comparison against NaN is False, so a
    # NaN would slip through the ordering test and only surface as an all-NaN
    # result many right-hand-side evaluations later.
    for value in grid:
        if not math.isfinite(value):
            raise ValueError(f"t must hold finite time points, found {value!r}")

    ascending = grid[1] > grid[0]
    for prev, cur in zip(grid, grid[1:]):
        if (cur > prev) != ascending or cur == prev:
            raise ValueError(
                "t must be strictly monotonic (ascending or descending); "
                f"found {prev!r} followed by {cur!r}"
            )
    return grid


def odeint(
    func: Callable[[Tensor, Tensor], Tensor],
    y0: Tensor,
    t: Tensor | Sequence[float],
    *,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    method: str | ButcherTableau | None = None,
    options: dict[str, object] | None = None,
    return_trajectory: bool = True,
) -> Tensor:
    r"""Integrate ``dy/dt = f(t, y)`` with an explicit Runge-Kutta method.

    Two families are reachable through the same call, and which one runs
    follows from ``method``:

    **Adaptive** (the default, and anything carrying an embedded error
    estimate) treats ``t`` as *output* times.  The solver picks its own step
    sizes to hold the local error inside ``rtol`` / ``atol`` and interpolates
    to each requested time, so a coarse ``t`` costs nothing in accuracy.

    **Fixed step** (``euler`` / ``midpoint`` / ``heun2`` / ``heun3`` / ``rk4``)
    treats ``t`` as the integration grid itself: consecutive entries are one
    step each, with no sub-stepping and no interpolation.  ``rtol`` / ``atol``
    are unused there, and accuracy is controlled by choosing a finer grid.

    Gradient mode is left entirely to the caller.  Under grad the whole solve
    is differentiable end-to-end (discretise-then-optimise) and every stage is
    retained for backward; wrap the call in :func:`lucid.no_grad` for sampling,
    where the stage graph is pure overhead.

    Parameters
    ----------
    func : callable
        Right-hand side ``f(t, y) -> dy/dt``.  Receives the stage time as a
        0-D tensor matching ``y0`` in dtype and device, and must return a
        tensor with the same shape and device as ``y0``.
    y0 : Tensor
        Initial state at ``t[0]``.  Any shape; must have a floating dtype.
    t : Tensor or sequence of float
        Strictly monotonic 1-D grid of at least two finite time points.
        Descending grids integrate backwards in time.
    rtol : float, default=1e-7
        Relative tolerance.  Adaptive methods only.
    atol : float, default=1e-9
        Absolute tolerance.  Adaptive methods only.
    method : str or ButcherTableau or None, default=None
        ``None`` selects ``"dopri5"``.  Otherwise one of ``"dopri5"``,
        ``"bosh3"``, ``"fehlberg2"``, ``"adaptive_heun"``, ``"euler"``,
        ``"midpoint"``, ``"heun2"``, ``"heun3"``, ``"rk4"``, or a custom
        :class:`ButcherTableau`.
    options : dict or None, default=None
        Per-method controller settings.  Adaptive methods accept
        ``min_step``, ``max_step``, ``first_step``, ``step_t``, ``jump_t``,
        ``safety``, ``ifactor``, ``dfactor``, ``max_num_steps`` (plus
        ``dtype`` and ``norm``, accepted and ignored).  Fixed-step methods
        accept none.
    return_trajectory : bool, default=True
        Return the state at every time in ``t``.  Set ``False`` to keep only
        the final state — for a sampling run of many steps over a batch of
        images, stacking the full trajectory multiplies peak memory by the
        number of steps.  **Lucid extension**, not part of the reference
        interface.

    Returns
    -------
    Tensor
        Shape ``(len(t), *y0.shape)`` when ``return_trajectory`` is ``True``
        (index 0 is ``y0`` itself), otherwise ``y0.shape``.  The dtype is the
        promotion of ``y0`` with everything ``func`` returns, matching what
        ``y + dt * k`` would produce.

    Raises
    ------
    ValueError
        If ``t`` is not a strictly monotonic 1-D grid of at least two finite
        points, if ``y0`` has a non-floating dtype, if ``method`` names no
        registered method, if ``options`` holds a key the method does not
        accept, or if ``func`` returns a tensor whose shape or device differs
        from ``y0``.
    TypeError
        If ``method`` is neither a string nor a :class:`ButcherTableau`, or
        if ``func`` returns something other than a tensor.
    RuntimeError
        If an adaptive solve exceeds ``max_num_steps`` or its step size
        collapses.

    Notes
    -----
    A fixed-step solve costs exactly ``(len(t) - 1) * method.stages`` calls to
    ``func``.  An adaptive solve costs as many as the tolerances demand, and
    reads one scalar back to the host per step to decide whether to accept it
    — that host synchronisation is intrinsic to adaptivity, which is why the
    two families have different performance characters.

    Higher-order differentiation works: the fused step opts into
    graph-recording backward, so ``create_graph=True`` through a solve behaves
    exactly as it would for the unfused arithmetic.

    Examples
    --------
    Exponential decay against its closed form:

    >>> import lucid, lucid.diffeq as diffeq
    >>> y0 = lucid.tensor([1.0], dtype=lucid.float64)
    >>> y = diffeq.odeint(lambda s, y: -y, y0, [0.0, 1.0], return_trajectory=False)
    >>> abs(float(y.item()) - 0.36787944117) < 1e-8
    True

    See Also
    --------
    lucid.diffeq.ButcherTableau : Coefficient table selected by ``method``.
    """
    tableau = _resolve_method(method)
    grid = _resolve_grid(t)

    if not y0.is_floating_point():
        raise ValueError(
            f"y0 must have a floating dtype for integration, got {y0.dtype}"
        )

    def scalar(value: float) -> Tensor:
        return lucid.tensor(value, dtype=y0.dtype, device=y0.device)

    def check(result: object, step: int, stage: int) -> Tensor:
        if not isinstance(result, Tensor):
            raise TypeError(
                f"func must return a Tensor, got {type(result).__name__} "
                f"at step {step}, stage {stage}"
            )
        if result.shape != y0.shape:
            raise ValueError(
                f"func returned shape {result.shape} but y0 has shape {y0.shape} "
                f"(step {step}, stage {stage})"
            )
        # Checked here rather than left to the engine so the message names
        # ``func`` — the caller's actual mistake — instead of surfacing as a
        # DeviceMismatch on the internal fused op.
        if result.device != y0.device:
            raise ValueError(
                f"func returned a tensor on {result.device} but y0 is on "
                f"{y0.device} (step {step}, stage {stage})"
            )
        return result

    if tableau.is_adaptive:
        y, trajectory = _adaptive.integrate(
            func,
            y0,
            grid,
            tableau,
            scalar,
            check,
            rtol=rtol,
            atol=atol,
            options=options,
            return_trajectory=return_trajectory,
        )
    else:
        if options:
            raise ValueError(
                f"method {tableau.name!r} is a fixed-step method and accepts no "
                f"options; got {sorted(options)}"
            )
        y, trajectory = _fixed_grid(
            func, y0, grid, tableau, scalar, check, return_trajectory
        )

    if not return_trajectory:
        return y
    # Promotion can lift the state above y0's dtype (a float64 RHS over a
    # float32 y0), which would leave trajectory[0] as the odd one out and
    # make the stack fail.  Settle the whole trajectory on the final dtype.
    trajectory = [s if s.dtype == y.dtype else s.to(y.dtype) for s in trajectory]
    return lucid.stack(trajectory, dim=0)


def _fixed_grid(
    func: Callable[[Tensor, Tensor], Tensor],
    y0: Tensor,
    grid: list[float],
    tableau: ButcherTableau,
    scalar: Callable[[float], Tensor],
    check: Callable[[object, int, int], Tensor],
    return_trajectory: bool,
) -> tuple[Tensor, list[Tensor]]:
    """Step once per grid interval, with no error control and no interpolation.

    Parameters
    ----------
    func : callable
        Right-hand side.
    y0 : Tensor
        Initial state.
    grid : list of float
        The integration grid; consecutive entries are one step each.
    tableau : ButcherTableau
        Any explicit tableau; the embedded error weights are ignored here.
    scalar : callable
        Builds the 0-D time tensor the right-hand side expects.
    check : callable
        Validates a right-hand-side result and returns it.
    return_trajectory : bool
        Whether to collect the state at every grid point.

    Returns
    -------
    tuple
        The final state, and the collected trajectory (empty when not
        requested).
    """
    y = y0
    trajectory: list[Tensor] = [y0] if return_trajectory else []

    for step in range(len(grid) - 1):
        t_start = grid[step]
        dt = grid[step + 1] - t_start

        ks: list[Tensor] = []
        for stage in range(tableau.stages):
            stage_y = _combine(y, ks, tableau.a[stage], dt)
            stage_t = scalar(t_start + tableau.c[stage] * dt)
            ks.append(check(func(stage_t, stage_y), step, stage))

        y = _combine(y, ks, tableau.b, dt)
        if return_trajectory:
            trajectory.append(y)

    return y, trajectory
