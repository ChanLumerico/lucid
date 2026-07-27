"""Fixed-step explicit Runge-Kutta integration of ``dy/dt = f(t, y)``.

The loop lives in Python on purpose.  The right-hand side is a Python callable
— typically a neural network — so driving the loop from C++ would force the
``ops`` layer to call back up into Python, inverting the engine's layer DAG.
What *does* live in C++ is the per-step arithmetic: every stage input and
every state update is the same affine form, fused into the single engine op
``_C_engine.diffeq.rk_combine``.  That collapses roughly ten temporaries per
RK4 step down to four.

The dominant cost of a solve is still the ``stages x steps`` right-hand-side
evaluations; the fusion trims the bandwidth-bound arithmetic around them, not
the network forwards themselves.
"""

import math
from typing import Callable, Sequence, SupportsFloat, cast

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._tensor.tensor import Tensor
from lucid.diffeq._tableau import ButcherTableau, _METHODS

__all__ = ["odeint"]


def _resolve_method(method: str | ButcherTableau) -> ButcherTableau:
    """Turn a method name or tableau instance into a tableau.

    Parameters
    ----------
    method : str or ButcherTableau
        A registered method name, or a tableau to use verbatim.

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


def _combine(
    y0: Tensor, ks: list[Tensor], coeffs: Sequence[float], dt: float
) -> Tensor:
    """Evaluate ``y0 + dt * sum_i coeffs[i] * ks[i]`` via the fused engine op.

    Parameters
    ----------
    y0 : Tensor
        Base state.
    ks : list of Tensor
        Stage derivatives accumulated so far.
    coeffs : sequence of float
        One weight per entry of ``ks``.
    dt : float
        Step size.

    Returns
    -------
    Tensor
        The combined state, or ``y0`` itself when the sum is empty.

    Notes
    -----
    An all-zero row contributes nothing, so the engine call is skipped and
    ``y0`` is returned unchanged — this covers the first stage of every
    method, whose tableau row is empty by construction.

    Engine ops are strict about dtype, so mixed operands are promoted here
    exactly as ``Tensor.__add__`` promotes them.  This is not a corner case:
    under ``autocast`` on Metal the right-hand side returns float16 while the
    state stays float32, and rejecting that would make the fused path fail
    where the plain ``y + dt * k`` spelling succeeds.
    """
    if not ks or not any(coeffs):
        return y0

    target = y0.dtype
    for k in ks:
        target = lucid.promote_types(target, k.dtype)
    base = y0 if y0.dtype == target else y0.to(target)
    stages = [k if k.dtype == target else k.to(target) for k in ks]

    return _wrap(
        _C_engine.diffeq.rk_combine(
            _unwrap(base), [_unwrap(s) for s in stages], list(coeffs), dt
        )
    )


def odeint(
    func: Callable[[Tensor, Tensor], Tensor],
    y0: Tensor,
    t: Tensor | Sequence[float],
    *,
    method: str | ButcherTableau = "rk4",
    return_trajectory: bool = True,
) -> Tensor:
    r"""Integrate ``dy/dt = f(t, y)`` on a fixed grid with an explicit RK method.

    The supplied grid ``t`` *is* the integration grid: consecutive entries are
    stepped in one Runge-Kutta step each, with no sub-stepping and no dense
    output.  Step size therefore comes from the spacing of ``t``, and
    controlling accuracy means choosing a finer grid.

    Gradient mode is left entirely to the caller.  Under grad, the whole solve
    is differentiable end-to-end (discretise-then-optimise) and every stage is
    retained for backward; wrap the call in :func:`lucid.no_grad` for sampling,
    where the stage graph is pure overhead.

    Parameters
    ----------
    func : callable
        Right-hand side ``f(t, y) -> dy/dt``.  Receives the stage time as a
        0-D tensor matching ``y0`` in dtype and device, and must return a
        tensor with the same shape as ``y0``.
    y0 : Tensor
        Initial state at ``t[0]``.  Any shape; must have a floating dtype.
    t : Tensor or sequence of float
        Strictly monotonic 1-D grid of at least two finite time points.
        Descending grids integrate backwards in time.
    method : str or ButcherTableau, default="rk4"
        ``"euler"``, ``"midpoint"``, ``"heun2"``, ``"heun3"``, ``"rk4"``, or a
        custom :class:`ButcherTableau`.
    return_trajectory : bool, default=True
        Return the state at every grid point.  Set ``False`` to keep only the
        final state — for a sampling run of many steps over a batch of
        images, stacking the full trajectory multiplies peak memory by the
        number of steps.

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
        registered method, or if ``func`` returns a tensor whose shape or
        device differs from ``y0``.
    TypeError
        If ``method`` is neither a string nor a :class:`ButcherTableau`, or
        if ``func`` returns something other than a tensor.

    Notes
    -----
    Cost is ``(len(t) - 1) * method.stages`` calls to ``func``.  Adaptive step
    control, error estimates, and the O(1)-memory adjoint are deliberately out
    of scope here — they need a per-step error norm read back to the host,
    which changes the performance character of the loop.

    Higher-order differentiation works: the fused step opts into
    graph-recording backward, so ``create_graph=True`` through a solve behaves
    exactly as it would for the unfused arithmetic.

    Examples
    --------
    Exponential decay against its closed form:

    >>> import lucid, lucid.diffeq as diffeq
    >>> y0 = lucid.tensor([1.0])
    >>> t = [i / 20 for i in range(21)]
    >>> y = diffeq.odeint(lambda s, y: -y, y0, t, return_trajectory=False)
    >>> abs(float(y.item()) - 0.36787944) < 1e-6
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

    y = y0
    trajectory: list[Tensor] = [y0] if return_trajectory else []

    for step in range(len(grid) - 1):
        t_start = grid[step]
        dt = grid[step + 1] - t_start

        ks: list[Tensor] = []
        for stage in range(tableau.stages):
            stage_y = _combine(y, ks, tableau.a[stage], dt)
            stage_t = lucid.tensor(
                t_start + tableau.c[stage] * dt, dtype=y0.dtype, device=y0.device
            )
            k = func(stage_t, stage_y)
            if not isinstance(k, Tensor):
                raise TypeError(
                    f"func must return a Tensor, got {type(k).__name__} "
                    f"at step {step}, stage {stage}"
                )
            if k.shape != y0.shape:
                raise ValueError(
                    f"func returned shape {k.shape} but y0 has shape {y0.shape} "
                    f"(step {step}, stage {stage})"
                )
            # Checked here rather than left to the engine so the message names
            # ``func`` — the caller's actual mistake — instead of surfacing as
            # a DeviceMismatch on the internal fused op.
            if k.device != y0.device:
                raise ValueError(
                    f"func returned a tensor on {k.device} but y0 is on "
                    f"{y0.device} (step {step}, stage {stage})"
                )
            ks.append(k)

        y = _combine(y, ks, tableau.b, dt)
        if return_trajectory:
            trajectory.append(y)

    if not return_trajectory:
        return y
    # Promotion can lift the state above y0's dtype (a float64 RHS over a
    # float32 y0), which would leave trajectory[0] as the odd one out and
    # make the stack fail.  Settle the whole trajectory on the final dtype.
    trajectory = [s if s.dtype == y.dtype else s.to(y.dtype) for s in trajectory]
    return lucid.stack(trajectory, dim=0)
