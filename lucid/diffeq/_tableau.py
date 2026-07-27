"""Butcher tableaux for the explicit Runge-Kutta family.

A Butcher tableau is the complete specification of a Runge-Kutta method: the
stage coefficient matrix :math:`a`, the combination weights :math:`b`, and the
stage times :math:`c`.  Splitting it out from the solver means a new method is
data, not code — the four built-ins below and any user-supplied tableau run
through exactly the same integration loop.

Only *explicit* methods are representable here: :math:`a` is required to be
strictly lower triangular, so stage :math:`i` depends only on stages before
it and can be evaluated in one forward sweep.  Implicit methods need a root
find per step and are out of scope.
"""

from dataclasses import dataclass

__all__ = ["ButcherTableau", "EULER", "MIDPOINT", "HEUN2", "HEUN3", "RK4"]


# Consistency conditions are exact for the built-in tableaux and hold to
# round-off for hand-entered ones, so the slack only has to absorb the error
# of summing a handful of floats.
_TOL = 1e-12


@dataclass(frozen=True)
class ButcherTableau:
    r"""Coefficient table defining one explicit Runge-Kutta method.

    Given a step from :math:`t` to :math:`t + \Delta t`, the method evaluates
    :math:`s` stage derivatives and combines them into the new state:

    .. math::

        k_i    &= f\!\left(t + c_i \Delta t,\;
                   y + \Delta t \textstyle\sum_{j<i} a_{ij} k_j\right) \\
        y_{n+1} &= y_n + \Delta t \textstyle\sum_i b_i k_i

    Parameters
    ----------
    a : tuple of tuple of float
        Strictly lower-triangular stage coefficients, stored row by row with
        the zeros above the diagonal omitted — row ``i`` holds exactly ``i``
        entries, so ``a[0]`` is empty.  Sequences of any kind are accepted
        and coerced to nested tuples.
    b : tuple of float
        Combination weights, one per stage.  Must sum to ``1``.
    c : tuple of float
        Stage times as fractions of the step.  Entry ``i`` must equal
        ``sum(a[i])``, the standard consistency condition.
    order : int
        Order of accuracy :math:`p`: halving the step size shrinks the global
        error by roughly :math:`2^p`.  Used by tests and diagnostics, never by
        the integration loop itself.
    name : str
        Human-readable method name, surfaced in error messages.

    Raises
    ------
    ValueError
        If ``a`` / ``b`` / ``c`` disagree on stage count, if any row of ``a``
        has the wrong length for a strictly lower-triangular matrix, if ``b``
        does not sum to ``1``, if any ``c[i] != sum(a[i])``, or if ``order``
        or ``name`` are unset.

    Examples
    --------
    The classical fourth-order method, written out in full:

    >>> import lucid.diffeq as diffeq
    >>> rk4 = diffeq.ButcherTableau(
    ...     a=((), (0.5,), (0.0, 0.5), (0.0, 0.0, 1.0)),
    ...     b=(1 / 6, 1 / 3, 1 / 3, 1 / 6),
    ...     c=(0.0, 0.5, 0.5, 1.0),
    ...     order=4,
    ...     name="rk4",
    ... )
    >>> rk4.stages
    4

    See Also
    --------
    lucid.diffeq.odeint : Consumes a tableau, by name or by instance.
    """

    a: tuple[tuple[float, ...], ...]
    b: tuple[float, ...]
    c: tuple[float, ...]
    order: int
    name: str

    def __post_init__(self) -> None:
        # Coerce first so that a tableau written with lists — the natural way
        # to type one out — still ends up frozen, hashable, and float-typed.
        object.__setattr__(
            self, "a", tuple(tuple(float(v) for v in row) for row in self.a)
        )
        object.__setattr__(self, "b", tuple(float(v) for v in self.b))
        object.__setattr__(self, "c", tuple(float(v) for v in self.c))

        n = len(self.b)
        if n == 0:
            raise ValueError("ButcherTableau requires at least one stage")
        if len(self.a) != n or len(self.c) != n:
            raise ValueError(
                f"ButcherTableau stage counts disagree: "
                f"len(a)={len(self.a)}, len(b)={n}, len(c)={len(self.c)}"
            )
        for i, row in enumerate(self.a):
            if len(row) != i:
                raise ValueError(
                    f"ButcherTableau row a[{i}] must hold {i} entries "
                    f"(a is strictly lower triangular), got {len(row)}"
                )
        weight_sum = sum(self.b)
        if abs(weight_sum - 1.0) > _TOL:
            raise ValueError(
                f"ButcherTableau weights b must sum to 1, got {weight_sum!r}"
            )
        for i, row in enumerate(self.a):
            row_sum = sum(row)
            if abs(self.c[i] - row_sum) > _TOL:
                raise ValueError(
                    f"ButcherTableau consistency violated at stage {i}: "
                    f"c[{i}]={self.c[i]!r} but sum(a[{i}])={row_sum!r}"
                )
        if self.order < 1:
            raise ValueError(f"ButcherTableau order must be >= 1, got {self.order}")
        if not self.name:
            raise ValueError("ButcherTableau name must be a non-empty string")

    @property
    def stages(self) -> int:
        """int: Number of stage derivatives evaluated per step.

        Equals the number of right-hand-side calls the method makes for each
        step, which is the dominant cost when the right-hand side is a neural
        network.
        """
        return len(self.b)


EULER = ButcherTableau(a=((),), b=(1.0,), c=(0.0,), order=1, name="euler")
"""ButcherTableau: Forward Euler — one stage, first order."""

MIDPOINT = ButcherTableau(
    a=((), (0.5,)), b=(0.0, 1.0), c=(0.0, 0.5), order=2, name="midpoint"
)
"""ButcherTableau: Explicit midpoint — two stages, second order."""

HEUN2 = ButcherTableau(a=((), (1.0,)), b=(0.5, 0.5), c=(0.0, 1.0), order=2, name="heun2")
"""ButcherTableau: Heun's method (explicit trapezoid) — two stages, second order."""

HEUN3 = ButcherTableau(
    a=((), (1.0 / 3.0,), (0.0, 2.0 / 3.0)),
    b=(0.25, 0.0, 0.75),
    c=(0.0, 1.0 / 3.0, 2.0 / 3.0),
    order=3,
    name="heun3",
)
"""ButcherTableau: Heun's third-order method — three stages, third order."""

RK4 = ButcherTableau(
    a=((), (0.5,), (0.0, 0.5), (0.0, 0.0, 1.0)),
    b=(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0),
    c=(0.0, 0.5, 0.5, 1.0),
    order=4,
    name="rk4",
)
"""ButcherTableau: The classical Runge-Kutta method — four stages, fourth order."""


# Name → tableau lookup backing ``odeint(method=...)``.  Private because the
# canonical way to reach a method is its name string or the module-level
# constant; a second public registry would be a second path to the same thing.
_METHODS: dict[str, ButcherTableau] = {
    EULER.name: EULER,
    MIDPOINT.name: MIDPOINT,
    HEUN2.name: HEUN2,
    HEUN3.name: HEUN3,
    RK4.name: RK4,
}
