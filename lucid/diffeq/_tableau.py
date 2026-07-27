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

__all__ = [
    "ButcherTableau",
    "EULER",
    "MIDPOINT",
    "HEUN2",
    "HEUN3",
    "RK4",
    "ADAPTIVE_HEUN",
    "FEHLBERG2",
    "BOSH3",
    "DOPRI5",
    "TSIT5",
]


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
        error by roughly :math:`2^p`.  Also the exponent the adaptive
        controller uses when proposing the next step size.
    name : str
        Human-readable method name, surfaced in error messages.
    b_error : tuple of float, optional
        Difference between the two embedded solutions' weights,
        :math:`b - \hat{b}`, one per stage.  Present exactly on the adaptive
        methods; its presence is what :attr:`is_adaptive` reports.  Must sum
        to ``0``, since both solutions are consistent.
    mid : tuple of float, optional
        Weights producing the state at the midpoint of a step, used to anchor
        the quartic interpolant that gives dense output.  Required whenever
        ``b_error`` is given.

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
    b_error: tuple[float, ...] | None = None
    mid: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        # Coerce first so that a tableau written with lists — the natural way
        # to type one out — still ends up frozen, hashable, and float-typed.
        object.__setattr__(
            self, "a", tuple(tuple(float(v) for v in row) for row in self.a)
        )
        object.__setattr__(self, "b", tuple(float(v) for v in self.b))
        object.__setattr__(self, "c", tuple(float(v) for v in self.c))
        if self.b_error is not None:
            object.__setattr__(self, "b_error", tuple(float(v) for v in self.b_error))
        if self.mid is not None:
            object.__setattr__(self, "mid", tuple(float(v) for v in self.mid))

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

        if self.b_error is not None:
            if len(self.b_error) != n:
                raise ValueError(
                    f"ButcherTableau b_error must hold {n} entries, "
                    f"got {len(self.b_error)}"
                )
            error_sum = sum(self.b_error)
            if abs(error_sum) > _TOL:
                raise ValueError(
                    f"ButcherTableau b_error must sum to 0 (both embedded "
                    f"solutions are consistent), got {error_sum!r}"
                )
            if self.mid is None:
                raise ValueError(
                    "ButcherTableau with b_error also needs mid, the midpoint "
                    "weights the dense-output interpolant is anchored on"
                )
        if self.mid is not None and len(self.mid) != n:
            raise ValueError(
                f"ButcherTableau mid must hold {n} entries, got {len(self.mid)}"
            )

    @property
    def is_adaptive(self) -> bool:
        """bool: Whether the tableau carries an embedded error estimate.

        Adaptive methods pair two solutions of different order; their weight
        difference (``b_error``) estimates the local error, which is what a
        step-size controller needs.  A tableau without it can only be stepped
        on a fixed grid.
        """
        return self.b_error is not None

    @property
    def is_fsal(self) -> bool:
        """bool: Whether the last stage derivative equals ``f(t + dt, y_next)``.

        True when the final stage row reproduces the solution weights, so the
        stage already evaluated the right-hand side at the new state.  That
        derivative can then be reused as the next step's first stage, saving
        one call per accepted step — a real saving when the right-hand side is
        a neural network.
        """
        return self.b[-1] == 0.0 and self.a[-1] == self.b[:-1]

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

HEUN2 = ButcherTableau(
    a=((), (1.0,)), b=(0.5, 0.5), c=(0.0, 1.0), order=2, name="heun2"
)
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


# ── Adaptive (embedded) pairs ────────────────────────────────────────────────
#
# Each carries ``b_error`` (the weight difference between the two embedded
# solutions) and ``mid`` (midpoint weights anchoring the dense-output
# interpolant).  Coefficients are written as exact ratios so the source reads
# like the published tableau and no decimal is silently truncated.

ADAPTIVE_HEUN = ButcherTableau(
    a=((), (1.0,)),
    b=(0.5, 0.5),
    c=(0.0, 1.0),
    order=2,
    name="adaptive_heun",
    b_error=(0.5, -0.5),
    mid=(0.5, 0.0),
)
"""ButcherTableau: Heun-Euler 2(1) — two stages, adaptive, second order."""

FEHLBERG2 = ButcherTableau(
    a=((), (0.5,), (1.0 / 256.0, 255.0 / 256.0)),
    b=(1.0 / 512.0, 255.0 / 256.0, 1.0 / 512.0),
    c=(0.0, 0.5, 1.0),
    order=2,
    name="fehlberg2",
    b_error=(-1.0 / 512.0, 0.0, 1.0 / 512.0),
    mid=(0.0, 0.5, 0.0),
)
"""ButcherTableau: Fehlberg 2(1) — three stages, adaptive, second order."""

BOSH3 = ButcherTableau(
    a=((), (0.5,), (0.0, 0.75), (2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0)),
    b=(2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0),
    c=(0.0, 0.5, 0.75, 1.0),
    order=3,
    name="bosh3",
    b_error=(2.0 / 9.0 - 7.0 / 24.0, 1.0 / 12.0, 1.0 / 9.0, -0.125),
    mid=(0.0, 0.5, 0.0, 0.0),
)
"""ButcherTableau: Bogacki-Shampine 3(2) — four stages, adaptive, third order."""

DOPRI5 = ButcherTableau(
    a=(
        (),
        (1.0 / 5.0,),
        (3.0 / 40.0, 9.0 / 40.0),
        (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
        (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0),
        (
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
        ),
        (
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
        ),
    ),
    b=(
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    ),
    c=(0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0),
    order=5,
    name="dopri5",
    b_error=(
        35.0 / 384.0 - 1951.0 / 21600.0,
        0.0,
        500.0 / 1113.0 - 22642.0 / 50085.0,
        125.0 / 192.0 - 451.0 / 720.0,
        -2187.0 / 6784.0 + 12231.0 / 42400.0,
        11.0 / 84.0 - 649.0 / 6300.0,
        -1.0 / 60.0,
    ),
    mid=(
        6025192743.0 / 60171106304.0,
        0.0,
        51252292925.0 / 130801643196.0,
        -2691868925.0 / 90256659456.0,
        187940372067.0 / 3189068634112.0,
        -1776094331.0 / 39487288512.0,
        11237099.0 / 470086768.0,
    ),
)
"""ButcherTableau: Dormand-Prince 5(4) — seven stages, adaptive, fifth order."""

TSIT5 = ButcherTableau(
    a=(
        (),
        (0.161,),
        (-0.008480655492356989, 0.335480655492356989),
        (2.897153057105493, -6.359448489975075, 4.362295432869581),
        (
            5.325864828439257,
            -11.748883564062828,
            7.495539342889836,
            -0.09249506636175525,
        ),
        (
            5.861455442946420,
            -12.920969317847109,
            8.159367898576159,
            -0.07158497328140100,
            -0.02826905039406838,
        ),
        (
            0.09646076681806523,
            0.01,
            0.4798896504144996,
            1.379008574103742,
            -3.290069515436081,
            2.324710524099774,
        ),
    ),
    b=(
        0.09468075576583946,
        0.009183565540343253,
        0.4877705284247616,
        1.234297566930479,
        -2.707712349983526,
        1.866628418170587,
        1.0 / 66.0,
    ),
    c=(0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0),
    order=5,
    name="tsit5",
    b_error=(
        -0.001780011052225771,
        -0.0008164344596567469,
        0.007880878010261996,
        -0.1447110071732629,
        0.5823571654525552,
        -0.4580821059291869,
        1.0 / 66.0,
    ),
    mid=(
        0.10741235230100377,
        0.01135625,
        0.39560903056045305,
        -0.34475214352593553,
        1.3161853649581645,
        -1.0170608542936508,
        0.031249999999999993,
    ),
)
"""ButcherTableau: Tsitouras 5(4) — seven stages, adaptive, fifth order."""


# Name → tableau lookup backing ``odeint(method=...)``.  Private because the
# canonical way to reach a method is its name string or the module-level
# constant; a second public registry would be a second path to the same thing.
_METHODS: dict[str, ButcherTableau] = {
    EULER.name: EULER,
    MIDPOINT.name: MIDPOINT,
    HEUN2.name: HEUN2,
    HEUN3.name: HEUN3,
    RK4.name: RK4,
    ADAPTIVE_HEUN.name: ADAPTIVE_HEUN,
    FEHLBERG2.name: FEHLBERG2,
    BOSH3.name: BOSH3,
    DOPRI5.name: DOPRI5,
    TSIT5.name: TSIT5,
}

# The method ``odeint`` picks when the caller passes ``method=None``.
_DEFAULT_METHOD = DOPRI5.name
