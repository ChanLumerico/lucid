"""Domain constraints for ``lucid.distributions``.

A :class:`Constraint` answers the question *"is this value in the support
of some random variable?"* via :meth:`Constraint.check`, which returns a
boolean :class:`Tensor` broadcast over the input.

The hierarchy mirrors the reference framework's set, but trimmed to the
constraints actually used by the distributions Lucid ships.
"""

import lucid
from lucid._tensor.tensor import Tensor


class Constraint:
    """Base class — every concrete constraint defines :meth:`check`."""

    is_discrete: bool = False
    event_dim: int = 0

    def check(self, value: Tensor) -> Tensor:
        raise NotImplementedError(
            f"{type(self).__name__}.check is not implemented"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class _Real(Constraint):
    """``ℝ`` — all finite reals."""

    def check(self, value: Tensor) -> Tensor:
        return lucid.isfinite(value)


class _Boolean(Constraint):
    """``{0, 1}``."""

    is_discrete = True

    def check(self, value: Tensor) -> Tensor:
        zero = lucid.zeros_like(value)
        one = lucid.ones_like(value)
        return (value == zero) | (value == one)


class _Positive(Constraint):
    """``(0, ∞)``."""

    def check(self, value: Tensor) -> Tensor:
        return value > 0


class _Nonnegative(Constraint):
    """``[0, ∞)``."""

    def check(self, value: Tensor) -> Tensor:
        return value >= 0


class _UnitInterval(Constraint):
    """``[0, 1]``."""

    def check(self, value: Tensor) -> Tensor:
        return (value >= 0) & (value <= 1)


class _OpenUnitInterval(Constraint):
    """``(0, 1)``."""

    def check(self, value: Tensor) -> Tensor:
        return (value > 0) & (value < 1)


class _GreaterThan(Constraint):
    """``(lower_bound, ∞)``."""

    def __init__(self, lower_bound: float) -> None:
        self.lower_bound = lower_bound

    def check(self, value: Tensor) -> Tensor:
        return value > self.lower_bound

    def __repr__(self) -> str:
        return f"GreaterThan(lower_bound={self.lower_bound})"


class _GreaterThanEq(Constraint):
    """``[lower_bound, ∞)``."""

    def __init__(self, lower_bound: float) -> None:
        self.lower_bound = lower_bound

    def check(self, value: Tensor) -> Tensor:
        return value >= self.lower_bound

    def __repr__(self) -> str:
        return f"GreaterThanEq(lower_bound={self.lower_bound})"


class _LessThan(Constraint):
    """``(-∞, upper_bound)``."""

    def __init__(self, upper_bound: float) -> None:
        self.upper_bound = upper_bound

    def check(self, value: Tensor) -> Tensor:
        return value < self.upper_bound

    def __repr__(self) -> str:
        return f"LessThan(upper_bound={self.upper_bound})"


class _Interval(Constraint):
    """``[lower_bound, upper_bound]``."""

    def __init__(self, lower_bound: float, upper_bound: float) -> None:
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def check(self, value: Tensor) -> Tensor:
        return (value >= self.lower_bound) & (value <= self.upper_bound)

    def __repr__(self) -> str:
        return (
            f"Interval(lower_bound={self.lower_bound}, "
            f"upper_bound={self.upper_bound})"
        )


class _IntegerInterval(Constraint):
    """``{lower_bound, lower_bound+1, ..., upper_bound}``."""

    is_discrete = True

    def __init__(self, lower_bound: int, upper_bound: int) -> None:
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def check(self, value: Tensor) -> Tensor:
        in_range = (value >= self.lower_bound) & (value <= self.upper_bound)
        # Integer-valued: floor(x) == x.
        return in_range & (lucid.floor(value) == value)

    def __repr__(self) -> str:
        return (
            f"IntegerInterval(lower_bound={self.lower_bound}, "
            f"upper_bound={self.upper_bound})"
        )


class _NonnegativeInteger(Constraint):
    """``{0, 1, 2, ...}``."""

    is_discrete = True

    def check(self, value: Tensor) -> Tensor:
        return (value >= 0) & (lucid.floor(value) == value)


class _Simplex(Constraint):
    """The K-simplex: ``x ≥ 0`` and ``Σ x = 1`` along the last dimension."""

    event_dim = 1

    def check(self, value: Tensor) -> Tensor:
        # Lucid's ``Tensor.all`` is a 0-dim reducer, so we hand-roll the
        # along-last-axis "all non-negative" check via ``min``.
        nonneg = value.min(dim=-1) >= 0
        sums = value.sum(dim=-1)
        unit = (sums - 1.0).abs() < 1e-6
        return nonneg & unit


class _PositiveDefinite(Constraint):
    """Matrix is positive-definite (symmetric, all eigenvalues > 0).

    Cheap check — relies on a successful Cholesky decomposition.  A failure
    in the linear solver is interpreted as a non-PD matrix.
    """

    event_dim = 2

    def check(self, value: Tensor) -> Tensor:
        try:
            lucid.linalg.cholesky(value)
            return lucid.tensor(True)
        except Exception:
            return lucid.tensor(False)


# Public singletons (functions/objects users actually reach for).
real = _Real()
boolean = _Boolean()
positive = _Positive()
nonnegative = _Nonnegative()
unit_interval = _UnitInterval()
open_unit_interval = _OpenUnitInterval()
simplex = _Simplex()
positive_definite = _PositiveDefinite()
nonnegative_integer = _NonnegativeInteger()


def greater_than(lower_bound: float) -> Constraint:
    return _GreaterThan(lower_bound)


def greater_than_eq(lower_bound: float) -> Constraint:
    return _GreaterThanEq(lower_bound)


def less_than(upper_bound: float) -> Constraint:
    return _LessThan(upper_bound)


def interval(lower_bound: float, upper_bound: float) -> Constraint:
    return _Interval(lower_bound, upper_bound)


def integer_interval(lower_bound: int, upper_bound: int) -> Constraint:
    return _IntegerInterval(lower_bound, upper_bound)
