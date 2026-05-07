"""``Distribution`` base class for ``lucid.distributions``.

Concrete distributions subclass this and override the bits that apply
(``rsample`` for reparameterisable, ``log_prob`` for everything,
``mean`` / ``variance`` / ``entropy`` for closed-form moments).  The
default behaviours mimic the reference framework: methods that aren't
implemented raise :class:`NotImplementedError`, ``sample`` is
``rsample`` detached, and ``stddev = √variance``.
"""

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions.constraints import Constraint


class Distribution:
    """Abstract base for a probability distribution.

    Subclasses set:
      - ``arg_constraints`` — dict of param-name → :class:`Constraint`,
        used by ``validate_args`` and to spell out the parameter domain.
      - ``support`` — :class:`Constraint` for the random variable.
      - ``has_rsample`` — ``True`` for reparameterisable families.
      - ``batch_shape``, ``event_shape``.

    Either ``rsample`` or ``sample`` (or both) must be overridden.
    """

    arg_constraints: dict[str, Constraint] = {}
    support: Constraint | None = None
    has_rsample: bool = False
    has_enumerate_support: bool = False

    _validate_args: bool = False

    def __init__(
        self,
        batch_shape: tuple[int, ...] = (),
        event_shape: tuple[int, ...] = (),
        validate_args: bool | None = None,
    ) -> None:
        self._batch_shape = tuple(batch_shape)
        self._event_shape = tuple(event_shape)
        if validate_args is not None:
            self._validate_args = validate_args
        if self._validate_args:
            self._validate_params()

    # ── shape introspection ────────────────────────────────────────────────

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch_shape

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._event_shape

    def _extended_shape(self, sample_shape: tuple[int, ...] = ()) -> tuple[int, ...]:
        return tuple(sample_shape) + self._batch_shape + self._event_shape

    # ── moments / closed-form quantities ───────────────────────────────────

    @property
    def mean(self) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__}.mean")

    @property
    def mode(self) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__}.mode")

    @property
    def variance(self) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__}.variance")

    @property
    def stddev(self) -> Tensor:
        return self.variance.sqrt()

    def entropy(self) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__}.entropy")

    # ── sampling ───────────────────────────────────────────────────────────

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Draw an iid sample.  Defaults to a detached ``rsample``."""
        return self.rsample(sample_shape).detach()

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Reparameterised sample — gradients flow through the noise.

        Concrete distributions must override either this or ``sample``.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.rsample is not reparameterisable; "
            f"use sample() for non-reparameterised draws."
        )

    # ── log-probabilities ──────────────────────────────────────────────────

    def log_prob(self, value: Tensor) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__}.log_prob")

    def cdf(self, value: Tensor) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__}.cdf")

    def icdf(self, value: Tensor) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__}.icdf")

    def prob(self, value: Tensor) -> Tensor:
        return self.log_prob(value).exp()

    # ── validation ─────────────────────────────────────────────────────────

    def _validate_params(self) -> None:
        for name, constraint in self.arg_constraints.items():
            if not hasattr(self, name):
                continue
            v = getattr(self, name)
            if not isinstance(v, lucid.Tensor):
                continue
            if not bool(constraint.check(v).all().item()):
                raise ValueError(
                    f"{type(self).__name__} parameter {name!r} "
                    f"out of constraint {constraint!r}"
                )

    def _validate_sample(self, value: Tensor) -> None:
        if self.support is None:
            return
        if not bool(self.support.check(value).all().item()):
            raise ValueError(
                f"{type(self).__name__}: value out of support {self.support!r}"
            )

    # ── pretty-printing ────────────────────────────────────────────────────

    def __repr__(self) -> str:
        params = []
        for name in self.arg_constraints:
            if not hasattr(self, name):
                continue
            v = getattr(self, name)
            if isinstance(v, lucid.Tensor):
                params.append(f"{name}={tuple(v.shape)}")
            else:
                params.append(f"{name}={v!r}")
        return f"{type(self).__name__}({', '.join(params)})"


class ExponentialFamily(Distribution):
    """Mix-in for exponential-family distributions.

    Subclasses provide ``_natural_params`` and ``_log_normalizer`` so
    that ``entropy`` can be derived from the standard exponential-family
    identity ``H = log Z(η) - η · E[T(X)] + E[h(X)]`` (per-distribution
    overrides remain for closed-form efficiency).
    """

    @property
    def _natural_params(self) -> tuple[Tensor, ...]:
        raise NotImplementedError(f"{type(self).__name__}._natural_params")

    def _log_normalizer(self, *natural_params: Tensor) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__}._log_normalizer")
