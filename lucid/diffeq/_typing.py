"""Names for the callables and modes that cross ``lucid.diffeq``'s seams.

The package passes a handful of callables between its layers, and every one of
them was spelled structurally at each site: ``Callable[[Tensor, Tensor],
Tensor]`` appeared thirty-one times, ``Callable[[float], Tensor]`` seventeen.
That last one is the reason this module exists rather than a few aliases being
enough — it stood for two entirely different things, a factory that builds the
0-D time tensor and an interpolant that evaluates a step's polynomial, and no
signature could tell them apart.

The protocols below are structural, so nothing has to subclass anything: an
ordinary function or a ``nn.Module`` satisfies :class:`RightHandSide` by having
the right shape.  What they add over a bare ``Callable`` is a name that says what the callable
means, and a place to document it.

Every parameter is positional-only.  A caller's ``f`` is their own function
and may name its arguments anything -- ``def f(time, state)`` has to satisfy
:class:`RightHandSide` -- so the names below document the roles without
becoming part of the contract.

Import policy
-------------
Bottom of the package: this module imports ``Tensor`` and nothing else from
``lucid``, so every other ``lucid.diffeq`` module can import it freely.
"""

from typing import Literal, Protocol, TypeIs

from lucid._tensor.tensor import Tensor

__all__: list[str] = []


# ── State ────────────────────────────────────────────────────────────────────

# What the solvers integrate.  A tuple state is packed to a single flat vector
# at the entry points and split again on the way out, so only the boundary
# modules ever see the tuple form.
type State = Tensor | tuple[Tensor, ...]


# ── Modes ────────────────────────────────────────────────────────────────────

# How a fixed-step solve reaches an output time that falls inside a step.
# ``"cubic"`` costs one extra right-hand-side evaluation per step and is worth
# it whenever output times are not the integration grid: ``"linear"`` caps the
# interpolation at second order regardless of the method's own.
type InterpKind = Literal["linear", "cubic"]


def is_interp_kind(value: object) -> TypeIs[InterpKind]:
    """Narrow an option-dict value to :data:`InterpKind`.

    The runtime membership test and the static narrowing are the same
    statement here, so a value that reaches :class:`FixedOptions` has been
    checked once and is typed accordingly -- no cast standing in for a check
    that happened somewhere else.
    """
    return value in ("linear", "cubic")


# ── Callables ────────────────────────────────────────────────────────────────


class RightHandSide(Protocol):
    """The ``f`` in ``dy/dt = f(t, y)``.

    Typically a neural network, which is why the integration loops stay in
    Python: driving them from C++ would make the engine's ops layer call back
    up into Python and invert the layer DAG.
    """

    def __call__(self, t: Tensor, y: Tensor, /) -> Tensor:
        """Return ``dy/dt`` at time ``t`` and state ``y``.

        Parameters
        ----------
        t : Tensor
            0-D time tensor, in the state's dtype and on its device.  The same
            dtype and device for the whole solve.
        y : Tensor
            Current state.

        Returns
        -------
        Tensor
            The derivative, matching ``y`` in shape and device.
        """
        ...


class EventFunction(Protocol):
    """A scalar ``g(t, y)`` whose sign change marks the moment a solve ends.

    Structurally the same as :class:`RightHandSide`, and deliberately a
    separate name: the two are never interchangeable in a call, and reading
    ``EventFunction`` at a parameter says which one is wanted.
    """

    def __call__(self, t: Tensor, y: Tensor, /) -> Tensor:
        """Return the event value at ``(t, y)`` as a single-element tensor."""
        ...


class ScalarFactory(Protocol):
    """Builds the 0-D time tensor a right-hand side expects.

    The dtype and device are fixed once per solve, so the loops carry this
    rather than re-deriving them at every stage.
    """

    def __call__(self, t: float, /) -> Tensor:
        """Return ``t`` as a 0-D tensor in the solve's dtype and device."""
        ...


class Interpolant(Protocol):
    """Evaluates one step's polynomial at a time inside that step.

    What dense output, off-grid output times, and event bisection all reduce
    to.  Distinct from :class:`ScalarFactory` despite the identical signature,
    which is exactly the confusion this module was written to end.
    """

    def __call__(self, t: float, /) -> Tensor:
        """Return the interpolated state at time ``t``."""
        ...


class StageCheck(Protocol):
    """Validates whatever a right-hand side returned for one stage.

    A caller's ``f`` is arbitrary Python and may return the wrong type, shape,
    device, or a non-finite value.  Checking at the stage that produced it is
    what makes the error name the stage rather than surfacing many evaluations
    later as an all-NaN result.
    """

    def __call__(self, value: object, step: int, stage: int, /) -> Tensor:
        """Return ``value`` as a validated Tensor, or raise naming the stage.

        Parameters
        ----------
        value : object
            Whatever the right-hand side returned.
        step : int
            Index of the step being taken, for the error message.
        stage : int
            Index of the stage within that step.
        """
        ...
