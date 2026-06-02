"""Single source of truth for compile-time optimizer hyperparameters + state schema.

The compile path has historically extracted hyperparameters ad-hoc inside each
``_Compiled<Optimizer>.__init__`` (see :file:`_optim.py`).  This module
centralizes the extraction + the per-optimizer state-buffer / per-step-scalar
schema, so adding a new optimizer is a single dataclass entry instead of a
new class with copy-pasted boilerplate.

The shape is intentionally minimal — Python-side only.  The C++ counterpart
``OptimizerSpec`` struct in :file:`lucid/_C/compile/MpsBuilder.h` mirrors a
subset of these fields (SGD / Adam / AdamW kinds today; expansion to all 8
kinds is tracked as a follow-up plan because each new C++ kind needs its own
``emit_<kind>_update`` MPSGraph implementation in :file:`MpsBuilder.mm`).

Audit reference: this addresses #6/#7 of the production-OOP audit
(:file:`obsidian/retro/retro-3-5-phase-compile-oop.md`).
"""

import enum
from dataclasses import dataclass, field
from typing import Callable, Mapping

from lucid.optim.optimizer import Optimizer


class OptimizerKind(enum.IntEnum):
    """Mirror of the C++ :enum:`lucid::compile::OptimizerSpec::Kind`.

    The integer values match the C++ enum exactly so the spec can be
    serialized to a single ``int64`` across the binding boundary
    without a separate translation table.  Future C++ values
    (RMSPROP, ADAGRAD, ADADELTA, ADAMAX, NADAM) are reserved here so
    Python can already round-trip a spec for the generic-fused-step
    path; only the hardcoded ``compile_fused_training_step`` branch
    requires a matching C++ ``emit_<kind>_update`` implementation.
    """

    SGD = 0
    ADAM = 1
    ADAMW = 2
    # Reserved (Python-only today; C++ implementations deferred).
    RMSPROP = 3
    ADAGRAD = 4
    ADADELTA = 5
    ADAMAX = 6
    NADAM = 7


@dataclass(frozen=True)
class OptimizerSpec:
    """Snapshot of an :class:`Optimizer`'s compile-relevant configuration.

    ``frozen=True`` because the spec is captured once at compile time and
    consumed by both the trace recorder (``_fused_step.py``) and the C++
    emitter (when going through :func:`compile_fused_training_step`).
    Mutating after capture would silently desync the executable.

    Attributes
    ----------
    kind : OptimizerKind
        Which optimizer family this spec describes.
    lr : float
        Learning rate.
    momentum : float
        SGD momentum coefficient.  ``0.0`` for non-SGD.
    dampening : float
        SGD dampening on the momentum update.
    weight_decay : float
        L2 weight-decay coefficient (coupled for SGD/Adam, decoupled
        for AdamW).
    nesterov : bool
        SGD lookahead.  Faithfully implemented by the compile path;
        eager SGD silently ignores this flag (see
        ``test_compile_optimizer_nesterov_correctness``).
    beta1, beta2 : float
        Adam-family first / second moment decay rates.
    eps : float
        Adam-family numerical stability.
    state_buffer_names : tuple[str, ...]
        Ordered names of the per-parameter state buffers this
        optimizer maintains (e.g. ``("mom",)`` for SGD-with-momentum,
        ``("m", "v")`` for Adam).  Empty when none.
    scalar_names : tuple[str, ...]
        Ordered names of per-step scalar feeds (e.g. Adam's
        ``("bias1", "bias2")``).  Empty when none.

    Notes
    -----
    The C++ side currently consumes only ``kind`` / ``lr`` /
    ``momentum`` / ``dampening`` / ``weight_decay`` / ``nesterov`` /
    ``beta1`` / ``beta2`` / ``eps`` (matching the
    ``OptimizerSpec`` struct in :file:`MpsBuilder.h`).  The
    ``state_buffer_names`` / ``scalar_names`` fields are Python-only;
    they describe the *shape* of the trace's auxiliary inputs so
    ``_fused_step.py`` can allocate the right placeholders without
    inspecting the ``_Compiled*`` instance directly.
    """

    kind: OptimizerKind
    lr: float
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    state_buffer_names: tuple[str, ...] = field(default_factory=tuple)
    scalar_names: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_optim(cls, opt: Optimizer) -> OptimizerSpec:
        """Capture a spec from an eager :class:`Optimizer` instance.

        Reads ``opt.param_groups[0]`` for the hyperparameters and
        dispatches on the class name to fill in the per-kind defaults.
        Raises :class:`TypeError` when the optimizer class is not on
        the supported list.

        Parameters
        ----------
        opt : Optimizer
            The eager optimizer.  Must have exactly one parameter
            group (the compile path rejects multi-group setups at the
            ``_CompiledStepBase`` layer; this helper expects that
            check to have already passed).

        Returns
        -------
        OptimizerSpec
            Immutable snapshot ready to consume.

        Raises
        ------
        TypeError
            When ``opt``'s class name is not in the supported map.
        """
        class_name = type(opt).__name__
        ctor = _CLASS_NAME_TO_FACTORY.get(class_name)
        if ctor is None:
            raise TypeError(
                f"OptimizerSpec.from_optim: unsupported optimizer "
                f"'{class_name}' — supported: "
                f"{sorted(_CLASS_NAME_TO_FACTORY)}"
            )
        return ctor(opt)


# ── Per-kind factories ──────────────────────────────────────────────


def _hp(g: Mapping[str, object], key: str, default: float) -> float:
    """Extract a float hyperparameter with the named default."""
    v = g.get(key, default)
    if v is None:
        return default
    if not isinstance(v, (int, float, bool)):
        raise TypeError(
            f"hyperparameter {key!r} must be numeric, got {type(v).__name__}"
        )
    return float(v)


def _sgd_spec(opt: Optimizer) -> OptimizerSpec:
    g = opt.param_groups[0]
    has_mom = _hp(g, "momentum", 0.0) != 0.0
    return OptimizerSpec(
        kind=OptimizerKind.SGD,
        lr=_hp(g, "lr", 0.0),
        momentum=_hp(g, "momentum", 0.0),
        dampening=_hp(g, "dampening", 0.0),
        weight_decay=_hp(g, "weight_decay", 0.0),
        nesterov=bool(g.get("nesterov", False)),
        state_buffer_names=("mom",) if has_mom else (),
    )


def _adam_spec(opt: Optimizer, *, adamw: bool = False) -> OptimizerSpec:
    g = opt.param_groups[0]
    return OptimizerSpec(
        kind=OptimizerKind.ADAMW if adamw else OptimizerKind.ADAM,
        lr=_hp(g, "lr", 0.0),
        beta1=_hp(g, "beta1", 0.9),
        beta2=_hp(g, "beta2", 0.999),
        eps=_hp(g, "eps", 1e-8),
        weight_decay=_hp(g, "weight_decay", 0.0),
        state_buffer_names=("m", "v"),
        scalar_names=("bias1", "bias2"),
    )


def _adamw_spec(opt: Optimizer) -> OptimizerSpec:
    return _adam_spec(opt, adamw=True)


def _rmsprop_spec(opt: Optimizer) -> OptimizerSpec:
    g = opt.param_groups[0]
    has_mom = _hp(g, "momentum", 0.0) != 0.0
    return OptimizerSpec(
        kind=OptimizerKind.RMSPROP,
        lr=_hp(g, "lr", 0.01),
        momentum=_hp(g, "momentum", 0.0),
        weight_decay=_hp(g, "weight_decay", 0.0),
        beta2=_hp(g, "alpha", 0.99),  # RMSprop calls β₂ "alpha"
        eps=_hp(g, "eps", 1e-8),
        state_buffer_names=("square_avg",) + (("mom",) if has_mom else ()),
    )


def _adagrad_spec(opt: Optimizer) -> OptimizerSpec:
    g = opt.param_groups[0]
    return OptimizerSpec(
        kind=OptimizerKind.ADAGRAD,
        lr=_hp(g, "lr", 0.01),
        weight_decay=_hp(g, "weight_decay", 0.0),
        eps=_hp(g, "eps", 1e-10),
        state_buffer_names=("sum",),
    )


def _adadelta_spec(opt: Optimizer) -> OptimizerSpec:
    g = opt.param_groups[0]
    return OptimizerSpec(
        kind=OptimizerKind.ADADELTA,
        lr=_hp(g, "lr", 1.0),
        weight_decay=_hp(g, "weight_decay", 0.0),
        beta2=_hp(g, "rho", 0.9),  # Adadelta calls β₂ "rho"
        eps=_hp(g, "eps", 1e-6),
        state_buffer_names=("square_avg", "acc_delta"),
    )


def _adamax_spec(opt: Optimizer) -> OptimizerSpec:
    g = opt.param_groups[0]
    return OptimizerSpec(
        kind=OptimizerKind.ADAMAX,
        lr=_hp(g, "lr", 0.002),
        beta1=_hp(g, "beta1", 0.9),
        beta2=_hp(g, "beta2", 0.999),
        eps=_hp(g, "eps", 1e-8),
        weight_decay=_hp(g, "weight_decay", 0.0),
        state_buffer_names=("m", "u"),  # u = infinity-norm running max
        scalar_names=("bias1",),  # only β₁ correction; β₂ branch uses max
    )


def _nadam_spec(opt: Optimizer) -> OptimizerSpec:
    g = opt.param_groups[0]
    return OptimizerSpec(
        kind=OptimizerKind.NADAM,
        lr=_hp(g, "lr", 0.002),
        beta1=_hp(g, "beta1", 0.9),
        beta2=_hp(g, "beta2", 0.999),
        eps=_hp(g, "eps", 1e-8),
        weight_decay=_hp(g, "weight_decay", 0.0),
        state_buffer_names=("m", "v"),
        scalar_names=("bias1", "bias2", "momentum_t"),
    )


_CLASS_NAME_TO_FACTORY: dict[str, Callable[[Optimizer], OptimizerSpec]] = {
    "SGD": _sgd_spec,
    "Adam": _adam_spec,
    "AdamW": _adamw_spec,
    "RMSprop": _rmsprop_spec,
    "Adagrad": _adagrad_spec,
    "Adadelta": _adadelta_spec,
    "Adamax": _adamax_spec,
    "NAdam": _nadam_spec,
}
