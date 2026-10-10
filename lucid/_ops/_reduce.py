"""The numerics the norm and variance reductions share.

``var`` / ``std`` (registry adapters), ``var_mean`` / ``std_mean``,
``linalg.vector_norm`` / ``linalg.norm``, ``cdist`` and
``nn.functional.pairwise_distance`` all end in the same three places, and
each used to get one of them wrong on its own:

* **the root of a sum** — ``sqrt(sum)`` and ``sum ** (1/p)`` have an
  infinite slope at zero, so a zero norm or a constant slice's ``std``
  sent ``inf * 0 = nan`` back through autograd.  :func:`_zero_safe_root`
  gives zero there, the subgradient the reference picks;
* **the divisor of a variance** — ``max(N - correction, 0)``, so too few
  elements answer ``nan`` (no spread to divide) or ``inf``, never a
  variance that silently skipped its correction (:func:`_bessel_correct`);
* **the axes** — a dim named twice, or one that is not an integer, is
  refused in the called op's name, rather than counted twice in ``N``
  (:func:`_reduction_axes`; every reduction adapter takes its axes here).

The four variance functions also read their arguments in one place,
:func:`_variance_operands`.
"""

import math
import operator
from collections.abc import Sequence
from typing import SupportsIndex

from lucid._C import engine as _C_engine
from lucid._deprecation import warn_deprecated
from lucid._dtype import _ENGINE_TO_DTYPE

_Impl = _C_engine.TensorImpl


# ── Axes ─────────────────────────────────────────────────────────────────────


def _to_axes(dim: object, name: str) -> list[int]:
    """``dim`` — ``None``, an index or a sequence of them — as engine axes."""
    if dim is None:
        return []
    if type(dim) is int:
        return [dim]
    # Plain ints in a tuple or list skip the protocol checks below, which
    # cost more than the reduction they guard.
    if type(dim) is tuple or type(dim) is list:
        axes = list(dim)
        for axis in axes:
            if type(axis) is not int:
                break
        else:
            return axes
    try:
        if isinstance(dim, SupportsIndex):
            return [operator.index(dim)]
        if isinstance(dim, Sequence) and not isinstance(dim, str):
            return [operator.index(d) for d in dim]
    except TypeError:
        pass  # an element that is not an index; refused below with the whole dim
    raise TypeError(
        f"{name}(): dim must be an int, a sequence of ints or None, got {dim!r}"
    )


def _reduction_axes(x_impl: _Impl, dim: object, name: str) -> list[int]:
    """``dim`` as engine axes, refusing a non-integer one or an axis named twice.

    The engine reduces a repeated axis once, so ``sum`` gave the right
    number and ``var`` the wrong one — its element count multiplied the
    axis in twice.  The reference refuses the call; so does this, in the
    name of the public op that was called.  A float used to truncate
    (``int(1.5)``) or fail as "not iterable".  An axis out of range is left
    for the engine to report.
    """
    axes = _to_axes(dim, name)
    if len(axes) > 1:
        ndim = len(x_impl.shape)
        if len({axis + ndim if axis < 0 else axis for axis in axes}) != len(axes):
            canonical = [axis + ndim if axis < 0 else axis for axis in axes]
            repeated = next(a for a in canonical if canonical.count(a) > 1)
            raise ValueError(
                f"{name}(): dim {repeated} appears multiple times in the "
                f"list of dims {tuple(axes)}"
            )
    return axes


# ── Roots and divisors ───────────────────────────────────────────────────────


def _zero_safe_root(s: _Impl, p: float) -> _Impl:
    """``s ** (1/p)`` for ``s >= 0``, with a zero gradient where ``s`` is 0.

    The root's slope is infinite at zero and the sum's gradient there is
    zero, so autograd multiplied them into ``nan``.  Both branches here are
    finite: the root is taken of 1 where ``s`` is 0, and that branch's
    result is then dropped for 0.  A ``nan`` or ``inf`` sum is not 0 and
    passes through as itself.  The values are the plain root's, so a sum
    no gradient will reach takes the plain root.
    """
    if not s.requires_grad:
        return _C_engine.sqrt(s) if p == 2 else _C_engine.pow_scalar(s, 1.0 / p)
    at_zero = _C_engine.equal(s, _C_engine.zeros_like(s))
    safe = _C_engine.where(at_zero, _C_engine.ones_like(s), s)
    # The root of an integer sum is floating, so its zero is taken from it.
    root = _C_engine.sqrt(safe) if p == 2 else _C_engine.pow_scalar(safe, 1.0 / p)
    return _C_engine.where(at_zero, _C_engine.zeros_like(root), root)


#: A set rather than a tuple: the membership test is on every norm's path.
_HALF = frozenset({_C_engine.F16, _C_engine.BF16})

#: The dtypes a norm is defined on.
_NORMABLE = frozenset(
    {
        _C_engine.F16,
        _C_engine.BF16,
        _C_engine.F32,
        _C_engine.F64,
        _C_engine.C64,
        _C_engine.C128,
    }
)


def _require_normable(impl: _Impl, op: str) -> _Impl:
    """``impl``, or a refusal if it is an integer or bool input to the norm ``op``.

    The integer path answered by accident, differently per order: ``ord``
    2 and 0 came back float32, 1 and ``inf`` in the input's own dtype, and
    every other order raised from deep inside ``pow_scalar``.  A
    ``DtypeMismatch`` is a ``TypeError`` and a ``RuntimeError``, the class
    the reference raises.  ``dtype=`` does not lift the refusal: the
    reference checks the input before any cast.
    """
    if impl.dtype not in _NORMABLE:
        raise _C_engine.DtypeMismatch(
            f"{op}: expected a floating point or complex tensor as input, "
            f"got {_ENGINE_TO_DTYPE[impl.dtype]}"
        )
    return impl


#: The orders that reduce the magnitudes directly, with no power or root.
_ABS_REDUCTIONS = {
    math.inf: _C_engine.max,
    -math.inf: _C_engine.min,
    1.0: _C_engine.sum,
}


def _p_norm(r: _Impl, p: float, axes: list[int], keepdim: bool) -> _Impl:
    """``(sum |r|^p)^(1/p)`` over ``axes`` for a real ``r`` and ``p != 0``.

    Its gradient is zero wherever the norm is — the subgradient the
    reference picks — and nowhere ``nan``.  The root's slope is infinite at
    a zero sum (:func:`_zero_safe_root`), and for ``p < 1`` so is
    ``|r|^p``'s at a zero entry; both used to meet a zero factor in
    autograd and leave ``nan`` behind.

    A half input reduces in float32 and answers in its own dtype, as the
    reference's op-math does: the sum of squares of 65 536 ones is already
    past float16's largest finite value, so the norm 256 came back ``inf``.
    """
    if r.dtype in _HALF:
        out = _p_norm(_C_engine.astype(r, _C_engine.F32), p, axes, keepdim)
        return _C_engine.astype(out, r.dtype)
    if p in _ABS_REDUCTIONS:
        return _ABS_REDUCTIONS[p](_C_engine.abs(r), axes, keepdim)
    if p == 2:
        # r² = |x|² for a real r, one product cheaper than ``abs`` first.
        return _zero_safe_root(_C_engine.sum(_C_engine.mul(r, r), axes, keepdim), 2)
    magnitude = _C_engine.abs(r)
    if p > 1 or not magnitude.requires_grad:
        powered = _C_engine.pow_scalar(magnitude, p)
    else:
        # A zero entry contributes 0 to the sum (``inf`` for p < 0, which
        # makes the norm 0) and nothing to the gradient.
        nonzero = _C_engine.not_equal(magnitude, _C_engine.zeros_like(magnitude))
        safe = _C_engine.where(nonzero, magnitude, _C_engine.ones_like(magnitude))
        raised = _C_engine.pow_scalar(safe, p)
        fill = _C_engine.full_like(raised, 0.0 if p > 0 else math.inf)
        powered = _C_engine.where(nonzero, raised, fill)
    s = _C_engine.sum(powered, axes, keepdim)
    if p > 0:
        return _zero_safe_root(s, p)
    return _C_engine.pow_scalar(s, 1.0 / p)


def _bessel_correct(
    result_impl: _Impl,
    x_impl: _Impl,
    axes_list: Sequence[int],
    correction: float,
) -> _Impl:
    """Turn the engine's ``sum / N`` variance into ``sum / max(N - correction, 0)``.

    With ``N <= correction`` the divisor is 0, as the reference has it: a
    slice with no spread answers ``0 / 0 = nan`` and one with spread
    ``inf``.  The correction used to be skipped there instead, so
    ``var([5.])`` answered 0.  An empty reduction is ``nan`` for any
    correction — the engine's CPU kernel answers 0 for it, Metal ``nan``,
    and scaling by ``inf`` sends both to ``nan``.

    With ``N < correction`` the reference's gradient still divides by the
    negative ``N - correction``, so it is finite under a ``nan`` / ``inf``
    value.  The value here is the forward's and the slope is that one, so
    a loss summed over many slices is not poisoned by a ``nan`` gradient
    the reference does not give.
    """
    shape = x_impl.shape
    n = math.prod([shape[a] for a in axes_list] if axes_list else shape)
    if correction == 0 and n > 0:
        return result_impl
    dof = n - correction
    if dof > 0:
        return _scale(result_impl, n / dof)
    undivided = _scale(result_impl, math.inf)
    if dof == 0 or not result_impl.requires_grad:
        return undivided
    slope = _scale(result_impl, n / dof)
    detached = _C_engine.sub(undivided.data_alias(), slope.data_alias())
    return _C_engine.add(slope, detached)


def _scale(impl: _Impl, factor: float) -> _Impl:
    """``impl * factor``, the factor a 0-d constant of ``impl``'s dtype and device.

    0-d rather than ``impl``'s shape: it broadcasts, and a full-size
    constant cost an allocation and a fill the size of the result.
    """
    factor_t = _C_engine.full([], factor, impl.dtype, impl.device)
    return _C_engine.mul(impl, factor_t)


# ── var / std arguments ──────────────────────────────────────────────────────

#: ``var`` / ``std`` read a third positional argument as ``keepdim`` until
#: 3.16.  The reference reads it as ``unbiased`` — ``var(x, dim, unbiased,
#: keepdim)`` — and the two cannot be told apart by type, both being bools.
#: The old reading still answers, and warns, until the release named here;
#: then the third positional argument becomes ``unbiased``.
_VAR_KEEPDIM_POSITIONAL = (
    "passing keepdim as the third positional argument of var()/std()"
)
_VAR_SINCE = "3.16.0"
_VAR_REMOVAL = "3.18.0"

_VAR_KEYWORDS = frozenset({"dim", "unbiased", "keepdim", "correction"})


def _as_correction(name: str, value: object) -> float:
    """``value`` as a ``correction``: a real number, a bool as 1 or 0."""
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(
        f"{name}(): correction must be a number, got {type(value).__name__}"
    )


def _variance_operands(
    name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    *,
    keepdim_third: bool,
) -> tuple[object, float, bool]:
    """``(dim, correction, keepdim)`` from any spelling of a variance call.

    The one reading of ``var`` / ``std`` / ``var_mean`` / ``std_mean``'s
    arguments.  The reference spellings are ``f(input, dim, unbiased,
    keepdim)``, ``f(input, unbiased)`` and keywords for the rest, so a bool
    where the ``dim`` goes is ``unbiased`` — it used to reduce axis 0 or 1.
    A number rather than a bool after ``dim`` is a ``correction``, as
    ``var_mean(x, dim, correction)`` has always been spelled.

    ``keepdim_third`` is ``var`` / ``std``'s released ``(dim, keepdim)``
    spelling: with it, exactly two arguments after ``input`` still answer
    that way and warn, until :data:`_VAR_REMOVAL`.
    """
    unknown = sorted(set(kwargs) - _VAR_KEYWORDS)
    if unknown:
        raise TypeError(f"{name}() got an unexpected keyword argument {unknown[0]!r}")
    if len(args) > 3:
        raise TypeError(
            f"{name}() takes at most 4 positional arguments but "
            f"{len(args) + 1} were given"
        )
    names: tuple[str, ...]
    if args and isinstance(args[0], bool):
        if len(args) > 1:
            raise TypeError(
                f"{name}(input, unbiased) takes no further positional arguments; "
                f"pass dim and keepdim by keyword"
            )
        names = ("unbiased",)
    elif len(args) == 2 and keepdim_third:
        warn_deprecated(
            _VAR_KEEPDIM_POSITIONAL,
            since=_VAR_SINCE,
            removal=_VAR_REMOVAL,
            alternative=f"{name}(input, dim, keepdim=...)",
        )
        names = ("dim", "keepdim")
    else:
        bias = (
            "unbiased" if len(args) < 2 or isinstance(args[1], bool) else "correction"
        )
        names = ("dim", bias, "keepdim")[: len(args)]
    bound = dict(kwargs)
    for key, value in zip(names, args, strict=True):
        if key in bound:
            raise TypeError(f"{name}() got multiple values for argument {key!r}")
        bound[key] = value
    unbiased = bound.get("unbiased")
    given = bound.get("correction")
    if unbiased is not None:
        correction = 1.0 if unbiased else 0.0
    else:
        correction = 1.0 if given is None else _as_correction(name, given)
    return bound.get("dim"), correction, bool(bound.get("keepdim", False))


def _corrected_variance(
    x_impl: _Impl, axes: list[int], correction: float, keepdim: bool
) -> _Impl:
    """The variance over ``axes`` divided by ``max(N - correction, 0)``."""
    biased = _C_engine.var(x_impl, axes, keepdim)
    return _bessel_correct(biased, x_impl, axes, correction)


def _variance(
    name: str, x_impl: _Impl, args: tuple[object, ...], kwargs: dict[str, object]
) -> _Impl:
    """The corrected variance a ``var`` / ``std`` call asks for."""
    if not kwargs and len(args) < 2 and not (args and isinstance(args[0], bool)):
        # ``x.var()`` / ``x.var(dim)``: nothing to parse.
        dim: object = args[0] if args else None
        correction, keepdim = 1.0, False
    else:
        dim, correction, keepdim = _variance_operands(
            name, args, kwargs, keepdim_third=True
        )
    axes = _reduction_axes(x_impl, dim, name)
    return _corrected_variance(x_impl, axes, correction, keepdim)


def _var_adapter(input: _Impl, *args: object, **kwargs: object) -> _Impl:
    """var(input, dim=None, unbiased=None, keepdim=False, *, correction=1).

    The parameters after ``input`` stay open because a third positional
    argument binds by count, not by name, during the ``keepdim``
    deprecation — see :func:`_variance_operands`.
    """
    return _variance("var", input, args, kwargs)


def _std_adapter(input: _Impl, *args: object, **kwargs: object) -> _Impl:
    """std(input, dim=None, unbiased=None, keepdim=False, *, correction=1).

    The square root of :func:`_var_adapter`'s variance, with a zero
    gradient where that variance is 0 (a constant slice).
    """
    return _zero_safe_root(_variance("std", input, args, kwargs), 2)
