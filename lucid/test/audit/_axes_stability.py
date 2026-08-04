"""Numerical stability, memory layout and reproducibility.

These are the axes that do not ask "is the formula right" but "does the
implementation survive the inputs a real run produces".  Each has a
precedent in this framework:

* ``log_softmax`` once computed ``log(softmax(x))`` and returned ``-inf``
  past ``|logit| ~ 90``, because the non-maximum probabilities round to
  zero before the log sees them.  Found while training a network, not by
  a unit test, because every unit test used well-conditioned logits.
* the transforms and ``pdist`` bugs were layout bugs: an index tensor
  built one way worked and another way did not.
* a sampler that is not reproducible under a fixed seed makes every
  other check in this tool unfalsifiable.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

import functools
import json
import pathlib

import lucid
from lucid.test.audit import _probe, _specs, _surface
from lucid.test.audit._axes import Axis, Context
from lucid.test.audit._result import Finding, Status

if TYPE_CHECKING:
    from lucid.test.audit._surface import Symbol


@functools.lru_cache(maxsize=1)
def _stability_contract() -> "dict[str, list[str]]":
    """Where the reference answers finitely, per symbol.

    Checked in, so the audit reads it without the reference installed.
    Regenerate with
    ``python -m lucid.test.audit.tools.stability_contract``.
    """
    path = pathlib.Path(__file__).with_name("stability_contract.json")
    try:
        data = json.loads(path.read_text())
    except OSError, ValueError:
        return {}
    table = data.get("finite_at")
    return table if isinstance(table, dict) else {}


class StabilityAxis(Axis):
    """A finite, in-domain input must not produce a NaN.

    Overflow to infinity is allowed — ``exp(1e8)`` is genuinely infinite
    — but a NaN out of finite input means an intermediate cancelled,
    divided by its own zero, or overflowed into a subtraction.  That is
    the shape of every stability bug worth finding.
    """

    name = "stability"
    summary = "finite in-domain inputs across 14 orders of magnitude"
    kinds = frozenset({"op", "method"})

    _SCALES = (1e-30, 1e-15, 1e-6, 1.0, 1e6, 1e15, 1e30)

    def applies(self, symbol: "Symbol") -> bool:
        return super().applies(symbol) and "stochastic" not in symbol.flags

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")
        call, _, _ = self._working_call(fn, symbol, ctx)
        if call is None:
            return self._finding(symbol, Status.SKIP, "no candidate invocation ran")
        try:
            base = np.abs(call.base) + 0.25
        except TypeError:
            return self._finding(
                symbol, Status.SKIP, "primary argument is not a tensor"
            )
        # A scalar operand is not necessarily a *value*.  ``eye(n)`` takes
        # a size, and multiplying it by 1e15 asks for a matrix of 1e30
        # elements — the process was killed by the OOM killer, signal 9,
        # and the sweep stopped there.  Scaling only means something for an
        # operand that carries data, so a bare scalar is skipped rather
        # than magnified.
        if base.ndim == 0 or base.size == 1:
            return self._finding(
                symbol,
                Status.SKIP,
                "primary is a scalar — scaling it would change a size, not a value",
            )

        # Where the reference is finite.  Absent means "not measured", and
        # an unmeasured op is checked at every scale as before — the table
        # narrows the claim, it does not gate it.
        reference_finite = _stability_contract().get(symbol.qualname)

        # What the check is actually about: does *changing the magnitude*
        # introduce a NaN?  A NaN that was there before anything was scaled
        # belongs to the probe, not to the op's conditioning.
        #
        # Only the primary argument is made positive.  Every companion
        # keeps whatever the domain drew, and ``moderate`` draws from
        # (-1.05, 1.35) — so ``xlogy(x, y)`` met a negative ``y``, answered
        # NaN for it at every scale including 1.0, and was reported for
        # arithmetic that was never in question.  Five findings, all of
        # them the second operand.
        #
        # Held per element rather than per array, so an op that is NaN
        # somewhere for that reason can still be caught going NaN
        # elsewhere under scaling.
        baseline_finite = None
        try:
            at_one = _probe.to_numpy(fn(*call.with_primary(base).args, **call.kwargs))
            if at_one is not None and at_one.dtype.kind in "fc":
                baseline_finite = ~np.isnan(at_one)
        except Exception:  # noqa: BLE001 - surveying, not asserting
            baseline_finite = None

        broken: list[str] = []
        domain: list[str] = []
        ran = 0
        for scale in self._SCALES:
            probe = base * scale
            if not np.isfinite(probe).all():
                continue
            # ``in-domain`` is the load-bearing word in this check, and the
            # axis had no way to know where the domain was: it reported
            # ``acos`` for answering NaN at 1e+6, where NaN is the only
            # answer the function has.  A NaN the reference also produces
            # is the domain speaking, not the implementation.
            if reference_finite is not None and f"{scale:g}" not in reference_finite:
                domain.append(f"1e{int(np.log10(scale)):+d}")
                continue
            try:
                out = _probe.to_numpy(fn(*call.with_primary(probe).args, **call.kwargs))
            except Exception:  # noqa: BLE001 - a refusal is not instability
                continue
            if out is None or out.dtype.kind not in "fc":
                continue
            ran += 1
            appeared = np.isnan(out)
            if baseline_finite is not None and baseline_finite.shape == appeared.shape:
                appeared = appeared & baseline_finite
            if appeared.any():
                broken.append(f"scale 1e{int(np.log10(scale)):+d} -> NaN")
        if not ran:
            if domain:
                # Every scale left the function's domain.  ``acosh`` on a
                # probe below 1 is the whole check falling outside where
                # the op is defined — nothing was asked, so nothing passed.
                return self._finding(
                    symbol,
                    Status.NOT_APPLICABLE,
                    "every scale is outside the domain: " + ", ".join(domain[:4]),
                )
            return self._finding(
                symbol, Status.SKIP, "no scale produced a float output"
            )
        if broken:
            return self._finding(
                symbol,
                Status.FAIL,
                "finite positive input produced NaN: " + ", ".join(broken[:4]),
                scales=broken,
            )
        note = f"{ran}/{len(self._SCALES)} scales finite"
        if domain:
            note += f" ({len(domain)} outside the domain)"
        return self._finding(symbol, Status.PASS, note)


class ExtremeValueAxis(Axis):
    """The specific limits where a naive formula is known to break.

    Generic scaling cannot express "softmax of a thousand still sums to
    one"; these are hand-written because each one is a property with a
    known answer, and each is a bug this class of code actually has.
    """

    name = "extreme"
    summary = "known-hard limits: saturated softmax, tiny log1p, huge logsumexp"
    kinds = frozenset({"op"})

    def applies(self, symbol: "Symbol") -> bool:
        return symbol.short in _EXTREME_CASES and symbol.inert

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")
        failures: list[str] = []
        checked = 0
        for label, build, verify in _EXTREME_CASES[symbol.short]:
            try:
                out = fn(*build())
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{label}: raised {type(exc).__name__}")
                continue
            got = _probe.to_numpy(out)
            if got is None:
                failures.append(f"{label}: no tensor")
                continue
            checked += 1
            problem = verify(np.asarray(got, dtype=np.float64))
            if problem:
                failures.append(f"{label}: {problem}")
        if failures:
            return self._finding(
                symbol, Status.FAIL, "; ".join(failures[:3]), failures=failures
            )
        if not checked:
            return self._finding(symbol, Status.SKIP, "no case ran")
        return self._finding(symbol, Status.PASS, f"{checked} extreme cases")


def _f64(values: Any) -> Any:
    return _probe.as_f64(np.asarray(values, dtype=np.float64))


def _finite(name: str) -> "Any":
    def check(out: np.ndarray) -> str:
        if not np.isfinite(out).all():
            return f"{name} is not finite: {out.reshape(-1)[:4]}"
        return ""

    return check


def _close_to(want: float, tol: float = 1e-6) -> "Any":
    def check(out: np.ndarray) -> str:
        got = float(np.asarray(out).reshape(-1)[0])
        if not np.isfinite(got) or abs(got - want) > tol * max(abs(want), 1.0):
            return f"expected ~{want}, got {got}"
        return ""

    return check


def _sums_to_one(out: np.ndarray) -> str:
    total = float(np.sum(out))
    if not np.isfinite(total) or abs(total - 1.0) > 1e-9:
        return f"probabilities sum to {total}"
    return ""


#: ``name -> [(label, build_args, verify)]``.  ``build_args`` returns the
#: positional arguments; ``verify`` returns "" for pass or a reason.
_EXTREME_CASES: dict[str, list[tuple[str, Any, Any]]] = {
    "softmax": [
        ("logits +1000", lambda: (_f64([1000.0, 999.0, 998.0]),), _sums_to_one),
        ("logits -1000", lambda: (_f64([-1000.0, -999.0, -998.0]),), _sums_to_one),
        ("wide spread", lambda: (_f64([-800.0, 0.0, 800.0]),), _sums_to_one),
    ],
    "log_softmax": [
        (
            "logits +1000",
            lambda: (_f64([1000.0, 999.0, 998.0]),),
            _finite("log-probabilities"),
        ),
        (
            "wide spread",
            lambda: (_f64([-800.0, 0.0, 800.0]),),
            _finite("log-probabilities"),
        ),
    ],
    "logsumexp": [
        (
            "all +1000",
            lambda: (_f64([1000.0, 1000.0]),),
            _close_to(1000.0 + float(np.log(2.0))),
        ),
        (
            "all -1000",
            lambda: (_f64([-1000.0, -1000.0]),),
            _close_to(-1000.0 + float(np.log(2.0))),
        ),
    ],
    "sigmoid": [
        ("+1000", lambda: (_f64([1000.0]),), _close_to(1.0)),
        ("-1000", lambda: (_f64([-1000.0]),), _close_to(0.0)),
    ],
    "tanh": [
        ("+1000", lambda: (_f64([1000.0]),), _close_to(1.0)),
        ("-1000", lambda: (_f64([-1000.0]),), _close_to(-1.0)),
    ],
    "softplus": [
        ("+1000 must not overflow", lambda: (_f64([1000.0]),), _close_to(1000.0, 1e-9)),
        (
            "-1000 must not underflow to -inf",
            lambda: (_f64([-1000.0]),),
            _finite("softplus"),
        ),
    ],
    "log1p": [
        (
            "1e-16 must not round to zero",
            lambda: (_f64([1e-16]),),
            _close_to(1e-16, 1e-3),
        ),
    ],
    "expm1": [
        (
            "1e-16 must not round to zero",
            lambda: (_f64([1e-16]),),
            _close_to(1e-16, 1e-3),
        ),
    ],
    "logaddexp": [
        (
            "both +1000",
            lambda: (_f64([1000.0]), _f64([1000.0])),
            _close_to(1000.0 + float(np.log(2.0))),
        ),
    ],
    "logsigmoid": [
        ("-1000", lambda: (_f64([-1000.0]),), _close_to(-1000.0, 1e-9)),
        ("+1000", lambda: (_f64([1000.0]),), _finite("logsigmoid")),
    ],
    "norm": [
        ("1e200 must not overflow", lambda: (_f64([1e200, 1e200]),), _finite("norm")),
        (
            "1e-200 must not underflow",
            lambda: (_f64([1e-200, 1e-200]),),
            _finite("norm"),
        ),
    ],
    "hypot": [
        (
            "1e200 must not overflow",
            lambda: (_f64([1e200]), _f64([1e200])),
            _finite("hypot"),
        ),
    ],
    "var": [
        (
            "large mean, small spread",
            lambda: (_f64([1e8, 1e8 + 1.0, 1e8 + 2.0]),),
            _close_to(1.0, 1e-3),
        ),
    ],
    "std": [
        (
            "large mean, small spread",
            lambda: (_f64([1e8, 1e8 + 1.0, 1e8 + 2.0]),),
            _close_to(1.0, 1e-3),
        ),
    ],
}


class ContiguityAxis(Axis):
    """A non-contiguous input must give the same answer as a packed one.

    Every layout bug this framework has had looked like this: the same
    values, reached through different strides, produced different
    results.  The probe is a strided view rather than a transpose so the
    logical shape is unchanged and only the memory walk differs.
    """

    name = "layout"
    summary = "strided (non-contiguous) inputs agree with packed ones"
    kinds = frozenset({"op", "method"})

    def applies(self, symbol: "Symbol") -> bool:
        return super().applies(symbol) and "stochastic" not in symbol.flags

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")
        call, _, _ = self._working_call(fn, symbol, ctx)
        if call is None:
            return self._finding(symbol, Status.SKIP, "no candidate invocation ran")
        if self._draws_randomly(fn, call):
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "two identical calls disagree — this measures the draw, not the op",
            )
        try:
            base = call.base
        except TypeError:
            return self._finding(
                symbol, Status.SKIP, "primary argument is not a tensor"
            )
        if base.ndim < 1:
            return self._finding(symbol, Status.SKIP, "0-d input has no layout to vary")

        # A tensor twice the size in the last axis, whose every second
        # column holds the probe.  Slicing it back gives the same values
        # through a different stride.
        padded = np.repeat(base, 2, axis=-1)
        padded[..., 1::2] = -7.0
        try:
            big = _probe.as_f64(padded)
            view = big[..., ::2]
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.SKIP, f"could not build a view: {exc!r}"
            )

        try:
            packed = _probe.to_numpy(fn(*call.with_primary(base).args, **call.kwargs))
            args = list(call.args)
            args[call.primary] = view
            strided = _probe.to_numpy(fn(*args, **call.kwargs))
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
            )
        if packed is None or strided is None or packed.shape != strided.shape:
            return self._finding(symbol, Status.SKIP, "outputs not comparable")
        if not np.allclose(
            packed.astype(float),
            strided.astype(float),
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
        ):
            return self._finding(
                symbol,
                Status.FAIL,
                "a strided view of the same values gave a different answer "
                f"(max diff {np.nanmax(np.abs(packed.astype(float) - strided.astype(float))):.3e})",
            )
        return self._finding(symbol, Status.PASS, "strided and packed agree")


class DeterminismAxis(Axis):
    """The same seed must give the same numbers.

    Only meaningful for the ops every other axis has to skip — the ones
    that draw.  Without this they would be entirely unchecked, and a
    sampler that ignores the seed makes every seeded test in the
    repository meaningless rather than merely wrong.
    """

    name = "determinism"
    summary = "stochastic ops reproduce under a fixed seed"
    kinds = frozenset({"op", "method"})

    def applies(self, symbol: "Symbol") -> bool:
        if symbol.kind not in self.kinds or not symbol.inert:
            return False
        return "stochastic" in symbol.flags

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")

        first: np.ndarray | None = None
        for domain in ctx.domains:
            for call in _specs.invocations(symbol.short, domain, symbol.qualname, fn):
                draws = []
                for _ in range(2):
                    lucid.manual_seed(1234)
                    try:
                        got = _probe.to_numpy(fn(*call.args, **call.kwargs))
                    except Exception:  # noqa: BLE001
                        draws = []
                        break
                    if got is None:
                        draws = []
                        break
                    draws.append(np.asarray(got, dtype=np.float64))
                if len(draws) == 2:
                    first = draws[0]
                    if not np.allclose(draws[0], draws[1], equal_nan=True):
                        return self._finding(
                            symbol,
                            Status.FAIL,
                            "two draws under the same seed differ by "
                            f"{np.nanmax(np.abs(draws[0] - draws[1])):.3e}",
                        )
                    # Guard the instrument: an op that ignores its inputs
                    # and returns a constant would pass trivially.
                    lucid.manual_seed(4321)
                    other = _probe.to_numpy(fn(*call.args, **call.kwargs))
                    if other is not None and np.allclose(
                        first, np.asarray(other, dtype=np.float64), equal_nan=True
                    ):
                        return self._finding(
                            symbol,
                            Status.VACUOUS,
                            "a different seed gave identical numbers — this op may not draw",
                        )
                    return self._finding(
                        symbol, Status.PASS, "reproducible and seed-sensitive"
                    )
        return self._finding(symbol, Status.SKIP, "no candidate invocation ran")


STABILITY_AXES: tuple[Axis, ...] = (
    StabilityAxis(),
    ExtremeValueAxis(),
    ContiguityAxis(),
    DeterminismAxis(),
)

__all__ = [
    "STABILITY_AXES",
    "ContiguityAxis",
    "DeterminismAxis",
    "ExtremeValueAxis",
    "StabilityAxis",
]
