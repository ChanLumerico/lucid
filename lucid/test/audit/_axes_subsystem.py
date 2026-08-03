"""Axes for the subsystems the numeric sweep cannot express.

``distributions``, ``diffeq``, ``quantization``, ``serialization`` and
``compile`` were enumerated in the denominator and had **no axis at
all** — 141 symbols counted and never checked, which is the exact shape
of the dishonesty this tool exists to remove.  Each gets the question
that is actually meaningful for it:

    distributions   does log_prob agree with the samples, is cdf monotone,
                    does icdf invert it
    diffeq          does halving the step reduce the error like h**p
    quantization    does quantize -> dequantize land inside one step
    serialization   does a tensor survive a round trip through disk
    compile         does the compiled function agree with the eager one

and :class:`SmokeAxis` catches everything left over, including the
stateful symbols every other axis refuses to touch — those run inside a
guard that snapshots and restores the global state, so they are checked
rather than merely counted.
"""

import math
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import lucid
from lucid.test.audit import _probe, _surface
from lucid.test.audit._axes import Axis, Context, _try_construct
from lucid.test.audit._result import Finding, Status

if TYPE_CHECKING:
    from types import TracebackType

    from lucid.test.audit._surface import Symbol


class StateGuard:
    """Snapshot the process-wide state, restore it on the way out.

    What makes a stateful symbol callable at all.  Without this the
    survey either skips them — leaving 100 symbols permanently unchecked
    — or calls them and poisons every op that follows, which is exactly
    what a first version did to 278 of them.
    """

    _KEYS = (
        ("is_grad_enabled", "set_grad_enabled"),
        ("get_default_dtype", "set_default_dtype"),
        ("get_default_device", "set_default_device"),
        ("get_num_threads", "set_num_threads"),
        ("get_rng_state", "set_rng_state"),
    )

    def __init__(self) -> None:
        self._saved: list[tuple[Any, Any]] = []

    def __enter__(self) -> "StateGuard":
        for getter_name, setter_name in self._KEYS:
            getter = getattr(lucid, getter_name, None)
            setter = getattr(lucid, setter_name, None)
            if getter is None or setter is None:
                continue
            try:
                self._saved.append((setter, getter()))
            except Exception:  # noqa: BLE001
                continue
        return self

    def __exit__(
        self,
        exc_type: "type[BaseException] | None",
        exc: "BaseException | None",
        tb: "TracebackType | None",
    ) -> None:
        for setter, value in reversed(self._saved):
            try:
                setter(value)
            except Exception:  # noqa: BLE001
                continue
        self._saved.clear()


class SmokeAxis(Axis):
    """Every callable gets invoked at least once and must not take the process with it.

    The weakest axis and the one that moves coverage most: a symbol no
    other axis can express is still a symbol that should not segfault,
    hang, or corrupt global state when called the obvious way.
    """

    name = "smoke"
    summary = "every callable is invoked once, stateful ones inside a state guard"
    #: Everything except the two kinds that already have a lifecycle axis
    #: of their own.  This is the floor: after it, no callable symbol in
    #: the framework is outside the audit.
    kinds = frozenset(
        {
            "op",
            "method",
            "class",
            "util",
            "compiled",
            "serialize",
            "quant",
            "distribution",
            "diffeq",
            "value",
        }
    )

    def applies(self, symbol: "Symbol") -> bool:
        if symbol.kind in ("module", "optim"):
            return False  # ModuleAxis / OptimAxis do this properly
        return callable(_surface.resolve(symbol))

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")

        guard = StateGuard() if not symbol.inert else _NullGuard()
        with guard:
            for args, kwargs, note in _smoke_arguments(symbol):
                try:
                    fn(*args, **kwargs)
                except Exception:  # noqa: BLE001 - a refusal is a fine answer here
                    continue
                return self._finding(symbol, Status.PASS, f"called as {note}")
            # Falling through to the op ladder covers anything with a
            # tensor-shaped signature the table above does not name.
            if symbol.inert:
                call, _, _ = self._working_call(fn, symbol, ctx)
                if call is not None:
                    return self._finding(symbol, Status.PASS, f"called as {call.note}")
        return self._finding(symbol, Status.SKIP, "no argument shape worked")


class _NullGuard:
    def __enter__(self) -> "_NullGuard":
        return self

    def __exit__(self, *_: Any) -> None:
        return None


def _smoke_arguments(
    symbol: "Symbol",
) -> "list[tuple[tuple[Any, ...], dict[str, Any], str]]":
    """Plausible calls for a symbol with no numeric spec.

    Stateful setters are handed back the value their own getter reports,
    so the call is a no-op that still exercises the code path.
    """
    name = symbol.short
    out: list[tuple[tuple[Any, ...], dict[str, Any], str]] = [((), {}, "no arguments")]

    if name.startswith("set_"):
        getter = getattr(lucid, "get_" + name[4:], None) or getattr(
            lucid, "is_" + name[4:], None
        )
        if getter is not None:
            try:
                out.insert(0, ((getter(),), {}, "its own getter's value"))
            except Exception:  # noqa: BLE001
                pass
    if "seed" in name:
        out.insert(0, ((0,), {}, "seed 0"))
    if "grad_enabled" in name or "deterministic" in name:
        out.insert(0, ((True,), {}, "True"))
    if "thread" in name:
        out.insert(0, ((1,), {}, "1"))

    tensor = _probe.as_f64(_probe.sample("positive", (2, 3)))
    out.extend(
        [
            ((tensor,), {}, "one tensor"),
            ((tensor, tensor), {}, "two tensors"),
            (((2, 3),), {}, "a shape"),
            ((2,), {}, "an int"),
        ]
    )
    return out


# ── distributions ────────────────────────────────────────────────────────────


class DistributionAxis(Axis):
    """Sample, score, and invert — the three must agree with each other.

    ``log_prob`` finite on the distribution's own samples is the check
    that catches a support mismatch; ``cdf`` monotone and ``icdf``
    inverting it catches a parameterisation that is off by a
    transformation.
    """

    name = "distribution"
    summary = "sample / log_prob / cdf / icdf consistency"
    kinds = frozenset({"distribution"})

    _PARAMS: tuple[tuple[tuple[Any, ...], dict[str, Any]], ...] = (
        ((), {}),
        ((0.5,), {}),
        ((0.0, 1.0), {}),
        ((1.0, 1.0), {}),
        ((2.0,), {}),
        ((3, 0.5), {}),
    )

    def applies(self, symbol: "Symbol") -> bool:
        if symbol.subsystem != "distributions":
            return False
        return isinstance(symbol.obj, type) or symbol.short == "kl_divergence"

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        if symbol.short == "kl_divergence":
            return self._check_kl(symbol)
        dist = None
        for args, kwargs in self._PARAMS:
            try:
                dist = symbol.obj(
                    *(_probe.as_f64(a) if isinstance(a, float) else a for a in args),
                    **kwargs,
                )
                break
            except Exception:  # noqa: BLE001
                continue
        if dist is None:
            return self._finding(symbol, Status.SKIP, "no constructor signature worked")

        problems: list[str] = []
        checked: list[str] = []

        draw = None
        for method in ("sample", "rsample"):
            fn = getattr(dist, method, None)
            if fn is None:
                continue
            try:
                draw = fn((64,))
            except Exception:  # noqa: BLE001
                try:
                    draw = fn()
                except Exception:  # noqa: BLE001
                    continue
            checked.append(method)
            break

        if draw is not None:
            values = _probe.to_numpy(draw)
            if values is not None and not np.isfinite(values).all():
                problems.append("sample produced a non-finite value")
            log_prob = getattr(dist, "log_prob", None)
            if log_prob is not None and values is not None:
                try:
                    scored = _probe.to_numpy(log_prob(draw))
                    checked.append("log_prob")
                    if scored is not None and not np.isfinite(scored).all():
                        problems.append(
                            "log_prob is not finite on the distribution's own samples"
                        )
                except Exception as exc:  # noqa: BLE001
                    problems.append(
                        f"log_prob raised {type(exc).__name__} on its own samples"
                    )

        cdf, icdf = getattr(dist, "cdf", None), getattr(dist, "icdf", None)
        if cdf is not None:
            grid = _probe.as_f64(np.linspace(-2.0, 2.0, 9))
            try:
                probs = _probe.to_numpy(cdf(grid))
                checked.append("cdf")
                if probs is not None:
                    flat = np.asarray(probs, dtype=np.float64).reshape(-1)
                    if np.any(np.diff(flat) < -1e-9):
                        problems.append("cdf is not monotone")
                    if flat.min() < -1e-9 or flat.max() > 1.0 + 1e-9:
                        problems.append(
                            f"cdf leaves [0, 1]: [{flat.min():.3f}, {flat.max():.3f}]"
                        )
                    if icdf is not None:
                        inside = _probe.as_f64(np.clip(flat, 1e-4, 1 - 1e-4))
                        back = _probe.to_numpy(icdf(inside))
                        checked.append("icdf")
                        if back is not None:
                            ok = np.isfinite(back)
                            if ok.any():
                                err = np.abs(
                                    np.asarray(back, dtype=np.float64).reshape(-1)[
                                        ok.reshape(-1)
                                    ]
                                    - np.asarray(_probe.to_numpy(grid)).reshape(-1)[
                                        ok.reshape(-1)
                                    ]
                                ).max()
                                if err > 1e-3:
                                    problems.append(f"icdf(cdf(x)) is off by {err:.2e}")
            except Exception:  # noqa: BLE001
                pass

        for stat in ("mean", "variance", "stddev", "entropy"):
            attr = getattr(dist, stat, None)
            if attr is None:
                continue
            try:
                value = attr() if callable(attr) else attr
                array = _probe.to_numpy(value)
                checked.append(stat)
                if array is not None and not np.isfinite(array).all():
                    problems.append(f"{stat} is not finite")
            except Exception:  # noqa: BLE001
                continue

        if problems:
            return self._finding(
                symbol, Status.FAIL, "; ".join(problems[:3]), problems=problems
            )
        if not checked:
            return self._finding(symbol, Status.SKIP, "no probeable method")
        return self._finding(
            symbol, Status.PASS, f"checked {', '.join(sorted(set(checked)))}"
        )

    def _check_kl(self, symbol: "Symbol") -> Finding:
        """A divergence from a distribution to itself is zero, and never negative."""
        normal = getattr(lucid.distributions, "Normal", None)
        if normal is None:
            return self._finding(symbol, Status.SKIP, "no Normal to compare")
        try:
            p = normal(_probe.as_f64(np.array(0.0)), _probe.as_f64(np.array(1.0)))
            q = normal(_probe.as_f64(np.array(1.5)), _probe.as_f64(np.array(2.0)))
            same = _probe.to_numpy(symbol.obj(p, p))
            apart = _probe.to_numpy(symbol.obj(p, q))
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
            )
        if same is None or apart is None:
            return self._finding(symbol, Status.SKIP, "no comparable output")
        if abs(float(np.asarray(same).reshape(-1)[0])) > 1e-9:
            return self._finding(
                symbol,
                Status.FAIL,
                f"KL(p, p) = {float(np.asarray(same).reshape(-1)[0])}, not 0",
            )
        if float(np.asarray(apart).reshape(-1)[0]) < -1e-9:
            return self._finding(
                symbol,
                Status.FAIL,
                f"KL is negative: {float(np.asarray(apart).reshape(-1)[0])}",
            )
        return self._finding(symbol, Status.PASS, "KL(p,p) = 0 and KL >= 0")


# ── differential equations ───────────────────────────────────────────────────


class DiffeqAxis(Axis):
    """A solver must converge at the order it claims.

    Halving the step has to cut the error by ``2**p``.  This is the check
    that distinguishes two methods of the same order from each other only
    weakly — which is how ``rk4`` once named a different tableau than the
    reference — so the tableau consistency conditions are checked too:
    ``sum(b) == 1`` and every ``sum(A[i]) == c[i]``.
    """

    name = "diffeq"
    summary = "solver convergence order and tableau consistency"
    kinds = frozenset({"diffeq", "value"})

    def applies(self, symbol: "Symbol") -> bool:
        # The published tableaux (DOPRI5, GL4, ...) are module-level
        # *instances*, so they enumerate as values rather than as
        # callables — 21 symbols that no axis reached until this said so.
        return symbol.subsystem == "diffeq"

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        obj = symbol.obj
        tableau = _as_tableau(obj)
        if tableau is not None:
            return self._check_tableau(symbol, tableau)
        if symbol.short in ("odeint", "odeint_adjoint"):
            return self._check_convergence(symbol, obj)
        return self._finding(symbol, Status.SKIP, "not a solver or a tableau")

    def _check_tableau(self, symbol: "Symbol", tableau: Any) -> Finding:
        a = np.asarray(tableau[0], dtype=np.float64)
        b = np.asarray(tableau[1], dtype=np.float64)
        c = np.asarray(tableau[2], dtype=np.float64)
        problems = []
        if abs(b.sum() - 1.0) > 1e-12:
            problems.append(f"weights sum to {b.sum():.12g}, not 1")
        rows = a.sum(axis=1) if a.ndim == 2 else a
        if rows.shape == c.shape and np.abs(rows - c).max() > 1e-12:
            problems.append(
                f"row sums differ from the nodes by {np.abs(rows - c).max():.2e}"
            )
        if problems:
            return self._finding(symbol, Status.FAIL, "; ".join(problems))
        return self._finding(symbol, Status.PASS, f"{len(b)} stages, consistent")

    def _check_convergence(self, symbol: "Symbol", solver: Any) -> Finding:
        # y' = -y, y(0) = 1 is deliberately *not* the test problem: every
        # four-stage fourth-order explicit method collapses onto the same
        # stability polynomial there and the check passes for the wrong
        # reason.  A non-autonomous, non-linear right-hand side does not.
        def rhs(t: Any, y: Any) -> Any:
            return lucid.cos(t) * y * y

        y0 = _probe.as_f64(np.array([0.5]))
        errors = []
        for steps in (8, 16, 32):
            grid = [i / steps for i in range(steps + 1)]
            try:
                out = solver(rhs, y0, grid, method="rk4", return_trajectory=False)
            except Exception as exc:  # noqa: BLE001
                return self._finding(
                    symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
                )
            got = _probe.to_numpy(out)
            if got is None:
                return self._finding(symbol, Status.SKIP, "solver returned no tensor")
            # dy/y^2 = cos(t) dt  =>  1/y0 - 1/y = sin(t)
            exact = 1.0 / (1.0 / 0.5 - math.sin(1.0))
            errors.append(abs(float(np.asarray(got).reshape(-1)[0]) - exact))

        if errors[-1] < 1e-13:
            return self._finding(symbol, Status.PASS, "converged to round-off")
        orders = [
            math.log2(errors[i] / errors[i + 1])
            for i in range(len(errors) - 1)
            if errors[i + 1] > 0
        ]
        if not orders:
            return self._finding(symbol, Status.PASS, "exact at every step count")
        observed = min(orders)
        if observed < 3.0:
            return self._finding(
                symbol,
                Status.FAIL,
                f"rk4 converged at order {observed:.2f}, expected ~4 "
                f"(errors {['%.2e' % e for e in errors]})",
            )
        return self._finding(symbol, Status.PASS, f"observed order {observed:.2f}")


def _as_tableau(obj: Any) -> "tuple[Any, Any, Any] | None":
    """``(A, b, c)`` if ``obj`` looks like a Butcher tableau."""
    parts = []
    for names in (("a", "A"), ("b", "B"), ("c", "C")):
        value = next((getattr(obj, n) for n in names if hasattr(obj, n)), None)
        if value is None:
            return None
        parts.append(value)
    try:
        return (
            np.asarray(parts[0], dtype=np.float64),
            np.asarray(parts[1], dtype=np.float64),
            np.asarray(parts[2], dtype=np.float64),
        )
    except Exception:  # noqa: BLE001
        return None


# ── quantization, serialization, compile ─────────────────────────────────────


class QuantizationAxis(Axis):
    """Quantise then dequantise must land within one step of the original."""

    name = "quant"
    summary = "quantize -> dequantize round trip stays within one step"
    kinds = frozenset({"quant"})

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        obj = symbol.obj
        if not callable(obj):
            return self._finding(symbol, Status.SKIP, "not callable")
        if "quantize" not in symbol.short and "qparams" not in symbol.short:
            return self._finding(symbol, Status.SKIP, "not a quantize entry point")

        x = _probe.as_f32(_probe.rng(3).uniform(-2.0, 2.0, (4, 8)))
        for args, kwargs in (
            ((x, 0.01, 0), {}),
            ((x,), {"scale": 0.01, "zero_point": 0}),
            ((x,), {}),
        ):
            try:
                q = obj(*args, **kwargs)
            except Exception:  # noqa: BLE001
                continue
            dq = None
            for name in ("dequantize", "dequantize_per_tensor"):
                fn = getattr(lucid.quantization, name, None) or getattr(
                    q, "dequantize", None
                )
                if fn is None:
                    continue
                try:
                    dq = _probe.to_numpy(
                        fn(q) if fn is not getattr(q, "dequantize", None) else fn()
                    )
                    break
                except Exception:  # noqa: BLE001
                    continue
            if dq is None:
                return self._finding(symbol, Status.PASS, "quantized without error")
            error = np.abs(dq.astype(float) - _probe.to_numpy(x).astype(float)).max()  # type: ignore[union-attr]
            if error > 0.02:
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"round trip is off by {error:.4f}, more than one step",
                )
            return self._finding(symbol, Status.PASS, f"round trip within {error:.4f}")
        return self._finding(symbol, Status.SKIP, "no argument shape worked")


class SerializationAxis(Axis):
    """A tensor must survive a round trip through disk, bit for bit."""

    name = "serialize"
    summary = "save / load round trip preserves values, dtype and shape"
    kinds = frozenset({"serialize"})

    def applies(self, symbol: "Symbol") -> bool:
        # ``save`` and ``load`` touch the filesystem, so they are flagged
        # stateful and the base class would refuse them.  This axis calls
        # them on purpose, inside a temporary directory it owns.
        return symbol.kind == "serialize" and symbol.short in ("save", "load")

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        if symbol.short not in ("save", "load"):
            return self._finding(symbol, Status.SKIP, "not the save/load pair")
        save = getattr(lucid, "save", None)
        load = getattr(lucid, "load", None)
        if save is None or load is None:
            return self._finding(symbol, Status.SKIP, "save/load not exposed")

        original = _probe.as_f64(_probe.sample("moderate", (3, 4)))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "probe.lct"
            try:
                save(original, str(path))
                restored = load(str(path))
            except Exception as exc:  # noqa: BLE001
                return self._finding(
                    symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
                )
            a, b = _probe.to_numpy(original), _probe.to_numpy(restored)
            if a is None or b is None:
                return self._finding(
                    symbol, Status.SKIP, "round trip produced no tensor"
                )
            if a.shape != b.shape:
                return self._finding(
                    symbol, Status.FAIL, f"shape {a.shape} became {b.shape}"
                )
            if not np.array_equal(a, b, equal_nan=True):
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"values changed by {np.nanmax(np.abs(a - b)):.3e} through the round trip",
                )
            if str(original.dtype) != str(restored.dtype):
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"dtype {original.dtype} became {restored.dtype}",
                )
        return self._finding(symbol, Status.PASS, "exact round trip")


class CompiledAxis(Axis):
    """A compiled function must agree with the eager one it came from."""

    name = "compiled"
    summary = "compiled output matches eager output"
    kinds = frozenset({"compiled"})

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        if symbol.short != "compile":
            return self._finding(symbol, Status.SKIP, "not the compile entry point")
        compile_fn = symbol.obj

        def eager(t: Any) -> Any:
            return lucid.tanh(t * 2.0) + lucid.exp(-t)

        x = _probe.as_f64(_probe.sample("moderate", (3, 4)))
        try:
            compiled = compile_fn(eager)
            got = _probe.to_numpy(compiled(x))
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:70]}"
            )
        want = _probe.to_numpy(eager(x))
        if got is None or want is None:
            return self._finding(symbol, Status.SKIP, "no comparable output")
        if not np.allclose(got.astype(float), want.astype(float), rtol=1e-6, atol=1e-9):
            return self._finding(
                symbol,
                Status.FAIL,
                f"compiled and eager differ by {np.abs(got.astype(float) - want.astype(float)).max():.3e}",
            )
        return self._finding(symbol, Status.PASS, "compiled matches eager")


class ClassContractAxis(Axis):
    """Plain classes: constructible, and their repr does not raise.

    The floor for a symbol nothing else can express.  A class that cannot
    be constructed with any obvious signature is reported SKIP so it
    shows up in the coverage gap rather than passing by omission.
    """

    name = "contract"
    summary = "plain classes construct and repr without raising"
    kinds = frozenset({"class", "distribution", "quant", "util", "diffeq"})

    def applies(self, symbol: "Symbol") -> bool:
        return symbol.kind in self.kinds and isinstance(symbol.obj, type)

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        # Signature-driven first, and only then the blind ladder.
        #
        # ``_try_construct`` tries a fixed list of argument tuples — ``(4,)``,
        # ``(3, 4)`` and so on — which is fine for a class that takes sizes
        # and wrong for one that does not.  ``MemoryStats(impl)`` accepted
        # the int, and its ``repr`` then failed with "'int' object has no
        # attribute 'current_bytes'", reported as a defect in a class nobody
        # would call that way.  Reading the signature refuses instead.
        from lucid.test.audit._axes_data import _construct

        instance, why = _construct(symbol.obj)
        if instance is None:
            # Falling back to the ladder here would undo the point: it is
            # the ladder's willingness to pass an int for anything that
            # produced these reports.  The ladder is only for a class whose
            # signature could not be read at all.
            if "no signature" not in why:
                return self._finding(symbol, Status.SKIP, f"construct: {why}")
            probe = _try_construct(symbol.obj)
            if probe is None:
                return self._finding(symbol, Status.SKIP, f"construct: {why}")
            instance = probe
        if instance is None:
            return self._finding(symbol, Status.SKIP, "no constructor signature worked")
        try:
            text = repr(instance)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.FAIL, f"repr raised {type(exc).__name__}: {exc}"
            )
        if not text:
            return self._finding(symbol, Status.FAIL, "repr returned an empty string")
        return self._finding(symbol, Status.PASS, "constructs and reprs")


class ConstantAxis(Axis):
    """Module-level values: dtypes, schemes, type aliases, the version.

    The last 60-odd symbols, and the reason the reach figure could not
    pass 92%.  They are not callable, so nothing above touches them —
    but a dtype that cannot build a tensor, a scheme constant that is
    silently the same object as another, or a ``__version__`` that is not
    a version are all real breakages, and each is one line to check.
    """

    name = "constant"
    summary = "dtypes build tensors, scheme constants are distinct, aliases resolve"
    kinds = frozenset({"value"})

    def applies(self, symbol: "Symbol") -> bool:
        return symbol.kind == "value"

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        value = symbol.obj
        if value is None:
            return self._finding(symbol, Status.FAIL, "the exported value is None")

        try:
            text = repr(value)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.FAIL, f"repr raised {type(exc).__name__}: {exc}"
            )
        if not text:
            return self._finding(symbol, Status.FAIL, "repr returned an empty string")

        if symbol.short == "__version__":
            if not isinstance(value, str) or not value[:1].isdigit():
                return self._finding(
                    symbol, Status.FAIL, f"not a version string: {value!r}"
                )
            return self._finding(symbol, Status.PASS, f"version {value}")

        # A dtype earns the strongest check available: build with it and
        # read the dtype back off the result.
        try:
            tensor = lucid.zeros((2, 2), dtype=value)
        except Exception:  # noqa: BLE001
            tensor = None
        if tensor is not None:
            if str(tensor.dtype) != str(value):
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"a tensor built with {value} reports dtype {tensor.dtype}",
                )
            return self._finding(
                symbol, Status.PASS, f"builds a tensor of {tensor.dtype}"
            )

        return self._finding(symbol, Status.PASS, f"exported value: {text[:48]}")


SUBSYSTEM_AXES: tuple[Axis, ...] = (
    DistributionAxis(),
    DiffeqAxis(),
    QuantizationAxis(),
    SerializationAxis(),
    CompiledAxis(),
    ClassContractAxis(),
    ConstantAxis(),
    SmokeAxis(),
)

__all__ = [
    "SUBSYSTEM_AXES",
    "ClassContractAxis",
    "CompiledAxis",
    "ConstantAxis",
    "DiffeqAxis",
    "DistributionAxis",
    "QuantizationAxis",
    "SerializationAxis",
    "SmokeAxis",
    "StateGuard",
]
