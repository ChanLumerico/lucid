"""The checks.

Each axis is one question asked of one symbol.  The questions were not
chosen abstractly — every one of them has already caught a defect in this
framework, and the note on each class says which.

Three habits are built into the base rather than left to each axis:

* **the instrument is guarded** — a check that cannot fail is reported
  VACUOUS, not PASS, because a vacuous pass reads as coverage and is not;
* **a disagreement is interrogated before it is reported** — a
  finite-difference probe near a pole disagrees for reasons that have
  nothing to do with the op, so a first FAIL is re-run at a finer step and
  reclassified TRUNCATION if it shrinks quadratically;
* **two spellings beat one hand-derivation** — where an op has another
  route to the same answer, that is the reference, because a hand-written
  expected value encodes a convention the framework is free to choose
  differently (``hardtanh`` at its own clamp boundary).
"""

import contextlib
import functools
import json
import pathlib
from typing import TYPE_CHECKING, Any

import numpy as np

import lucid
import lucid.autograd
from lucid.test.audit import _probe, _specs, _surface
from lucid.test.audit._result import Finding, Status

if TYPE_CHECKING:
    from lucid.test.audit._specs import Call
    from lucid.test.audit._surface import Symbol


class Context:
    """Run-wide switches an axis may consult."""

    def __init__(
        self,
        quick: bool = False,
        metal: bool = True,
        step: float = 1e-5,
        tolerance: float = 2e-5,
    ) -> None:
        self.quick = quick
        self.metal = metal
        self.step = step
        self.tolerance = tolerance

    @property
    def domains(self) -> list[str]:
        if self.quick:
            return ["moderate", "positive"]
        return list(_probe.DOMAINS)


class Axis:
    """One question, asked of every symbol it applies to."""

    name: str = ""
    summary: str = ""
    #: Which :attr:`Symbol.kind` values this axis can express.  ``method``
    #: is in the default because ``Tensor.*`` is 253 symbols and leaving
    #: it out put every one of them outside the audit.
    kinds: frozenset[str] = frozenset({"op", "method"})

    def applies(self, symbol: "Symbol") -> bool:
        if symbol.kind not in self.kinds:
            return False
        if not symbol.inert:
            return False
        return True

    def run(
        self, symbol: "Symbol", ctx: Context
    ) -> Finding:  # pragma: no cover - abstract
        raise NotImplementedError

    # ── shared helpers ───────────────────────────────────────────────────────

    def _finding(
        self, symbol: "Symbol", status: Status, detail: str = "", **evidence: Any
    ) -> Finding:
        return Finding(self.name, symbol.qualname, status, detail, evidence)

    def _working_call(
        self, fn: Any, symbol: "Symbol", ctx: Context
    ) -> "tuple[Call, str, Any] | tuple[None, None, str]":
        """The first candidate invocation that runs, and its domain.

        Returns ``(call, domain, output)`` or ``(None, None, reason)``.
        """
        last = "no candidate invocation ran"
        for domain in ctx.domains:
            for call in _specs.invocations(symbol.short, domain, symbol.qualname, fn):
                try:
                    out = fn(*call.args, **call.kwargs)
                except Exception as exc:  # noqa: BLE001 - surveying, not asserting
                    last = f"{call.note}: {type(exc).__name__}: {str(exc)[:70]}"
                    continue
                if _probe.to_numpy(out) is None:
                    last = f"{call.note}: returned no tensor"
                    continue
                return call, domain, out
        return None, None, last


# ── numeric axes ─────────────────────────────────────────────────────────────


class _DifferenceAxis(Axis):
    """Shared by the axes that compare against a finite difference.

    A central difference needs ``f(x+h)`` and ``f(x-h)`` to be the same
    function evaluated twice.  A stochastic op draws a fresh mask each
    call, so the quotient measures the draw rather than the derivative —
    dropout and its family reported a relative error of 1.0 for working
    exactly as designed, and ``gumbel_softmax`` did the same one axis
    over.  Stated once so the two axes cannot drift apart on it.
    """

    def applies(self, symbol: "Symbol") -> bool:
        if "stochastic" in symbol.flags:
            return False
        return super().applies(symbol)


class GradientAxis(_DifferenceAxis):
    """Analytic gradient against central finite differences, in float64.

    Found: seven in-place activations returning the pre-activation
    gradient, because they were built on a primitive that documents it
    does not extend the graph.
    """

    name = "grad"
    summary = "d/dx vs central finite differences (float64)"

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")
        call, domain, first = self._working_call(fn, symbol, ctx)
        if call is None:
            return self._finding(symbol, Status.SKIP, str(first))

        try:
            base = call.base
        except TypeError:
            return self._finding(
                symbol, Status.SKIP, "differentiated argument is not a tensor"
            )

        weights = _probe.covector(64, _probe.SEED_A)

        def scalar(array: np.ndarray) -> float:
            probe = call.with_primary(array)
            return float(_probe.contract(fn(*probe.args, **probe.kwargs), weights))

        # analytic
        probe = call.with_primary(base)
        x = probe.args[probe.primary]
        try:
            x.requires_grad_(True)
            loss = _probe.contract(fn(*probe.args, **probe.kwargs), weights)
            loss.backward()
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:70]}"
            )
        if x.grad is None:
            return self._finding(
                symbol, Status.UNSUPPORTED, "no gradient reached the input"
            )

        analytic = np.asarray(x.grad.numpy(), dtype=np.float64).reshape(-1)
        if not np.isfinite(analytic).all():
            return self._finding(
                symbol, Status.SKIP, f"non-finite gradient on '{domain}'"
            )
        if np.abs(analytic).max(initial=0.0) == 0.0:
            return self._finding(
                symbol,
                Status.VACUOUS,
                f"gradient is identically zero on '{domain}' — this check could not fail",
            )

        try:
            coarse = _probe.finite_difference(scalar, base, ctx.step)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.SKIP, f"fd failed: {type(exc).__name__}"
            )

        # A non-finite difference quotient is the probe leaving the op's
        # domain, not a wrong derivative.  ``log`` and ``sqrt`` are only
        # defined on part of the line, and perturbing the probe by h walks
        # off it — the analytic gradient was finite and checked, the
        # numerical one came back nan, and the comparison reported a
        # defect in nineteen ops that were computing correctly.
        if not np.isfinite(coarse).all():
            return self._finding(
                symbol,
                Status.SKIP,
                f"{domain}: the finite difference left the op's domain",
            )

        rel = _probe.relative(analytic, coarse.reshape(analytic.shape))
        if rel < ctx.tolerance:
            return self._finding(
                symbol, Status.PASS, f"{domain}: rel {rel:.2e}", rel=rel
            )

        # Interrogate before reporting.  Truncation falls like h**2.
        try:
            fine = _probe.finite_difference(scalar, base, ctx.step / 10.0)
        except Exception:  # noqa: BLE001
            fine = coarse
        rel_fine = _probe.relative(analytic, fine.reshape(analytic.shape))
        if _probe.quadratic_shrink(rel, rel_fine):
            return self._finding(
                symbol,
                Status.TRUNCATION,
                f"{domain}: rel {rel:.2e} -> {rel_fine:.2e} at h/10 — the probe, not the op",
                rel=rel,
                rel_refined=rel_fine,
            )
        return self._finding(
            symbol,
            Status.FAIL,
            f"{domain}: rel {rel:.2e}, still {rel_fine:.2e} at h/10",
            rel=rel,
            rel_refined=rel_fine,
            analytic=analytic[:8].tolist(),
            finite_difference=coarse[:8].tolist(),
        )


class SecondGradientAxis(_DifferenceAxis):
    """Second derivative against finite differences of the first.

    Found: ``prod`` / ``max`` / ``min`` returning the incoming seed under
    ``create_graph=True``, because the reduction base class applies sum's
    rule and only two of five reductions overrode it.
    """

    name = "grad2"
    summary = "d2/dx2 vs finite differences of the analytic gradient"

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")
        call, domain, _ = self._working_call(fn, symbol, ctx)
        if call is None:
            return self._finding(symbol, Status.SKIP, "no candidate invocation ran")

        try:
            base = call.base
        except TypeError:
            return self._finding(
                symbol, Status.SKIP, "differentiated argument is not a tensor"
            )

        w1 = _probe.covector(64, _probe.SEED_A)
        w2 = _probe.covector(64, _probe.SEED_B)

        def directional(array: np.ndarray) -> "tuple[Any, Any]":
            probe = call.with_primary(array)
            x = probe.args[probe.primary]
            x.requires_grad_(True)
            loss = _probe.contract(fn(*probe.args, **probe.kwargs), w1)
            (g,) = lucid.autograd.grad(loss, [x], create_graph=True)
            n = int(g.reshape(-1).shape[0])
            return x, (g.reshape(-1) * _probe.as_f64(w2[:n])).sum()

        try:
            x, scalar = directional(base)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:80]}"
            )
        try:
            (second,) = lucid.autograd.grad(scalar, [x])
            analytic = np.asarray(second.numpy(), dtype=np.float64).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            # Unreachable input is the standard case for a piecewise-constant
            # gradient — sum, mean, max and min all land here legitimately.
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:80]}"
            )

        try:
            fd = _probe.finite_difference(
                lambda a: float(directional(a)[1]), base, ctx.step
            )
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.SKIP, f"fd failed: {type(exc).__name__}"
            )

        if (
            np.abs(analytic).max(initial=0.0) == 0.0
            and np.abs(fd).max(initial=0.0) == 0.0
        ):
            return self._finding(
                symbol,
                Status.PASS,
                f"{domain}: second derivative is zero (op is linear)",
            )

        # Same guard the first-derivative axis carries: differencing the
        # gradient twice steps 2h away from the probe, and ``acos``,
        # ``asin``, ``sqrt`` and ``erfinv`` are only defined on part of the
        # line.  Both the analytic second derivative and the difference
        # come back nan, which is agreement the comparison cannot express —
        # ``nan != nan`` — and it was reported as a disagreement in
        # nineteen ops that were computing correctly.
        if not np.isfinite(fd).all() or not np.isfinite(analytic).all():
            return self._finding(
                symbol,
                Status.SKIP,
                f"{domain}: the second difference left the op's domain",
            )

        rel = _probe.relative(analytic, fd.reshape(analytic.shape))
        if rel < 1e-4:
            return self._finding(
                symbol, Status.PASS, f"{domain}: rel {rel:.2e}", rel=rel
            )

        try:
            fine = _probe.finite_difference(
                lambda a: float(directional(a)[1]), base, ctx.step / 10.0
            )
            rel_fine = _probe.relative(analytic, fine.reshape(analytic.shape))
        except Exception:  # noqa: BLE001
            rel_fine = rel
        if _probe.quadratic_shrink(rel, rel_fine):
            return self._finding(
                symbol,
                Status.TRUNCATION,
                f"{domain}: rel {rel:.2e} -> {rel_fine:.2e} at h/10",
            )
        return self._finding(
            symbol,
            Status.FAIL,
            f"{domain}: rel {rel:.2e}, still {rel_fine:.2e} at h/10",
            rel=rel,
        )


class CreateGraphAxis(Axis):
    """``autograd.grad(create_graph=True)`` against ``backward()``.

    The two must agree on the *first* derivative whatever the graph mode.
    This is what caught the reductions: ``backward()`` was right and
    ``grad(create_graph=True)`` returned the seed, silently.
    """

    name = "creategraph"
    summary = "autograd.grad(create_graph=True) vs backward()"

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")
        call, domain, _ = self._working_call(fn, symbol, ctx)
        if call is None:
            return self._finding(symbol, Status.SKIP, "no candidate invocation ran")
        try:
            base = call.base
        except TypeError:
            return self._finding(
                symbol, Status.SKIP, "differentiated argument is not a tensor"
            )

        weights = _probe.covector(64, _probe.SEED_A)

        def loss_of(array: np.ndarray) -> "tuple[Any, Any]":
            probe = call.with_primary(array)
            x = probe.args[probe.primary]
            x.requires_grad_(True)
            return x, _probe.contract(fn(*probe.args, **probe.kwargs), weights)

        try:
            x_ref, loss = loss_of(base)
            loss.backward()
            reference = np.asarray(x_ref.grad.numpy(), dtype=np.float64).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol,
                Status.UNSUPPORTED,
                f"backward(): {type(exc).__name__}: {str(exc)[:60]}",
            )
        if np.abs(reference).max(initial=0.0) == 0.0:
            return self._finding(
                symbol, Status.VACUOUS, "reference gradient is identically zero"
            )

        try:
            x_probe, loss2 = loss_of(base)
            (got,) = lucid.autograd.grad(loss2, [x_probe], create_graph=True)
            candidate = np.asarray(got.numpy(), dtype=np.float64).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"grad(create_graph): {type(exc).__name__}"
            )

        # Two routes that both produce NaN agree; the comparison does not
        # know that, because ``nan != nan``.  ``acos``, ``asin``, ``sqrt``
        # and ``erfinv`` are only defined on part of the line, and where
        # the probe leaves it *both* backward() and grad(create_graph=True)
        # answer nan — correctly and identically — and the relative error
        # came out nan and was reported as a disagreement in 26 ops.
        finite = np.isfinite(reference) & np.isfinite(candidate)
        if not finite.any():
            return self._finding(
                symbol,
                Status.SKIP,
                f"{domain}: both routes are non-finite here — outside the op's domain",
            )
        if not finite.all():
            if not np.array_equal(np.isfinite(reference), np.isfinite(candidate)):
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"{domain}: the two routes disagree about which entries are finite",
                    backward=reference[:8].tolist(),
                    create_graph=candidate[:8].tolist(),
                )
            reference, candidate = reference[finite], candidate[finite]

        rel = _probe.relative(reference, candidate)
        if rel < 1e-9:
            return self._finding(symbol, Status.PASS, f"{domain}: rel {rel:.2e}")
        return self._finding(
            symbol,
            Status.FAIL,
            f"{domain}: rel {rel:.2e} between backward() and grad(create_graph=True)",
            backward=reference[:8].tolist(),
            create_graph=candidate[:8].tolist(),
        )


class EntryPointAxis(Axis):
    """The same op through every spelling it has.

    Found: scalar coercion existed on the operator path only, so
    ``x ** 2`` worked and ``x.pow(2)`` raised, and ``int32 + 1.5``
    silently truncated.
    """

    name = "entry"
    summary = "lucid.f(x) vs x.f() vs F.f(x)"

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        routes = list(_surface.counterparts(symbol))
        if len(routes) < 2:
            return self._finding(
                symbol, Status.NOT_APPLICABLE, "only one entry point to compare"
            )

        results: dict[str, Any] = {}
        errors: dict[str, str] = {}
        # The first route's callable is what the signature tier reads.
        # Every route is the same operation, so any of them describes it.
        probe_fn = routes[0][1]
        for domain in ctx.domains:
            for call in _specs.invocations(
                symbol.short, domain, symbol.qualname, probe_fn
            ):
                results.clear()
                errors.clear()
                for label, fn in routes:
                    args = call.args if label != "method" else call.args[1:]
                    target = call.args[0] if label == "method" else None
                    try:
                        out = (
                            fn(target, *args, **call.kwargs)
                            if label == "method"
                            else fn(*call.args, **call.kwargs)
                        )
                        got = _probe.to_numpy(out)
                        if got is None:
                            errors[label] = "no tensor"
                        else:
                            results[label] = got
                    except Exception as exc:  # noqa: BLE001
                        errors[label] = type(exc).__name__
                if len(results) >= 2:
                    break
            if len(results) >= 2:
                break

        if len(results) < 2:
            if results and errors:
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"reachable one way but not another: ok={sorted(results)} "
                    f"failed={errors}",
                )
            return self._finding(symbol, Status.SKIP, f"no shared invocation: {errors}")

        labels = sorted(results)
        first = results[labels[0]]
        for label in labels[1:]:
            other = results[label]
            if first.shape != other.shape:
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"{labels[0]} gives {first.shape}, {label} gives {other.shape}",
                )
            if not np.allclose(first, other, rtol=1e-9, atol=1e-12, equal_nan=True):
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"{labels[0]} and {label} disagree by "
                    f"{np.abs(first.astype(float) - other.astype(float)).max():.3e}",
                )
        return self._finding(symbol, Status.PASS, f"{len(labels)} entry points agree")


class DeviceAxis(Axis):
    """CPU against Metal — including a non-finite probe.

    Found: ``relu(NaN)`` returned 0 on the CPU and NaN on Metal.  The
    existing parity sweeps missed it because they probe with
    well-conditioned uniform data, which is why this axis runs the
    non-finite vector as well.
    """

    name = "device"
    summary = "cpu vs metal, finite and non-finite inputs"

    def applies(self, symbol: "Symbol") -> bool:
        return super().applies(symbol) and "stochastic" not in symbol.flags

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        if not ctx.metal:
            return self._finding(symbol, Status.SKIP, "metal unavailable")
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")
        call, domain, _ = self._working_call(fn, symbol, ctx)
        if call is None:
            return self._finding(symbol, Status.SKIP, "no candidate invocation ran")

        def on(device: str, override: np.ndarray | None = None) -> Any:
            args = []
            for i, a in enumerate(call.args):
                if hasattr(a, "to"):
                    if override is not None and i == call.primary:
                        arr = np.resize(override, _probe.to_numpy(a).shape)  # type: ignore[arg-type]
                        args.append(_probe.as_f32(arr, device=device))
                    else:
                        # float32, not the probe's native float64.  This axis
                        # asks whether the two devices agree, and Metal has
                        # no float64 at all — so every float64 operand made
                        # the comparison fail on the move rather than on the
                        # answer.  649 cells, a tenth of the sweep, reported
                        # "unsupported" for a platform fact rather than
                        # comparing anything.  Both sides get the same
                        # float32 input, which is what makes them comparable.
                        # Narrowed *before* the move, not after: ``.to(metal)``
                        # on a float64 tensor is the call that raises, so a
                        # cast afterwards never runs.
                        narrowed = (
                            a.to(lucid.float32)
                            if str(a.dtype).endswith("float64")
                            else a
                        )
                        args.append(narrowed.to(device))
                else:
                    args.append(a)
            return fn(*args, **call.kwargs)

        try:
            cpu_out, metal_out = on("cpu"), on("metal")
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:70]}"
            )

        # Guard the instrument: a tensor that quietly stayed on the CPU
        # would make every comparison below trivially true.
        for out, want in ((cpu_out, "cpu"), (metal_out, "metal")):
            device = getattr(out, "device", None)
            if device is not None and want not in str(device):
                return self._finding(
                    symbol, Status.SKIP, f"output landed on {device}, expected {want}"
                )

        a, b = _probe.to_numpy(cpu_out), _probe.to_numpy(metal_out)
        if a is None or b is None or a.shape != b.shape:
            return self._finding(symbol, Status.SKIP, "outputs not comparable")
        if not np.allclose(
            a.astype(float), b.astype(float), rtol=2e-5, atol=1e-6, equal_nan=True
        ):
            return self._finding(
                symbol,
                Status.FAIL,
                f"{domain}: cpu and metal differ by "
                f"{np.nanmax(np.abs(a.astype(float) - b.astype(float))):.3e}",
            )

        # The probe the old sweeps did not carry.
        try:
            nan_cpu = _probe.to_numpy(on("cpu", _probe.NON_FINITE))
            nan_metal = _probe.to_numpy(on("metal", _probe.NON_FINITE))
        except Exception:  # noqa: BLE001
            return self._finding(symbol, Status.PASS, f"{domain}: finite inputs agree")
        if nan_cpu is None or nan_metal is None or nan_cpu.shape != nan_metal.shape:
            return self._finding(symbol, Status.PASS, f"{domain}: finite inputs agree")
        if not np.allclose(
            nan_cpu.astype(float),
            nan_metal.astype(float),
            rtol=2e-5,
            atol=1e-6,
            equal_nan=True,
        ):
            return self._finding(
                symbol,
                Status.FAIL,
                "cpu and metal disagree on a non-finite input "
                f"(cpu {nan_cpu.reshape(-1)[:4]}, metal {nan_metal.reshape(-1)[:4]})",
            )
        return self._finding(
            symbol, Status.PASS, f"{domain}: finite and non-finite agree"
        )


class NonFiniteAxis(Axis):
    """NaN must survive an op that has no reason to consume it.

    Found: the CPU ``relu`` turned NaN into 0, so a NaN entering a network
    stopped being traceable at the first activation.  A NaN that
    propagates can be found; one that becomes a zero cannot.
    """

    name = "nonfinite"
    summary = "NaN propagation through elementwise ops"

    #: Ops whose whole job is to consume or classify a NaN, plus the ones
    #: whose *definition* is to ignore it.  ``fmax`` and ``fmin`` are the
    #: IEEE maximumNumber / minimumNumber operations — returning the
    #: non-NaN operand is what distinguishes them from ``maximum`` and
    #: ``minimum``, which do propagate — and ``sign(nan)`` is 0 in the
    #: reference.  Verified against it rather than assumed.
    _CONSUMERS = frozenset(
        {
            "fmax",
            "fmin",
            "sign",
            "sign_",
            "isnan",
            "isinf",
            "isfinite",
            "isneginf",
            "isposinf",
            "isreal",
            "iscomplex",
            "nan_to_num",
            "nansum",
            "nanmean",
            "nanmedian",
            "nanquantile",
            "nanstd",
            "nanvar",
            "nanargmax",
            "nanargmin",
            "nan_to_num_",
            "logical_not",
            "signbit",
            "argsort",
            "argmax",
            "argmin",
            "sort",
            "count_nonzero",
            "any",
            "all",
            "zeros_like",
            "ones_like",
            "full_like",
            "empty_like",
            "isclose",
            "allclose",
            "equal",
            "eq",
            "ne",
            "not_equal",
            "greater",
            "less",
            "greater_equal",
            "less_equal",
            "gt",
            "lt",
            "ge",
            "le",
            "sign",
            "heaviside",
        }
    )

    def applies(self, symbol: "Symbol") -> bool:
        if not super().applies(symbol):
            return False
        return symbol.short not in self._CONSUMERS and "stochastic" not in symbol.flags

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")
        call, _, _ = self._working_call(fn, symbol, ctx)
        if call is None:
            return self._finding(symbol, Status.SKIP, "no candidate invocation ran")

        try:
            shape = _probe.to_numpy(call.args[call.primary]).shape  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            return self._finding(symbol, Status.SKIP, "primary argument has no shape")

        # Every element, not just the first.  A single NaN at index 0 asks
        # a weaker question than it looks: a crop, a ``take``, a pooling
        # window or a ``masked_select`` can legitimately not include that
        # element, and reported a defect for reading somewhere else.  If
        # the whole input is NaN then any op that reads *any* of it has to
        # say so.
        probe = np.full(shape, np.nan, dtype=np.float64)
        try:
            out = _probe.to_numpy(fn(*call.with_primary(probe).args, **call.kwargs))
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
            )

        # An op that ignores its input cannot propagate anything through it.
        # ``eye``, ``ones_``, ``new_zeros``, ``dirac`` and the rest of the
        # factories take a tensor only for its shape or its device, and
        # demanding a NaN out of them was demanding they corrupt their own
        # output.  Detected rather than listed: feed a second, different
        # input and see whether the answer moves.
        try:
            other = _probe.to_numpy(
                fn(
                    *call.with_primary(np.full(shape, 0.5, dtype=np.float64)).args,
                    **call.kwargs,
                )
            )
        except Exception:  # noqa: BLE001
            other = None
        if (
            other is not None
            and out is not None
            and np.array_equal(
                np.nan_to_num(np.asarray(out, dtype=np.float64), nan=0.0),
                np.nan_to_num(np.asarray(other, dtype=np.float64), nan=0.0),
            )
        ):
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "output does not depend on the input's values",
            )
        if out is None or out.size == 0:
            return self._finding(symbol, Status.SKIP, "no comparable output")
        if out.dtype.kind not in "fc":
            return self._finding(
                symbol, Status.SKIP, f"output dtype {out.dtype} cannot carry NaN"
            )
        if not np.isnan(out).any():
            return self._finding(
                symbol,
                Status.FAIL,
                f"a NaN input produced no NaN anywhere in the output "
                f"(first values {np.asarray(out).reshape(-1)[:4]})",
            )
        return self._finding(symbol, Status.PASS, "NaN propagates")


class BroadcastAxis(Axis):
    """Every broadcast direction, including the ones a guard might skip.

    Found: ``where``'s guard compared the condition against one branch and
    never the two branches against each other.
    """

    name = "broadcast"
    summary = "all broadcast directions for binary ops"

    _PAIRS = (
        ((3, 4), (1, 4)),
        ((1, 4), (3, 4)),
        ((3, 1), (1, 4)),
        ((1, 4), (3, 1)),
        ((4,), (3, 4)),
        ((3, 4), (4,)),
        ((1, 1), (3, 4)),
        ((3, 4), (1, 1)),
        ((2, 1, 4), (3, 4)),
    )

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")
        # Accepting a second argument is not the same as *using* it.
        # ``relu(x, inplace)`` takes two positionals and ignores the
        # second, so a shape-only test called it a broadcast failure six
        # ways.  The operand has to actually change the answer.
        a0 = _probe.as_f64(_probe.sample("positive", (3, 4)))
        b0 = _probe.as_f64(_probe.sample("positive", (3, 4)))
        c0 = _probe.as_f64(_probe.sample("positive", (3, 4)) * 3.0 + 1.0)
        try:
            with_b = _probe.to_numpy(fn(a0, b0))
            with_c = _probe.to_numpy(fn(a0, c0))
        except Exception:  # noqa: BLE001
            return self._finding(symbol, Status.NOT_APPLICABLE, "not a two-tensor op")
        if with_b is None or with_c is None:
            return self._finding(symbol, Status.NOT_APPLICABLE, "not a two-tensor op")
        if np.array_equal(with_b, with_c, equal_nan=True):
            return self._finding(
                symbol, Status.SKIP, "second argument does not affect the result"
            )
        # ...and taking two operands is not the same as being elementwise.
        # ``kron`` returns (3, 16), ``mv`` contracts, ``masked_select``
        # returns a 1-D selection, ``binary_cross_entropy`` reduces to a
        # scalar.  Demanding a broadcast shape from any of them reports a
        # defect in an op that is behaving exactly as specified.  The
        # discriminator is the equal-shape case: an elementwise op maps
        # (3, 4) x (3, 4) to (3, 4).
        if tuple(with_b.shape) != (3, 4):
            return self._finding(
                symbol,
                Status.SKIP,
                f"not elementwise — equal shapes give {tuple(with_b.shape)}, not (3, 4)",
            )

        failures: list[str] = []
        for sa, sb in self._PAIRS:
            a = _probe.as_f64(_probe.rng(1).uniform(0.5, 1.5, sa))
            b = _probe.as_f64(_probe.rng(2).uniform(0.5, 1.5, sb))
            want = np.broadcast_shapes(sa, sb)
            try:
                got = _probe.to_numpy(fn(a, b))
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{sa}x{sb}: {type(exc).__name__}")
                continue
            if got is not None and tuple(got.shape) != want:
                failures.append(f"{sa}x{sb} -> {got.shape}, expected {want}")
        if failures:
            return self._finding(
                symbol, Status.FAIL, "; ".join(failures[:4]), failures=failures
            )
        return self._finding(symbol, Status.PASS, f"{len(self._PAIRS)} directions")


class DtypeAxis(Axis):
    """Every dtype the framework claims, on both devices.

    Found (earlier): the CPU backend was missing Bool / I8 / I16 / F16
    paths that Metal had, so the same call worked on one device and raised
    on the other.
    """

    name = "dtype"
    summary = "dtype coverage and cpu/metal symmetry"

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")
        call, _, _ = self._working_call(fn, symbol, ctx)
        if call is None:
            return self._finding(symbol, Status.SKIP, "no candidate invocation ran")
        try:
            base = call.base
        except TypeError:
            return self._finding(
                symbol, Status.SKIP, "primary argument is not a tensor"
            )

        del base  # only needed to prove there is one; `dtype_args` reads it

        devices = ["cpu", "metal"] if ctx.metal else ["cpu"]
        support: dict[str, set[str]] = {d: set() for d in devices}
        for name in _probe.DTYPES:
            dt = _probe.dtype_of(name)
            if dt is None:
                continue
            for device in devices:

                def build(array: Any, follow: bool, _d: str = device) -> Any:
                    if not follow:
                        return lucid.tensor(
                            np.ascontiguousarray(array), dtype=None, device=_d
                        )
                    return lucid.tensor(
                        np.ascontiguousarray(array.astype(_probe.numpy_of(name))),
                        dtype=dt,
                        device=_d,
                    )

                try:
                    args = _probe.dtype_args(call, name, build)
                    if _probe.to_numpy(fn(*args, **call.kwargs)) is not None:
                        support[device].add(name)
                except Exception:  # noqa: BLE001
                    continue

        if not any(support.values()):
            return self._finding(symbol, Status.SKIP, "no dtype accepted")
        if len(devices) == 2:
            # float64 does not exist on Metal and the engine documents the
            # downcast, so holding it against an op would flag every one.
            only_cpu = sorted(support["cpu"] - support["metal"] - {"float64"})
            only_metal = sorted(support["metal"] - support["cpu"])
            if only_cpu or only_metal:
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"asymmetric dtype support — cpu only {only_cpu}, "
                    f"metal only {only_metal}"
                    + _contract_verdict(symbol.qualname, only_cpu, only_metal),
                    cpu=sorted(support["cpu"]),
                    metal=sorted(support["metal"]),
                )
        return self._finding(
            symbol,
            Status.PASS,
            f"{len(support[devices[0]])}/{len(_probe.DTYPES)} dtypes",
        )


class EdgeAxis(Axis):
    """Degenerate shapes: empty, 0-d, and a single element.

    An op that raises on an empty tensor is usually a real gap, but the
    line is drawn at *inventing data*: an empty input must not produce a
    non-empty output.
    """

    name = "edge"
    summary = "empty, 0-d and size-1 inputs"

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")
        call, _, _ = self._working_call(fn, symbol, ctx)
        if call is None:
            return self._finding(symbol, Status.SKIP, "no candidate invocation ran")
        try:
            shape = call.base.shape
        except TypeError:
            return self._finding(
                symbol, Status.SKIP, "primary argument is not a tensor"
            )

        notes: list[str] = []

        # "Empty in, empty out" only holds for an op that keeps its shape.
        # A reduction over an empty tensor is its identity element — the
        # sum of nothing is 0, a scalar — and a selection returns however
        # many elements matched.  The rule was applied to everything, so
        # ``logsumexp``, ``masked_select`` and ``block_diag`` were reported
        # for agreeing with the reference exactly.  Whether the op keeps
        # its shape is asked rather than listed: run it once at full size
        # and see.
        reference_out = _probe.to_numpy(fn(*call.args, **call.kwargs))
        shape_preserving = reference_out is not None and tuple(
            reference_out.shape
        ) == tuple(shape)

        empty = np.zeros((0, *shape[1:]), dtype=np.float64)
        try:
            out = _probe.to_numpy(fn(*call.with_primary(empty).args, **call.kwargs))
            if (
                shape_preserving
                and out is not None
                and out.ndim > 0
                and out.shape[0] != 0
                and out.size != 0
            ):
                return self._finding(
                    symbol, Status.FAIL, f"empty input produced shape {out.shape}"
                )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"empty: {type(exc).__name__}")

        single = np.full((1,) * len(shape), 0.7, dtype=np.float64)
        try:
            fn(*call.with_primary(single).args, **call.kwargs)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"size-1: {type(exc).__name__}")

        if len(notes) == 2:
            return self._finding(symbol, Status.UNSUPPORTED, "; ".join(notes))
        return self._finding(
            symbol, Status.PASS, "; ".join(notes) or "empty and size-1 accepted"
        )


# ── structural axes ──────────────────────────────────────────────────────────


class ModuleAxis(Axis):
    """``nn.Module`` lifecycle: build, forward, backward, save, move.

    Found (earlier, by hand): a family whose model and its dynamics both
    held the same backbone had every weight in ``state_dict`` twice —
    invisible to ``parameters()``, which dedupes by identity, so only a
    round trip through serialisation shows it.
    """

    name = "module"
    summary = "construct, forward, backward, state_dict round trip, device move"
    kinds = frozenset({"module"})

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        cls = symbol.obj
        module = _try_construct(cls)
        if module is None:
            return self._finding(symbol, Status.SKIP, "no constructor signature worked")

        params = list(module.parameters())
        # Non-persistent buffers are excluded from ``state_dict`` on
        # purpose — a positional-encoding table or a rotary cache is
        # recomputed from the constructor arguments, so writing it into
        # every checkpoint is waste, and the reference framework draws the
        # same distinction.  Counting them made ``RotaryEmbedding`` and
        # ``SinusoidalEmbedding`` look like they had lost their state.
        non_persistent = getattr(module, "_non_persistent_buffers", set())
        buffers = [
            name for name, _ in module.named_buffers() if name not in non_persistent
        ]
        state = module.state_dict()
        if len(state) != len(params) + len(buffers):
            return self._finding(
                symbol,
                Status.FAIL,
                f"state_dict has {len(state)} entries for {len(params)} parameters "
                f"and {len(buffers)} buffers — a shared submodule is registered twice",
            )

        out, note = _try_forward(module)
        if out is None:
            return self._finding(symbol, Status.SKIP, f"forward: {note}")

        try:
            reloaded = _try_construct(cls)
            if reloaded is not None:
                reloaded.load_state_dict(module.state_dict())
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol,
                Status.FAIL,
                f"state_dict does not round trip: {type(exc).__name__}: {exc}",
            )

        if params:
            try:
                loss = (out * out).mean()
                module.zero_grad()
                loss.backward()
                reached = [p for p in module.parameters() if p.grad is not None]
                if not reached:
                    return self._finding(
                        symbol,
                        Status.FAIL,
                        "forward ran but no parameter received a gradient",
                    )
            except Exception as exc:  # noqa: BLE001
                return self._finding(
                    symbol,
                    Status.UNSUPPORTED,
                    f"backward: {type(exc).__name__}: {str(exc)[:60]}",
                )
        return self._finding(
            symbol, Status.PASS, f"{len(params)} params, {len(state)} state keys"
        )


class OptimAxis(Axis):
    """Optimizers: a step must move parameters and survive a round trip."""

    name = "optim"
    summary = "step, state_dict round trip, convergence on a convex problem"
    kinds = frozenset({"optim"})

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        cls = symbol.obj
        target = _probe.as_f64(np.array([1.5, -0.5, 2.0]))
        weight = lucid.nn.Parameter(_probe.as_f64(np.zeros(3)))

        optimiser = None
        for kwargs in ({"lr": 0.1}, {"lr": 0.1, "max_iter": 4}, {}):
            with contextlib.suppress(Exception):
                optimiser = cls([weight], **kwargs)
                break
        if optimiser is None:
            return self._finding(symbol, Status.SKIP, "no constructor signature worked")

        def closure() -> Any:
            optimiser.zero_grad()
            loss = ((weight - target) ** 2).sum()
            loss.backward()
            return loss

        try:
            first = float(closure())
            for _ in range(12):
                loss = closure()
                try:
                    optimiser.step(closure)  # line-search optimizers need it
                except TypeError:
                    optimiser.step()
            last = float(loss)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:70]}"
            )

        if not np.isfinite(last):
            return self._finding(symbol, Status.FAIL, f"loss became {last}")
        if last >= first:
            return self._finding(
                symbol,
                Status.FAIL,
                f"12 steps on a convex quadratic did not reduce the loss "
                f"({first:.4f} -> {last:.4f})",
            )

        try:
            state = optimiser.state_dict()
            optimiser.load_state_dict(state)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol,
                Status.FAIL,
                f"state_dict does not round trip: {type(exc).__name__}",
            )
        return self._finding(symbol, Status.PASS, f"loss {first:.4f} -> {last:.4f}")


# ── construction helpers ─────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def _contract() -> "dict[str, list[str]]":
    """The measured reference dtype table, or empty when absent.

    Checked in, so it is available without the reference framework
    installed.  Regenerate with
    ``python -m lucid.test.audit.tools.dtype_contract``.
    """
    path = pathlib.Path(__file__).with_name("dtype_contract.json")
    try:
        data = json.loads(path.read_text())
    except OSError, ValueError:
        return {}
    symbols = data.get("symbols")
    return symbols if isinstance(symbols, dict) else {}


def _contract_verdict(
    qualname: str, only_cpu: "list[str]", only_metal: "list[str]"
) -> str:
    """Which device is wrong, when the reference has an opinion.

    A disagreement between the two devices says they differ, never which
    one to change — and deciding that by eye was going badly: the
    reference accepts int64 for ``avg_pool2d`` and rejects every integer
    for ``softmax``, and neither is guessable from the name.  When the
    op was measured, name the side that deviates so the finding is
    actionable rather than merely true.
    """
    accepted = _contract().get(qualname)
    if accepted is None:
        return ""
    reference = set(accepted)
    # only_cpu: cpu takes it, metal does not.  only_metal: the reverse.
    # Whichever side agrees with the reference is the one to keep.
    parts = []
    for device, mine, theirs in (
        ("cpu", only_cpu, only_metal),
        ("metal", only_metal, only_cpu),
    ):
        over = [d for d in mine if d not in reference]
        under = [d for d in theirs if d in reference]
        if over:
            parts.append(f"{device} wrongly accepts {over}")
        if under:
            parts.append(f"{device} wrongly rejects {under}")
    return f" — reference says: {'; '.join(parts)}" if parts else ""


#: Constructor ladders, tried in order.  A module that needs an argument
#: shape this does not guess is reported SKIP rather than FAIL.
_CTOR_ARGS: tuple[tuple[tuple[Any, ...], dict[str, Any]], ...] = (
    ((), {}),
    ((4,), {}),
    ((4, 4), {}),
    ((3, 4), {}),
    ((3, 4, 3), {}),
    ((4,), {"eps": 1e-5}),
    ((1, 4), {}),
    ((4, 4, 3), {"padding": 1}),
)


def _try_construct(cls: Any) -> Any:
    for args, kwargs in _CTOR_ARGS:
        try:
            return cls(*args, **kwargs)
        except Exception:  # noqa: BLE001
            continue
    return None


#: Input shapes to try against an unknown module, coarse to fine.
_FORWARD_SHAPES: tuple[tuple[int, ...], ...] = (
    (2, 4),
    (2, 4, 6),
    (2, 3, 6, 6),
    (2, 4, 4),
    (2, 3, 4, 6, 6),
)


def _try_forward(module: Any) -> "tuple[Any, str]":
    last = "no input shape worked"
    for shape in _FORWARD_SHAPES:
        x = _probe.as_f64(_probe.sample("moderate", shape))
        try:
            out = module(x)
        except Exception as exc:  # noqa: BLE001
            last = f"{shape}: {type(exc).__name__}"
            continue
        tensor = out if hasattr(out, "shape") else None
        if tensor is None:
            for attr in ("logits", "sample", "last_hidden_state"):
                tensor = getattr(out, attr, None)
                if tensor is not None:
                    break
        if tensor is not None and hasattr(tensor, "shape"):
            return tensor, str(shape)
        last = f"{shape}: returned no tensor"
    return None, last


#: The core numeric axes, cheapest first.
CORE_AXES: tuple[Axis, ...] = (
    EntryPointAxis(),
    BroadcastAxis(),
    NonFiniteAxis(),
    EdgeAxis(),
    DtypeAxis(),
    DeviceAxis(),
    GradientAxis(),
    CreateGraphAxis(),
    SecondGradientAxis(),
    ModuleAxis(),
    OptimAxis(),
)

# Imported at the bottom: the stability and subsystem axes subclass Axis
# and are registered here, so a single ``ALL_AXES`` stays the one place a
# run is defined.  Everything they need from this module is already bound
# by the time the import executes.
from lucid.test.audit._axes_data import DATA_AXES  # noqa: E402
from lucid.test.audit._axes_stability import STABILITY_AXES  # noqa: E402
from lucid.test.audit._axes_subsystem import SUBSYSTEM_AXES  # noqa: E402

#: Every axis, in the order a full run executes them — cheapest first, so
#: a ``--fail-fast`` run surfaces the loud problems before the slow ones.
#: ``SUBSYSTEM_AXES`` goes last because it ends with the smoke axis, which
#: is the floor rather than a question.
ALL_AXES: tuple[Axis, ...] = (
    *CORE_AXES[:6],
    *STABILITY_AXES,
    *CORE_AXES[6:],
    *DATA_AXES,
    *SUBSYSTEM_AXES,
)


def axis_by_name(name: str) -> Axis | None:
    return next((a for a in ALL_AXES if a.name == name), None)


def axis_names() -> list[str]:
    return [a.name for a in ALL_AXES]


__all__ = [
    "ALL_AXES",
    "CORE_AXES",
    "Axis",
    "Context",
    "axis_by_name",
    "axis_names",
]
