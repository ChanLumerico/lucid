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

import annotationlib
import contextlib
import inspect
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

    def _draws_randomly(self, fn: Any, call: "Call") -> bool:
        """Whether two identical calls disagree.

        The definition of stochastic, asked rather than looked up.  It was
        looked up before — a tuple of substrings, ``rand``, ``dropout``,
        ``gumbel`` — and ``nn.init.orthogonal`` and ``nn.init.sparse``
        matched none of them while drawing a fresh random matrix on every
        call.  Four separate axes reported them for it, each measuring the
        draw rather than the op: a finite difference across two draws, a
        cpu/metal comparison across two RNG streams that differ by design,
        and a strided view across a third.

        Two calls, no list, nothing to keep in sync.
        """
        # Rebuilt between calls, not reused.  An op that writes into its own
        # input makes the second call read what the first one wrote, so two
        # draws look identical and the op reads as deterministic — which
        # is what happened to ``nn.init.orthogonal`` and ``nn.init.sparse``
        # the moment they stopped replacing the impl and started writing
        # through it.  One fix disabling another is the reason this is
        # worth stating: the probe has to repeat the *experiment*, not the
        # call.
        primary = call.args[call.primary] if call.args else None
        base = None
        if hasattr(primary, "dtype") and hasattr(primary, "shape"):
            try:
                base = call.base
            except TypeError:
                base = None
        try:
            one = call if base is None else call.with_primary(base)
            first = _probe.to_numpy(fn(*one.args, **one.kwargs))
            two = call if base is None else call.with_primary(base)
            second = _probe.to_numpy(fn(*two.args, **two.kwargs))
        except Exception:  # noqa: BLE001 - surveying, not asserting
            return False
        if first is None or second is None or first.shape != second.shape:
            return False
        if first.dtype.kind not in "fciub" or second.dtype.kind not in "fciub":
            return False
        return not np.array_equal(first, second, equal_nan=first.dtype.kind == "f")

    @staticmethod
    def _comparable(array: np.ndarray) -> np.ndarray:
        """An array in a form ``np.allclose`` can read.

        ``astype(float)`` on a complex array **discards the imaginary
        part**, silently.  Every complex op was being compared on half of
        its answer, so a backend could have got the imaginary part
        entirely wrong and this axis would have called it agreement.
        """
        if array.dtype.kind == "c":
            return np.stack([array.real, array.imag], axis=-1).astype(float)
        return array.astype(float)

    @staticmethod
    def _same_multiset(a: np.ndarray, b: np.ndarray) -> bool:
        """Whether the two hold the same magnitudes in a different order.

        A decomposition is not unique.  ``eigvals`` has no defined order,
        and ``svd``'s singular vectors are fixed only up to the sign of
        each column — so the two devices can return different valid
        answers to the same question, and did: Metal listed the same four
        eigenvalues as the CPU starting from a different one.

        Reported separately rather than passed, because "the same numbers
        arranged differently" is a weaker statement than agreement and the
        finding should say which one was established.
        """
        if a.size < 2 or a.shape != b.shape:
            return False
        return bool(
            np.allclose(
                np.sort(np.abs(a).reshape(-1)),
                np.sort(np.abs(b).reshape(-1)),
                rtol=2e-5,
                atol=1e-6,
                equal_nan=True,
            )
        )

    def _ignores_its_values(self, fn: Any, call: "Call") -> bool:
        """Whether the answer is the same for two different inputs.

        Consulted only once the two devices have already disagreed, where
        it separates "computed something different" from "reported
        something about the device".  ``Tensor.is_metal`` is False on the
        CPU and True on Metal for every input there is, which is the
        correct answer twice rather than a defect.
        """
        try:
            base = call.base
        except TypeError:
            return False
        try:
            first = _probe.to_numpy(fn(*call.with_primary(base).args, **call.kwargs))
            other = _probe.to_numpy(
                fn(*call.with_primary(base * 2.0 + 1.0).args, **call.kwargs)
            )
        except Exception:  # noqa: BLE001 - surveying, not asserting
            return False
        if first is None or other is None or first.shape != other.shape:
            return False
        return bool(np.array_equal(first, other, equal_nan=first.dtype.kind == "f"))

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
                symbol, Status.SKIP, "differentiated argument is not a tensor"
            )

        weights = _probe.covector(64, _probe.SEED_A)

        def scalar(array: np.ndarray) -> float:
            probe = call.with_primary(array)
            return float(_probe.contract(fn(*probe.args, **probe.kwargs), weights))

        # analytic
        #
        # Re-drawn if the gradient comes back identically zero, because
        # that is usually a fact about the sample rather than the op.
        # ``maximum(a, b)`` differentiates to zero wherever ``b`` wins, so
        # a companion that happens to dominate every element leaves
        # nothing for the finite difference to confirm — which is what
        # the two draws of ``moderate`` did here, all six elements of one
        # above all six of the other.  That is a 1-in-64 accident and not
        # a property of ``maximum``, so reporting it as vacuous told the
        # reader about the seed.
        #
        # An op whose gradient really is zero everywhere — ``floor``,
        # ``sign`` — stays vacuous after every retry, and then the verdict
        # is about the op.
        for attempt in range(4):
            if attempt:
                try:
                    base = _probe.sample(domain, base.shape, attempt)
                except Exception:  # noqa: BLE001 - the first draw stands
                    break
            probe = call.with_primary(base)
            x = probe.args[probe.primary]
            try:
                x.requires_grad_(True)
                produced = fn(*probe.args, **probe.kwargs)
                loss = _probe.contract(produced, weights)
                loss.backward()
            except Exception:  # noqa: BLE001 - reported by the real pass below
                break
            if x.grad is None:
                break
            probe_grad = np.asarray(x.grad.numpy(), dtype=np.float64)
            if np.abs(probe_grad).max(initial=0.0) != 0.0:
                break

        probe = call.with_primary(base)
        x = probe.args[probe.primary]
        returned_itself = False
        try:
            x.requires_grad_(True)
            produced = fn(*probe.args, **probe.kwargs)
            # Read here, not after the backward pass: contracting the
            # output against the covector puts a node on ``x`` itself when
            # the op returned ``x``, so asking later always answers "it
            # has one".
            returned_itself = produced is x and x._impl.grad_fn is None
            loss = _probe.contract(produced, weights)
            loss.backward()
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:70]}"
            )
        if x.grad is None:
            return self._finding(
                symbol, Status.UNSUPPORTED, "no gradient reached the input"
            )

        # An op that overwrote its input is asking two different
        # questions of the two methods.  ``nn.init.eye(x)`` writes into
        # ``x`` and leaves it a leaf, so the finite difference measures a
        # function of the *old* values — correctly zero, since the
        # answer does not depend on them — while the analytic pass
        # differentiates the tensor that is now the result.  There is no
        # change to the framework that reconciles those; they are
        # derivatives of different maps.
        #
        # Ops that mutate and *do* leave a node behind are unaffected and
        # still checked: ``exp_`` adopts one whose saved input is the
        # pre-write snapshot, so both methods see the same function.
        # Compared by identity rather than by value: the invocation search
        # has already run the op once on these very arguments, so an
        # in-place one has *already* written its answer and a second call
        # changes nothing.  ``init.eye`` looked like it left its input
        # alone because the input was an identity matrix by the time this
        # ran.
        if returned_itself:
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "overwrites its input and leaves no node — the difference and "
                "the analytic pass differentiate different maps",
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
        # A function of a *constrained* input cannot be differenced
        # coordinate by coordinate.
        #
        # ``eigvalsh`` reads one triangle and takes the matrix to be
        # symmetric, so perturbing an entry in the other triangle changes
        # nothing at all — while its gradient is stated symmetrically and
        # puts half the sensitivity on each of the mirrored pair.  Six of a
        # 4x4's sixteen coordinates come back with a difference of exactly
        # zero against a non-zero derivative, and the rest off by the
        # factor of two that splitting implies.  Lucid's gradient agrees
        # with the reference to the last digit; it is the probe that has
        # stepped off the symmetric matrices, where the function is not
        # defined.
        untouched = (np.abs(coarse.reshape(-1)) < 1e-12) & (np.abs(analytic) > 1e-6)
        if untouched.any():
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                f"{domain}: {int(untouched.sum())} of {analytic.size} coordinates do "
                "not move the result at all — the input is constrained and a "
                "coordinate-wise difference leaves its domain",
                rel=rel,
            )

        # A corner is not a wrong derivative.
        #
        # Where the function has a kink at the probe point, the two
        # one-sided slopes differ and the central difference reports their
        # average, which matches neither.  Nothing about the op is wrong
        # and no step size helps: ``cosine_embedding_loss`` is a hinge, and
        # its analytic gradient agrees with the reference exactly while the
        # difference straddles the corner.
        #
        # Specific, because a *wrong* derivative still has the two
        # one-sided slopes agreeing with each other — they are both
        # measuring the same function.  Only a corner separates them.
        try:
            flat = base.reshape(-1)
            at = scalar(base)
            forward = np.empty(flat.size, dtype=np.float64)
            backward = np.empty(flat.size, dtype=np.float64)
            for i in range(flat.size):
                up, down = flat.copy(), flat.copy()
                up[i] += ctx.step
                down[i] -= ctx.step
                forward[i] = (scalar(up.reshape(base.shape)) - at) / ctx.step
                backward[i] = (at - scalar(down.reshape(base.shape))) / ctx.step
            sided = _probe.relative(forward, backward)
        except Exception:  # noqa: BLE001
            sided = 0.0
        if sided > 1e-3 and np.isfinite(sided):
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                f"{domain}: the one-sided slopes differ by {sided:.2e} — a corner "
                "at the probe, where a central difference is the average of two",
                rel=rel,
                one_sided=sided,
            )

        # Refining made it worse or left it alone.  Before calling that a
        # wrong derivative, try the other direction: a step *below the
        # output's resolution* cannot be refined into agreement, only
        # coarsened.  ``Tensor.half`` casts to float16, where 1e-5 does not
        # move a value near 1 at all — ``f(x+h)`` and ``f(x-h)`` round
        # to the same number and the difference is exactly zero, against an
        # analytic gradient of 1.  Coarsening recovers it; a genuinely
        # wrong formula does not move.
        try:
            blunt = _probe.finite_difference(scalar, base, ctx.step * 1000.0)
            rel_blunt = _probe.relative(analytic, blunt.reshape(analytic.shape))
        except Exception:  # noqa: BLE001
            rel_blunt = rel
        # Judged by how fast it falls, not against the tolerance.  float16
        # is coarse enough that no step reaches 2e-5 — the point is that
        # the disagreement collapses as the step *grows*, which is the
        # opposite of truncation and the signature of a step below the
        # output's resolution.
        if _probe.quadratic_shrink(rel, rel_blunt):
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                f"{domain}: rel {rel:.2e} at h, {rel_blunt:.2e} at 1000h — the step "
                "is below the output's precision, not the derivative wrong",
                rel=rel,
                rel_coarsened=rel_blunt,
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
                symbol, Status.SKIP, "differentiated argument is not a tensor"
            )

        weights = _probe.covector(64, _probe.SEED_A)

        def loss_of(array: np.ndarray) -> "tuple[Any, Any]":
            probe = call.with_primary(array)
            x = probe.args[probe.primary]
            x.requires_grad_(True)
            return x, _probe.contract(fn(*probe.args, **probe.kwargs), weights)

        # Re-drawn on an all-zero reference, for the same reason the grad
        # axis re-draws: a gradient that is zero everywhere leaves the
        # two routes nothing to disagree about, and for a comparison op
        # that is usually the companion having won every element rather
        # than anything about the op.  An op whose gradient really is
        # zero stays zero through every draw.
        reference = np.zeros(0)
        for attempt in range(4):
            if attempt:
                try:
                    base = _probe.sample(domain, base.shape, attempt)
                except Exception:  # noqa: BLE001 - the first draw stands
                    break
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
            if np.abs(reference).max(initial=0.0) != 0.0:
                break
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


def _receiver_position(free_fn: Any, method_fn: Any) -> int:
    """Which of the free function's arguments the method's receiver is.

    Almost always the first — ``x.exp()`` is ``lucid.exp(x)`` — and the
    axis assumed it always.  ``where`` is the exception:
    ``lucid.where(condition, x, y)`` is spelled ``x.where(condition, y)``,
    so the receiver is the *second* free argument.  Passing the arguments
    positionally to both asked two different questions, and the axis
    reported the two different answers as a defect.  The reference has
    the same asymmetry, so this was never Lucid's to fix.

    Derived rather than listed: the method names every free parameter it
    still takes, so the one it does not name is the one it became.
    """
    try:
        free_names = [
            name
            for name, param in inspect.signature(free_fn).parameters.items()
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        ]
        method_names = {name for name in inspect.signature(method_fn).parameters}
    except TypeError, ValueError, NameError:
        return 0
    method_names.discard("self")
    missing = [i for i, name in enumerate(free_names) if name not in method_names]
    if len(missing) == 1:
        return missing[0]
    # The names did not settle it.  With two arguments position 0 is the
    # only reading that does not have the method reversing its operands,
    # which no API does — and that assumption is what found the
    # scalar-coercion gap this axis exists for.  With three there is a
    # real choice and ``where`` makes it: ``lucid.where(cond, x, y)`` is
    # ``x.where(condition, y)``, the receiver in the middle.  Nothing here
    # can tell which, so it says so instead of guessing.
    return 0 if len(free_names) <= 2 else -1


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
                    at = _receiver_position(probe_fn, fn) if label == "method" else 0
                    if at < 0:
                        # The two spellings name their arguments
                        # differently and there is more than one candidate
                        # for the receiver, so there is no alignment to
                        # compare — ``lucid.where(cond, x, y)`` against
                        # ``Tensor.where(self, condition, other)`` shares
                        # not one parameter name.  Guessing position 0 made
                        # the axis call ``where(cond, x, y)`` against
                        # ``cond.where(x, y)``, which are two different
                        # questions, and report the two answers.
                        return self._finding(
                            symbol,
                            Status.NOT_APPLICABLE,
                            "the spellings order their arguments differently and "
                            "share no parameter name — no alignment to compare",
                        )
                    args = (
                        call.args
                        if label != "method"
                        else call.args[:at] + call.args[at + 1 :]
                    )
                    target = call.args[at] if label == "method" else None
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

    @staticmethod
    def _takes_device_argument(fn: Any) -> bool:
        """Whether the callable has a ``device`` parameter of its own."""
        try:
            return (
                "device"
                in inspect.signature(
                    fn, annotation_format=annotationlib.Format.FORWARDREF
                ).parameters
            )
        except Exception:  # noqa: BLE001
            return False

    def _device_kwarg_check(self, symbol: "Symbol", fn: Any, call: Any) -> Finding:
        """Does a factory put its output where ``device=`` says?

        ``zeros``, ``arange``, ``eye``, ``linspace`` and the signal
        windows build a tensor from nothing, so there is no input device
        for the output to follow and the cpu-vs-metal comparison has
        nothing to compare.  They do take a ``device`` argument, though,
        and whether they honour it is a real question that was going
        unasked — twenty-odd symbols reported as skips.
        """
        try:
            parameters = inspect.signature(
                fn, annotation_format=annotationlib.Format.FORWARDREF
            ).parameters
        except Exception:  # noqa: BLE001
            return self._finding(
                symbol, Status.NOT_APPLICABLE, "builds from no tensor input"
            )
        if "device" not in parameters:
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "builds from no tensor input and takes no device argument",
            )
        try:
            out = fn(*call.args, **{**call.kwargs, "device": "metal"})
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"device='metal': {type(exc).__name__}"
            )
        device = getattr(out, "device", None)
        if device is None:
            return self._finding(symbol, Status.SKIP, "returned no tensor")
        if "metal" not in str(device):
            return self._finding(
                symbol,
                Status.FAIL,
                f"asked for device='metal' and built the result on {device}",
            )
        return self._finding(symbol, Status.PASS, "honours its device argument")

    def run(self, symbol: "Symbol", ctx: Context) -> Finding:
        if not ctx.metal:
            return self._finding(symbol, Status.SKIP, "metal unavailable")
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")
        call, domain, _ = self._working_call(fn, symbol, ctx)
        if call is None:
            return self._finding(symbol, Status.SKIP, "no candidate invocation ran")
        if self._draws_randomly(fn, call):
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "two identical calls disagree — this measures the draw, not the op",
            )

        moved_any = False

        def on(device: str, override: np.ndarray | None = None) -> Any:
            nonlocal moved_any
            args = []
            for i, a in enumerate(call.args):
                if hasattr(a, "to"):
                    moved_any = True
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

        # A tensor that quietly stayed on the CPU makes every comparison
        # below trivially true, so the landing device is checked first.
        # What it *means* depends on whether anything was moved.
        #
        # If the call had tensor arguments and they were all moved, an
        # output on the other device is a defect and not an unanswerable
        # question: ``linalg.matrix_rank`` and ``histc`` each took a Metal
        # matrix and returned a CPU tensor, and the next op raised
        # DeviceMismatch.  Reported as SKIP, both sat unnoticed among 45
        # cells that were mostly factories.
        #
        # A factory has no tensor argument to move, so its output follows
        # the default device and that is correct.  The question worth
        # asking there is a different one — whether it honours its own
        # ``device=`` argument — and it is asked below.
        for out, want in ((cpu_out, "cpu"), (metal_out, "metal")):
            device = getattr(out, "device", None)
            if device is None or want in str(device):
                continue
            # An op that takes a ``device`` argument decides its own
            # output device, whatever its inputs are on.  Checked before
            # ``moved_any``, because a tensor argument is not proof of a
            # transform: ``signal.windows.general_cosine(M, a)`` reads
            # ``a`` as plain coefficients — ``float(a[k])`` — so moving
            # it to Metal moves nothing that reaches the output, and
            # calling that a failure blames the op for the probe.
            if self._takes_device_argument(fn):
                return self._device_kwarg_check(symbol, fn, call)
            if moved_any:
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"every tensor input was moved to {want}, and the output "
                    f"came back on {device}",
                )
            return self._finding(
                symbol, Status.NOT_APPLICABLE, "builds from no tensor input"
            )

        a, b = _probe.to_numpy(cpu_out), _probe.to_numpy(metal_out)
        if a is None or b is None or a.shape != b.shape:
            return self._finding(symbol, Status.SKIP, "outputs not comparable")
        af, bf = self._comparable(a), self._comparable(b)
        if not np.allclose(af, bf, rtol=2e-5, atol=1e-6, equal_nan=True):
            if self._ignores_its_values(fn, call):
                return self._finding(
                    symbol,
                    Status.NOT_APPLICABLE,
                    "the answer does not depend on the values — this reports "
                    "the device, and both answers are right",
                )
            if self._same_multiset(af, bf):
                return self._finding(
                    symbol,
                    Status.NOT_APPLICABLE,
                    "same magnitudes in a different arrangement — a "
                    "decomposition has no canonical order or sign",
                )
            return self._finding(
                symbol,
                Status.FAIL,
                f"{domain}: cpu and metal differ by {np.nanmax(np.abs(af - bf)):.3e}",
            )

        # The probe the old sweeps did not carry — but only where a NaN
        # means anything.  The substitution rewrites the *primary*
        # argument, and for ``one_hot`` that argument is a tensor of class
        # indices: replacing it with NaN and casting to float asks the two
        # backends what the zeroth column of a NaN-th class is, and they
        # answered differently because the question has no answer.
        primary = call.args[call.primary] if call.args else None
        if primary is None or not str(getattr(primary, "dtype", "")).endswith(
            ("float16", "float32", "float64")
        ):
            return self._finding(
                symbol,
                Status.PASS,
                f"{domain}: finite inputs agree (primary is not a float — "
                "a non-finite probe would not mean anything)",
            )
        try:
            nan_cpu = _probe.to_numpy(on("cpu", _probe.NON_FINITE))
            nan_metal = _probe.to_numpy(on("metal", _probe.NON_FINITE))
        except Exception:  # noqa: BLE001
            return self._finding(symbol, Status.PASS, f"{domain}: finite inputs agree")
        if nan_cpu is None or nan_metal is None or nan_cpu.shape != nan_metal.shape:
            return self._finding(symbol, Status.PASS, f"{domain}: finite inputs agree")
        if not np.allclose(
            self._comparable(nan_cpu),
            self._comparable(nan_metal),
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
        if self._draws_randomly(fn, call):
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "two identical calls disagree — this measures the draw, not the op",
            )

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
        follows_first = True
        follows_second = True
        for sa, sb in self._PAIRS:
            a = _probe.as_f64(_probe.rng(1).uniform(0.5, 1.5, sa))
            b = _probe.as_f64(_probe.rng(2).uniform(0.5, 1.5, sb))
            want = np.broadcast_shapes(sa, sb)
            try:
                got = _probe.to_numpy(fn(a, b))
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{sa}x{sb}: {type(exc).__name__}")
                follows_first = follows_second = False
                continue
            if got is None:
                continue
            follows_first &= tuple(got.shape) == sa
            follows_second &= tuple(got.shape) == sb
            if tuple(got.shape) != want:
                failures.append(f"{sa}x{sb} -> {got.shape}, expected {want}")
        # ...and being elementwise on equal shapes is not the same as
        # broadcasting.  ``isin`` asks "is each element of a somewhere in
        # b", so its answer has *a*'s shape whatever b's is; ``selu_``
        # writes into its first operand and cannot change shape;
        # ``new_tensor`` builds from the second and takes that one.  Each
        # passes the equal-shape test above and none of them broadcasts.
        #
        # Read off the results rather than listed: an op whose output
        # follows one operand in *every* direction is shaped by that
        # operand, not by the pair.
        if failures and (follows_first or follows_second):
            which = "first" if follows_first else "second"
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                f"the output follows the {which} operand in every direction — "
                "shaped by one argument, not broadcast between them",
            )
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
                    kwargs = _probe.dtype_kwargs(call, build)
                    if _probe.to_numpy(fn(*args, **kwargs)) is not None:
                        support[device].add(name)
                except Exception:  # noqa: BLE001
                    continue

        if not any(support.values()):
            return self._finding(symbol, Status.SKIP, "no dtype accepted")
        if len(devices) == 2:
            # An op whose *result* is float64 whatever it was handed cannot
            # exist on Metal at all — `Tensor.double` and `float_power` say
            # so in their names.  Every input dtype then reads as
            # cpu-only, which is true and is not a defect: the question the
            # axis asks does not apply to them.
            if _produces_float64(fn, call):
                return self._finding(
                    symbol,
                    Status.NOT_APPLICABLE,
                    "result is float64 by definition, which Metal has no dtype for",
                )
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
                # Only the *primary* was emptied.  Where another tensor
                # argument is still full size, it is the one supplying the
                # shape and nothing was invented: ``solve_ex(A_empty, B)``
                # answers with B's batch, and ``new_tensor`` builds from
                # its data argument and takes that.  Emptiness can only
                # propagate from an operand the answer's shape depends on.
                others = [
                    a
                    for i, a in enumerate(call.args)
                    if i != call.primary and hasattr(a, "shape")
                ]
                if any(int(np.prod(tuple(a.shape))) != 0 for a in others):
                    return self._finding(
                        symbol,
                        Status.NOT_APPLICABLE,
                        "another operand is still full size and supplies the shape",
                    )
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

        # A container holds layers and has no forward of its own —
        # ``ModuleList`` and ``ParameterDict`` raise NotImplementedError by
        # design, and the base ``Module`` does too.  Running the lifecycle
        # against them reported a shape probe failing to find an input for
        # something that never accepts one.
        if getattr(type(module), "forward", None) is getattr(
            lucid.nn.Module, "forward", None
        ):
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "a container: it holds layers and defines no forward",
            )

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


def _produces_float64(fn: Any, call: "Call") -> bool:
    """Whether the op answers in float64 from a float32 input.

    The discriminator for "this op cannot exist on Metal".  Asked by
    running it rather than by matching on the name, so a future
    ``as_double`` is caught without being listed anywhere.
    """

    def build(array: np.ndarray, follow: bool) -> Any:
        return lucid.tensor(
            np.ascontiguousarray(array.astype(np.float32) if follow else array),
            dtype=lucid.float32 if follow else None,
            device="cpu",
        )

    try:
        out = fn(*_probe.dtype_args(call, "float32", build), **call.kwargs)
    except Exception:  # noqa: BLE001
        return False
    first = out[0] if isinstance(out, tuple | list) and out else out
    return str(getattr(first, "dtype", "")).endswith("float64")


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


def _ctor_args_with_qconfig() -> "list[tuple[tuple[Any, ...], dict[str, Any]]]":
    """The ladder again, each rung carrying a qconfig.

    ``qat.Conv1d`` / ``2d`` / ``3d`` and their fused ReLU variants
    declare ``(*args, **kwargs)``, so a signature says nothing about them
    and only the ladder can reach them — and the plain ladder is refused
    with "requires a qconfig".  ``qat.Conv2d(3, 3, 3, qconfig=...)``
    builds.
    """
    try:
        qconfig = _default_qconfig()
    except Exception:  # noqa: BLE001
        return []
    return [(args, {**kwargs, "qconfig": qconfig}) for args, kwargs in _CTOR_ARGS]


#: Constructor arguments by parameter name.
#:
#: ``nn`` layer constructors are almost entirely small integers with
#: self-describing names, which is what makes deriving them possible at
#: all.  Sizes agree with :data:`_FORWARD_SHAPES` so that a module built
#: here has a chance of accepting one of the inputs tried against it.
_CTOR_BY_NAME: "dict[str, Any]" = {
    # widths, all agreeing with the (2, 4) / (2, 3, 6, 6) probe inputs
    "in_features": 4,
    "out_features": 4,
    "in1_features": 4,
    "in2_features": 4,
    "in_channels": 3,
    "out_channels": 3,
    "num_features": 4,
    "num_channels": 4,
    "channels": 4,
    "features": 4,
    "hidden_size": 4,
    "input_size": 4,
    "embed_dim": 4,
    "embedding_dim": 4,
    "d_model": 4,
    "dim": 4,
    "head_dim": 2,
    "size": 4,
    "num_embeddings": 8,
    "vocab_size": 8,
    "normalized_shape": 4,
    "unflattened_size": (2, 2),
    "output_size": 2,
    "num_positions": 8,
    "max_position_embeddings": 8,
    "height": 6,
    "width": 6,
    # counts
    "num_heads": 2,
    "nhead": 2,
    "num_kv_heads": 2,
    "num_layers": 1,
    "num_classes": 4,
    "groups": 1,
    "num_groups": 1,
    "upscale_factor": 2,
    "downscale_factor": 2,
    # windowing
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
    "output_padding": 0,
    "dilation": 1,
    # scalars
    "eps": 1e-5,
    "p": 0.5,
    "bias": True,
    "momentum": 0.1,
    "affine": True,
}


def _default_qconfig() -> Any:
    """A QConfig, built on demand.

    The quantisation layers declare ``qconfig`` with a default of
    ``None`` and then refuse it: ``qat.Linear requires a qconfig``.  A
    signature-derived constructor honours the author's default and so
    never supplied one, which left the whole ``qat`` / ``intrinsic.qat``
    stack — fifteen classes — unconstructible and therefore unaudited.
    """
    import lucid.quantization as quantization

    return quantization.QConfig(
        quantization.MinMaxObserver, quantization.MinMaxObserver
    )


def _default_qscheme() -> Any:
    import lucid.quantization as quantization

    return quantization.QScheme.PER_TENSOR_AFFINE


def _default_qdtype() -> Any:
    import lucid.quantization as quantization

    return quantization.quint8


def _default_qparams(kind: str) -> Any:
    """A ``scale`` or ``zero_point`` for the quantised activations."""
    import numpy as _np

    if kind == "scale":
        return _probe.as_f32(_np.array(0.1))
    return _probe.as_int(_np.array(0))


def _default_submodule(name: str) -> Any:
    """A concrete layer for a parameter annotated as the ``Module`` base.

    The fused ``intrinsic`` layers take the layers they fuse, and the
    annotation names the base class — which says nothing about what
    would actually work.  The parameter name does.
    """
    import lucid.nn as nn

    return {
        "conv": lambda: nn.Conv2d(3, 3, 3, padding=1),
        "bn": lambda: nn.BatchNorm2d(3),
        "relu": lambda: nn.ReLU(),
        "linear": lambda: nn.Linear(4, 4),
        "encoder_layer": lambda: nn.TransformerEncoderLayer(4, 2),
        "decoder_layer": lambda: nn.TransformerDecoderLayer(4, 2),
    }[name]()


#: Constructor arguments that have to be built rather than named, keyed
#: on the parameter name.  Kept separate from :data:`_CTOR_BY_NAME` so
#: that table stays a table of constants and nothing is constructed at
#: import time.
_CTOR_FACTORY: "dict[str, Any]" = {
    "qconfig": _default_qconfig,
    "scale": lambda: _default_qparams("scale"),
    "zero_point": lambda: _default_qparams("zero_point"),
    "conv": lambda: _default_submodule("conv"),
    "bn": lambda: _default_submodule("bn"),
    "relu": lambda: _default_submodule("relu"),
    "linear": lambda: _default_submodule("linear"),
    "encoder_layer": lambda: _default_submodule("encoder_layer"),
    "decoder_layer": lambda: _default_submodule("decoder_layer"),
    # ``QScheme`` is an enum and answers to the enum rule; ``QDtype`` is a
    # dataclass whose instances are module-level constants, so there is
    # nothing on the class to enumerate and the name has to say it.
    "qscheme": _default_qscheme,
    "qdtype": _default_qdtype,
}


def _ctor_value(name: str, annotation: Any, depth: int) -> Any:
    """One constructor argument, by name first and annotation second.

    Raises ``KeyError`` when neither says anything, so the caller can
    fall back to the fixed ladder rather than pass a wrong-typed value
    and read the resulting ``TypeError`` as "this class is unbuildable".
    """
    if name in _CTOR_BY_NAME:
        return _CTOR_BY_NAME[name]
    if name in _CTOR_FACTORY:
        try:
            return _CTOR_FACTORY[name]()
        except Exception as exc:  # noqa: BLE001
            raise KeyError(name) from exc

    if isinstance(annotation, type):
        # A sub-module, which is how the fused ``intrinsic`` layers are
        # spelled: ``ConvReLU2d(conv, relu)`` takes the two layers it
        # fuses.  Built one level deep, so a cycle cannot run away.
        if depth < 1 and _is_module_class(annotation):
            built = _construct_module(annotation, depth + 1)
            if built is None:
                raise KeyError(name)
            return built
        if issubclass(annotation, bool):
            return True
        if issubclass(annotation, int):
            return 4
        if issubclass(annotation, float):
            return 0.5
        members = list(getattr(annotation, "__members__", {}).values())
        if members:  # an Enum: the first member is as good as any
            return members[0]

    text = str(annotation)
    if "int" in text and "tuple" in text:
        return 1
    if text.endswith("int") or "_Size" in text:
        return 2
    raise KeyError(name)


def _is_module_class(obj: Any) -> bool:
    try:
        import lucid.nn as _nn

        return isinstance(obj, type) and issubclass(obj, _nn.Module)
    except Exception:  # noqa: BLE001
        return False


def _construct_module(cls: Any, depth: int = 0) -> Any:
    """Build ``cls`` from its own signature, or ``None``.

    The ladder this replaces tried eight fixed argument tuples —
    ``(4,)``, ``(4, 4)``, ``(3, 4, 3)`` and so on — and reported
    everything it could not hit as "no constructor signature worked".
    That was 114 of the framework's ``nn.Module`` classes, every one of
    them then contributing nothing to any axis: not the module lifecycle,
    not the smoke pass, nothing.  A class that is never constructed is a
    class the audit cannot notice a regression in.

    Reading the signature works here because ``nn`` constructors are
    almost entirely small integers with self-describing names.  Where the
    name says nothing the annotation usually does, and where neither does
    the fixed ladder still runs underneath.
    """
    try:
        # ``FORWARDREF`` rather than the default.
        #
        # H7 requires annotations that exist only under ``TYPE_CHECKING``,
        # and PEP 649 leaves those unevaluated until something asks.
        # ``inspect.signature`` asking is what raised ``NameError: name
        # 'QConfig' is not defined`` on every quantised layer — not a
        # defect in the layer, this function reading it wrongly.
        # ``FORWARDREF`` resolves the names it can and returns the rest as
        # forward references, which the annotation fallback below reads as
        # text in any case.
        signature = inspect.signature(
            cls, annotation_format=annotationlib.Format.FORWARDREF
        )
    except Exception:  # noqa: BLE001 - an unreadable signature is not a finding
        return None

    kwargs: "dict[str, Any]" = {}
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if parameter.default is not inspect.Parameter.empty:
            # ...unless the author's default is one the class then
            # rejects.  ``qconfig`` defaults to ``None`` and every
            # quantisation layer raises "requires a qconfig" on it.
            if parameter.name == "qconfig" and parameter.default is None:
                try:
                    kwargs["qconfig"] = _default_qconfig()
                except Exception:  # noqa: BLE001
                    pass
            continue  # a default the author chose is better than a guess
        try:
            kwargs[parameter.name] = _ctor_value(
                parameter.name, parameter.annotation, depth
            )
        except KeyError:
            return None

    try:
        return cls(**kwargs)
    except Exception:  # noqa: BLE001 - the fixed ladder is the fallback
        return None


def _try_construct(cls: Any) -> Any:
    derived = _construct_module(cls)
    if derived is not None:
        return derived
    for args, kwargs in list(_CTOR_ARGS) + _ctor_args_with_qconfig():
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


def _module_input_shapes(module: Any) -> "list[tuple[int, ...]]":
    """Shapes this particular module might accept, its own answer first.

    A constructed module knows its input width — ``Linear(4, 4)`` wants
    a trailing 4, ``Conv2d(3, ...)`` wants 3 channels — and asking it is
    the difference between probing and guessing.  Five fixed shapes left
    84 modules stuck at ``forward`` with ShapeMismatch after they had
    already been built successfully, which is the least useful place to
    stop: the object exists and nothing is asked of it.

    Its own answer comes from the declared width where the module
    exposes one, and from the first parameter otherwise — a weight of
    rank 2 is ``(out, in)`` and one of rank 4 is ``(out, in, kh, kw)``,
    so in both the second axis is what the input must supply.
    """
    widths: "list[int]" = []
    for attribute in ("in_features", "in_channels", "num_features", "embedding_dim"):
        value = getattr(module, attribute, None)
        if isinstance(value, int) and 0 < value <= 64:
            widths.append(value)
    normalized = getattr(module, "normalized_shape", None)
    if isinstance(normalized, int):
        widths.append(normalized)
    elif isinstance(normalized, (tuple, list)) and normalized:
        widths.append(int(normalized[-1]))

    shapes: "list[tuple[int, ...]]" = []
    for width in widths:
        shapes += [(2, width), (2, width, 6), (2, width, 6, 6), (2, 6, width)]

    try:
        first = next(iter(module.parameters()), None)
    except Exception:  # noqa: BLE001
        first = None
    if first is not None and hasattr(first, "shape"):
        dims = tuple(int(d) for d in first.shape)
        if len(dims) == 2:
            shapes += [(2, dims[1]), (2, 6, dims[1])]
        elif len(dims) == 3:
            shapes.append((2, dims[1], 6))
        elif len(dims) == 4:
            shapes.append((2, dims[1], 6, 6))
        elif len(dims) == 5:
            shapes.append((2, dims[1], 4, 6, 6))
        elif len(dims) == 1:
            shapes += [(2, dims[0]), (2, dims[0], 6, 6)]

    shapes += list(_FORWARD_SHAPES)
    seen: "set[tuple[int, ...]]" = set()
    return [sh for sh in shapes if not (sh in seen or seen.add(sh))]


def _forward_inputs(module: Any, shape: "tuple[int, ...]") -> "list[list[Any]]":
    """Argument lists to try for one shape.

    Three things the single float64 tensor could not reach: a module
    whose forward takes a pair (every loss: ``input`` and ``target``) or
    a triple (attention: query, key, value); one whose parameters are
    float32, where a float64 probe is a DtypeMismatch and not a defect;
    and an embedding, which indexes and wants integers.
    """
    if getattr(module, "num_embeddings", None) is not None:
        rows = int(module.num_embeddings)
        idx = _probe.as_int(_probe.rng(_probe.SEED_B).integers(0, rows, shape[:2]))
        return [[idx]]

    dtypes = [_probe.as_f64]
    try:
        first = next(iter(module.parameters()), None)
        if first is not None and str(first.dtype).endswith("float32"):
            dtypes.insert(0, _probe.as_f32)
    except Exception:  # noqa: BLE001
        pass

    out: "list[list[Any]]" = []
    for cast in dtypes:
        one = cast(_probe.sample("moderate", shape, 0))
        two = cast(_probe.sample("moderate", shape, 1))
        three = cast(_probe.sample("moderate", shape, 2))
        out += [[one], [one, two], [one, two, three]]

        # A classification loss wants class *indices*, not a second float
        # tensor of the same shape: ``CrossEntropyLoss``, ``NLLLoss``,
        # ``CTCLoss`` and the margin losses all read the second argument
        # as an integer label per row and raised TypeError on a float one.
        if len(shape) >= 2:
            classes = int(shape[-1])
            labels = _probe.as_int(
                _probe.rng(_probe.SEED_B).integers(0, max(classes, 1), (shape[0],))
            )
            out.append([one, labels])
            # ...and the pair losses want a +/-1 sign per row.
            signs = cast(_probe.rng(_probe.SEED_A).choice([-1.0, 1.0], (shape[0],)))
            out.append([one, two, signs])
    return out


def _try_forward(module: Any) -> "tuple[Any, str]":
    last = "no input shape worked"
    for shape in _module_input_shapes(module):
        for args in _forward_inputs(module, shape):
            try:
                out = module(*args)
            except Exception as exc:  # noqa: BLE001
                last = f"{shape}: {type(exc).__name__}"
                continue
            tensor = out if hasattr(out, "shape") else None
            if tensor is None and isinstance(out, (tuple, list)) and out:
                # RNNs and attention return ``(output, state)``; the
                # first element is the one the axes can measure.
                tensor = out[0] if hasattr(out[0], "shape") else None
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
