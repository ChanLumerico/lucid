"""Input construction: domains, probes and the numeric helpers.

Two things here are load-bearing.

**Domains.**  Ops are defined on different sets, and guessing per-op by
name is its own source of error, so a check tries a ladder of domains and
keeps the first on which the forward and backward passes are both finite.
The ladder deliberately avoids sampling near a pole: a draw that lands on
``x = 6e-4`` makes ``1/x`` and ``psi`` disagree with central differences
at 1e-4 for reasons that have nothing to do with either op.

**Seeds.**  Outputs are contracted against a *fixed random* covector,
never ``ones``.  A constant seed makes softmax's gradient identically
zero, so a finite-difference check passes while testing nothing — the
failure mode the VACUOUS status exists to name.
"""

import math
from typing import TYPE_CHECKING, Any

import numpy as np

import lucid

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

#: Default probe shape.  Small enough that a per-element finite-difference
#: sweep is cheap, large enough to exercise broadcasting and reductions.
SHAPE: tuple[int, int] = (2, 3)

#: Where the covectors come from.  Fixed so a run is reproducible and a
#: finding can be re-derived by hand.
SEED_A = 11
SEED_B = 23
SEED_X = 7


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ── domains ──────────────────────────────────────────────────────────────────
# Tried in order.  ``moderate`` first because most ops accept it; the rest
# narrow onto the sets that log / sqrt / acosh / atanh need.  None of them
# samples within 0.05 of a pole.

DOMAINS: dict[str, "Callable[[Sequence[int]], np.ndarray]"] = {
    "moderate": lambda s: rng(SEED_X).uniform(-1.2, 1.2, s) + 0.15,
    "positive": lambda s: rng(SEED_X).uniform(0.35, 1.75, s),
    "unit": lambda s: rng(SEED_X).uniform(-0.75, 0.75, s),
    "gt_one": lambda s: rng(SEED_X).uniform(1.3, 2.6, s),
    "small_pos": lambda s: rng(SEED_X).uniform(0.08, 0.48, s),
    "wide": lambda s: rng(SEED_X).standard_normal(s) * 2.0,
}

#: The probe that finds NaN-swallowing.  ``relu`` returned 0 for the first
#: entry on the CPU and NaN on Metal; no well-conditioned sample can show
#: that, which is why device parity needs its own vector.
NON_FINITE = np.array([np.nan, np.inf, -np.inf, -1.0, 0.0, 2.0], dtype=np.float64)

#: Every dtype the framework claims, for the coverage axis.
DTYPES: tuple[str, ...] = (
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
)


def dtype_of(name: str) -> Any:
    return getattr(lucid, name, None)


# ── tensor factories ─────────────────────────────────────────────────────────


def as_f64(array: Any, device: str = "cpu") -> Any:
    """A float64 tensor — the working precision for every numeric check."""
    return lucid.tensor(
        np.ascontiguousarray(np.asarray(array, dtype=np.float64)),
        dtype=lucid.float64,
        device=device,
    )


def as_f32(array: Any, device: str = "cpu") -> Any:
    return lucid.tensor(
        np.ascontiguousarray(np.asarray(array, dtype=np.float32)), device=device
    )


def as_int(array: Any, device: str = "cpu") -> Any:
    return lucid.tensor(
        np.ascontiguousarray(np.asarray(array, dtype=np.int64)),
        dtype=lucid.int64,
        device=device,
    )


def covector(n: int, seed: int = SEED_A) -> np.ndarray:
    """The weights an output is contracted against.

    Random, never constant.  See the module docstring.
    """
    return rng(seed).standard_normal(max(n, 1))


def sample(domain: str, shape: "Sequence[int]" = SHAPE) -> np.ndarray:
    return np.asarray(DOMAINS[domain](tuple(shape)), dtype=np.float64)


# ── numeric helpers ──────────────────────────────────────────────────────────


def to_numpy(value: Any) -> np.ndarray | None:
    """Whatever an op returned, as an array, or ``None``.

    Not only tensors.  ``x.ndim``, ``lucid.is_floating_point(x)`` and
    ``x.size()`` answer with a plain int, bool or tuple, and a survey that
    only recognises tensors reports every predicate and every shape
    accessor as uncallable — 30-odd symbols marked as gaps that were
    working perfectly.
    """
    if value is None:
        return None
    if hasattr(value, "numpy"):
        try:
            return np.asarray(value.numpy())
        except Exception:  # noqa: BLE001 - surveying, not asserting
            return None
    if isinstance(value, bool | int | float | complex):
        return np.asarray(value)
    if isinstance(value, tuple | list):
        if value and all(isinstance(v, bool | int | float) for v in value):
            return np.asarray(value)
        for item in value:
            got = to_numpy(item)
            if got is not None:
                return got
        return None
    return None


def contract(out: Any, weights: np.ndarray) -> Any:
    """Reduce any output to a differentiable scalar against ``weights``."""
    if isinstance(out, tuple | list):
        out = next((o for o in out if hasattr(o, "shape")), None)
    if out is None or not hasattr(out, "shape"):
        raise TypeError("op did not return a tensor")
    flat = out.reshape(-1)
    n = int(flat.shape[0])
    if n == 0:
        raise ValueError("op returned an empty tensor")
    return (flat * as_f64(weights[:n])).sum()


def relative(a: np.ndarray, b: np.ndarray) -> float:
    """Scale-free disagreement, safe on zeros."""
    scale = max(
        float(np.abs(a).max(initial=0.0)), float(np.abs(b).max(initial=0.0)), 1e-12
    )
    return float(np.abs(a - b).max(initial=0.0) / scale)


def quadratic_shrink(coarse: float, fine: float, ratio: float = 10.0) -> bool:
    """Whether refining the step shrank the disagreement like ``h**2``.

    The discriminator between a finite-difference artefact and a real
    formula error.  Truncation falls by ``ratio**2`` when ``h`` falls by
    ``ratio``; a wrong derivative does not move.  A generous band, because
    round-off starts to dominate once the disagreement approaches
    ``sqrt(eps)``.
    """
    if not math.isfinite(coarse) or not math.isfinite(fine) or coarse <= 0.0:
        return False
    if fine <= 1e-11:
        return True
    return (coarse / fine) > (ratio**1.5)


def finite_difference(
    evaluate: "Callable[[np.ndarray], float]",
    base: np.ndarray,
    step: float,
) -> np.ndarray:
    """Central differences of a scalar functional, one entry at a time."""
    flat = base.reshape(-1)
    out = np.empty(flat.size, dtype=np.float64)
    for i in range(flat.size):
        plus, minus = flat.copy(), flat.copy()
        plus[i] += step
        minus[i] -= step
        out[i] = (
            evaluate(plus.reshape(base.shape)) - evaluate(minus.reshape(base.shape))
        ) / (2.0 * step)
    return out


def metal_available() -> bool:
    """Whether the device axis can run at all."""
    try:
        lucid.tensor(np.zeros((1,), dtype=np.float32), device="metal")
    except Exception:  # noqa: BLE001
        return False
    return True


__all__ = [
    "DOMAINS",
    "DTYPES",
    "NON_FINITE",
    "SHAPE",
    "as_f32",
    "as_f64",
    "as_int",
    "contract",
    "covector",
    "dtype_of",
    "finite_difference",
    "metal_available",
    "quadratic_shrink",
    "relative",
    "rng",
    "sample",
    "to_numpy",
]
