"""Statistical composite ops: quantile, cov, corrcoef, cdist."""

import math
import random
from typing import Sequence, TYPE_CHECKING

import lucid
from lucid._C import engine as _C_engine

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

# ── quantile helpers ──────────────────────────────────────────────────────────


def _quantile_sorted(
    sorted_x: Tensor,
    q_list: Sequence[float],
    dim: int | None,
    keepdim: bool,
    interpolation: str,
    n: int,
) -> list[Tensor]:
    """Core quantile computation on an already-sorted tensor."""
    results = []
    for qi in q_list:
        idx_f = qi * (n - 1)
        lo = int(math.floor(idx_f))
        hi = min(lo + 1, n - 1)
        frac = idx_f - lo

        if dim is None:
            val_lo = sorted_x[lo]
            val_hi = sorted_x[hi]
        else:
            lo_t = lucid.tensor([lo], dtype=lucid.int64)
            hi_t = lucid.tensor([hi], dtype=lucid.int64)
            val_lo = sorted_x.index_select(dim, lo_t).squeeze(dim)
            val_hi = sorted_x.index_select(dim, hi_t).squeeze(dim)

        if interpolation == "linear":
            val = val_lo * (1.0 - frac) + val_hi * frac
        elif interpolation == "lower":
            val = val_lo
        elif interpolation == "higher":
            val = val_hi
        elif interpolation == "midpoint":
            val = (val_lo + val_hi) * 0.5
        elif interpolation == "nearest":
            val = (
                sorted_x[round(idx_f)]
                if dim is None
                else sorted_x.index_select(
                    dim, lucid.tensor([round(idx_f)], dtype=lucid.int64)
                ).squeeze(dim)
            )
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation!r}")

        if keepdim and dim is not None:
            val = val.unsqueeze(dim)
        results.append(val)
    return results


def _parse_q(q: float | Sequence[float] | Tensor) -> tuple[list[float], bool]:
    """Return (q_list of floats, scalar_flag)."""
    if isinstance(q, lucid.Tensor):
        if q.ndim == 0:
            return [float(q.item())], True
        return [float(q[i].item()) for i in range(q.shape[0])], False
    elif hasattr(q, "__iter__"):
        return [float(v) for v in q], False
    else:
        return [float(q)], True


# ── public API ────────────────────────────────────────────────────────────────


def quantile(
    input: Tensor,
    q: float | Sequence[float] | Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    interpolation: str = "linear",
) -> Tensor:
    """Compute the *q*-th quantile of ``input``.

    Supports ``interpolation`` in ``{'linear','lower','higher','midpoint','nearest'}``.
    """
    q_list, scalar_q = _parse_q(q)

    if dim is None:
        flat = input.reshape(-1)
        sorted_x = lucid.sort(flat)
        n = flat.shape[0]
        results = _quantile_sorted(sorted_x, q_list, None, keepdim, interpolation, n)
    else:
        sorted_x = lucid.sort(input, dim)
        n = input.shape[dim]
        results = _quantile_sorted(sorted_x, q_list, dim, keepdim, interpolation, n)

    if scalar_q:
        return results[0]
    return lucid.stack(results)


def nanquantile(
    input: Tensor,
    q: float | Sequence[float] | Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    interpolation: str = "linear",
) -> Tensor:
    """Like :func:`quantile` but ignores NaN values."""
    # Replace NaN with +inf so they sort to the end, then count valid entries.
    nan_mask = lucid.isnan(input)
    safe = lucid.where(nan_mask, lucid.full_like(input, math.inf), input)
    q_list, scalar_q = _parse_q(q)

    if dim is None:
        flat_safe = safe.reshape(-1)
        flat_nan = nan_mask.reshape(-1)
        sorted_x = lucid.sort(flat_safe)
        # Count non-NaN elements
        non_nan_vals = lucid.where(
            flat_nan, lucid.full_like(flat_safe, 0.0), lucid.full_like(flat_safe, 1.0)
        )
        n = int(non_nan_vals.sum().item())
        if n == 0:
            nan_val = lucid.tensor(math.nan, dtype=input.dtype)
            if scalar_q:
                return nan_val
            return lucid.stack([nan_val] * len(q_list))
        results = _quantile_sorted(sorted_x, q_list, None, keepdim, interpolation, n)
    else:
        sorted_x = lucid.sort(safe, dim)
        non_nan_count = lucid.where(
            nan_mask,
            lucid.full_like(safe, 0.0),
            lucid.full_like(safe, 1.0),
        ).sum(dim=dim, keepdim=True)
        # Use the mean non-NaN count along this dim as n
        n = int(non_nan_count.mean().item())
        if n == 0:
            out_shape = list(input.shape)
            out_shape[dim] = len(q_list)
            nan_val = lucid.full(out_shape, math.nan, dtype=input.dtype)
            if scalar_q:
                return nan_val.squeeze(dim)
            return nan_val
        results = _quantile_sorted(sorted_x, q_list, dim, keepdim, interpolation, n)

    if scalar_q:
        return results[0]
    return lucid.stack(results)


# ── covariance & correlation ──────────────────────────────────────────────────


def cov(input: Tensor, correction: int = 1) -> Tensor:
    """Covariance matrix of ``input``.

    ``input`` has shape ``(N,)`` or ``(N, M)`` where *N* = variables,
    *M* = observations.  Returns an ``(N, N)`` matrix.
    """
    if input.ndim == 1:
        x = input.unsqueeze(0)  # (1, M)
    else:
        x = input  # (N, M)

    n_obs = x.shape[1]
    mean = x.sum(dim=1, keepdim=True) * (1.0 / n_obs)  # (N, 1)
    xc = x - mean  # (N, M)
    denom = float(max(1, n_obs - correction))
    return lucid.matmul(xc, xc.swapaxes(-1, -2)) * (1.0 / denom)  # (N, N)


def corrcoef(input: Tensor) -> Tensor:
    """Pearson correlation matrix of ``input`` (shape ``(N,)`` or ``(N, M)``)."""
    c = cov(input, correction=1)  # (N, N)
    d = lucid.sqrt(lucid.diagonal(c))  # (N,) — std deviations
    # Outer product: d_col * d_row gives the normalisation matrix
    d_col = d.reshape(-1, 1)  # (N, 1)
    d_row = d.reshape(1, -1)  # (1, N)
    return c / (d_col * d_row)


# ── pairwise distances ────────────────────────────────────────────────────────


def cdist(x1: Tensor, x2: Tensor, p: float = 2.0) -> Tensor:
    """Pairwise distance matrix between rows of ``x1`` and ``x2``.

    ``x1`` shape ``(..., P, M)``, ``x2`` shape ``(..., R, M)``.
    Returns shape ``(..., P, R)``.
    """
    if p == 2.0:
        # ||a - b||² = ||a||² + ||b||² − 2 a·bᵀ  (numerically stable)
        x1_sq = (x1 * x1).sum(dim=-1, keepdim=True)  # (..., P, 1)
        x2_sq = (x2 * x2).sum(dim=-1, keepdim=True)  # (..., R, 1)
        cross = lucid.matmul(x1, x2.swapaxes(-1, -2))  # (..., P, R)
        sq = x1_sq + x2_sq.swapaxes(-1, -2) - 2.0 * cross
        sq = lucid.clamp(sq, 0.0, float("inf"))  # clamp numerical noise
        return lucid.sqrt(sq)
    elif p == 1.0:
        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)  # (..., P, R, M)
        return lucid.abs(diff).sum(dim=-1)
    elif p == float("inf"):
        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
        return lucid.abs(diff).max(dim=-1)
    elif p == 0.0:
        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
        # count non-zero: use abs > 0 comparison via isfinite trick
        abs_diff = lucid.abs(diff)
        # 1.0 where nonzero else 0.0  — engine has no lt/gt yet; use where+isnan workaround
        nonzero = lucid.where(
            lucid.isnan(abs_diff) | (abs_diff == lucid.zeros_like(abs_diff)),
            lucid.zeros_like(abs_diff),
            lucid.ones_like(abs_diff),
        )
        return nonzero.sum(dim=-1)
    else:
        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)  # (..., P, R, M)
        abs_diff = lucid.abs(diff)
        return (abs_diff**p).sum(dim=-1) ** (1.0 / p)


# ── bincount ──────────────────────────────────────────────────────────────────


def bincount(
    input: Tensor,
    weights: Tensor | None = None,
    minlength: int = 0,
) -> Tensor:
    """Count occurrences of each integer value in ``input``.

    ``input`` must be a 1-D non-negative integer tensor.
    Returns a 1-D tensor of length ``max(input.max() + 1, minlength)``.
    """
    flat = input.reshape(-1)
    n = int(flat.shape[0])
    vals = [int(flat[i].item()) for i in range(n)]
    if vals and min(vals) < 0:
        raise ValueError("bincount: input must contain non-negative integers")
    length = max((max(vals) + 1 if vals else 0), minlength)
    if weights is not None:
        wflat = weights.reshape(-1)
        result: list = [0.0] * length
        for i, v in enumerate(vals):
            result[v] += float(wflat[i].item())
        return lucid.tensor(result, dtype=lucid.float64)
    else:
        counts: list = [0] * length
        for v in vals:
            counts[v] += 1
        return lucid.tensor(counts, dtype=lucid.int64)


def multinomial(
    input: Tensor,
    num_samples: int,
    replacement: bool = False,
    *,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    """Draw ``num_samples`` indices from the categorical distribution defined by ``input``.

    ``input`` can be 1-D (single distribution) or 2-D (batch of distributions).
    Returns an integer tensor of shape ``(num_samples,)`` or
    ``(batch_size, num_samples)`` respectively.

    This op is not differentiable.
    """

    def _sample_row(probs_list: list[float], k: int, replace: bool) -> list[int]:
        total = sum(probs_list)
        probs_list = [p / total for p in probs_list]
        population = list(range(len(probs_list)))
        if replace:
            return random.choices(population, weights=probs_list, k=k)
        # Without replacement: repeated weighted choice, removing chosen item.
        chosen: list = []
        remaining = list(range(len(probs_list)))
        w = list(probs_list)
        for _ in range(k):
            if not remaining:
                raise ValueError(
                    "multinomial: not enough elements for sampling without replacement"
                )
            (sel,) = random.choices(remaining, weights=w, k=1)
            chosen.append(sel)
            pos = remaining.index(sel)
            remaining.pop(pos)
            w.pop(pos)
        return chosen

    if input.ndim == 1:
        n = int(input.shape[0])
        probs = [float(input[i].item()) for i in range(n)]
        idx = _sample_row(probs, num_samples, replacement)
        return lucid.tensor(idx, dtype=lucid.int64)
    else:
        batch = int(input.shape[0])
        n_cat = int(input.shape[1])
        rows = []
        for b in range(batch):
            probs = [float(input[b, i].item()) for i in range(n_cat)]
            rows.append(_sample_row(probs, num_samples, replacement))
        return lucid.tensor(rows, dtype=lucid.int64)


def poisson(
    input: Tensor,
    *,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    """Sample element-wise from ``Poisson(rate=input[i])``.

    Returns an int64 tensor of the same shape as ``input``.  Negative
    rates raise ``ValueError``; zero rates always return zero.

    Algorithm:
      * Knuth's multiplication method for ``rate < 30`` (exact, but
        runtime grows with the rate).
      * Normal approximation ``Pois(λ) ≈ ⌊N(λ, √λ) + 0.5⌋`` for
        ``rate ≥ 30`` — bias < 0.05 standard deviations and orders of
        magnitude faster than Knuth in the tail.

    Uses Lucid's Philox PRNG so :func:`manual_seed` controls the stream;
    pass an explicit ``generator`` for an isolated sequence.
    """
    from lucid._factories.random import _active_default_gen

    g: _C_engine.Generator = (
        generator if generator is not None else _active_default_gen()
    )

    flat: Tensor = input.reshape(-1)
    n: int = int(flat.numel())
    out: list[int] = []
    _SMALL_RATE_CUTOFF: float = 30.0
    _TWO_PI: float = 2.0 * math.pi

    for i in range(n):
        r: float = float(flat[i].item())
        if r < 0.0:
            raise ValueError(
                f"poisson: rate must be non-negative, got {r} at flat index {i}"
            )
        if r == 0.0:
            out.append(0)
            continue

        if r < _SMALL_RATE_CUTOFF:
            # Knuth: keep multiplying U(0,1) draws until product ≤ e^{-rate}.
            L: float = math.exp(-r)
            k: int = 0
            p: float = 1.0
            while True:
                k += 1
                u: float = g.next_uniform_float()
                # Guard against u==0 (Philox 24-bit mantissa never returns
                # exactly 0, but be defensive against future RNG changes).
                if u <= 0.0:
                    u = 1e-300
                p *= u
                if p <= L:
                    break
            out.append(k - 1)
        else:
            # Normal approximation: Box-Muller for one std-normal, scale
            # by √rate, shift by rate, round to nearest non-negative int.
            u1: float = g.next_uniform_float()
            u2: float = g.next_uniform_float()
            if u1 <= 0.0:
                u1 = 1e-30
            z: float = math.sqrt(-2.0 * math.log(u1)) * math.cos(_TWO_PI * u2)
            s: int = int(math.floor(r + math.sqrt(r) * z + 0.5))
            out.append(max(0, s))

    return lucid.tensor(out, dtype=lucid.int64).reshape(input.shape)


def histogram2d(
    x: Tensor,
    y: Tensor,
    bins: int | tuple[int, int] = 10,
    range: tuple[tuple[float, float], tuple[float, float]] | None = None,
    density: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Joint 2-D histogram of paired observations ``(x[i], y[i])``.

    Returns ``(counts, x_edges, y_edges)``: a ``(bins_x, bins_y)`` count
    matrix plus per-axis edge tensors of length ``bins_x + 1`` and
    ``bins_y + 1``.

    Mirrors the reference framework's contract: ``range`` is a pair of
    ``(lo, hi)`` tuples.  Unset ranges default to per-axis min/max.
    """
    from lucid._C import engine as _C_engine
    from lucid._dispatch import _unwrap, _wrap

    nbins_a, nbins_b = (bins, bins) if isinstance(bins, int) else bins
    if range is None:
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        lo_a, hi_a = float(x_flat.min().item()), float(x_flat.max().item())
        lo_b, hi_b = float(y_flat.min().item()), float(y_flat.max().item())
        if lo_a == hi_a:
            hi_a = lo_a + 1.0
        if lo_b == hi_b:
            hi_b = lo_b + 1.0
    else:
        (lo_a, hi_a), (lo_b, hi_b) = range
        lo_a, hi_a = float(lo_a), float(hi_a)
        lo_b, hi_b = float(lo_b), float(hi_b)

    counts_impl, edges_impl = _C_engine.histogram2d(
        _unwrap(x),
        _unwrap(y),
        int(nbins_a),
        int(nbins_b),
        lo_a,
        hi_a,
        lo_b,
        hi_b,
        density,
    )
    # Engine packs both edge arrays into a single (a+b+2,)-length tensor;
    # split it back into the ``(bins_a + 1, bins_b + 1)`` pair the user
    # expects from a NumPy/SciPy-like API.
    counts = _wrap(counts_impl)
    edges = _wrap(edges_impl)
    x_edges = edges[: nbins_a + 1]
    y_edges = edges[nbins_a + 1 :]
    return counts, x_edges, y_edges


def histogramdd(
    input: Tensor,
    bins: int | Sequence[int] = 10,
    range: Sequence[tuple[float, float]] | None = None,
    density: bool = False,
) -> tuple[Tensor, list[Tensor]]:
    """N-dimensional histogram.

    ``input`` must have shape ``(N, D)`` — N samples in D dimensions.
    Returns ``(counts, edges)`` where ``counts`` has shape
    ``(bins_0, bins_1, …, bins_{D-1})`` and ``edges`` is a list of D
    1-D tensors holding the per-axis bin boundaries.
    """
    from lucid._C import engine as _C_engine
    from lucid._dispatch import _unwrap, _wrap

    if input.ndim != 2:
        raise ValueError(
            f"histogramdd: expected (N, D) input, got shape {tuple(input.shape)}"
        )
    D: int = int(input.shape[1])
    if isinstance(bins, int):
        bins_seq: list[int] = [int(bins)] * D
    else:
        bins_seq = [int(b) for b in bins]
        if len(bins_seq) != D:
            raise ValueError(
                f"histogramdd: bins must be an int or a length-{D} sequence, "
                f"got length {len(bins_seq)}"
            )

    # ``range`` shadows the Python builtin in this function's signature,
    # so capture the builtin once for use below.
    py_range = __builtins__["range"] if isinstance(__builtins__, dict) else __builtins__.range  # type: ignore[index, union-attr]

    if range is None:
        ranges: list[tuple[float, float]] = []
        for d in py_range(D):
            col = input[:, d]
            lo: float = float(col.min().item())
            hi: float = float(col.max().item())
            if lo == hi:
                hi = lo + 1.0
            ranges.append((lo, hi))
    else:
        ranges = [(float(lo), float(hi)) for (lo, hi) in range]
        if len(ranges) != D:
            raise ValueError(
                f"histogramdd: range must be a length-{D} sequence, "
                f"got length {len(ranges)}"
            )

    counts_impl, edges_impl = _C_engine.histogramdd(
        _unwrap(input), bins_seq, ranges, density
    )
    counts = _wrap(counts_impl)
    edges_flat = _wrap(edges_impl)
    # Split the concatenated edges into D per-axis tensors.
    edges_list: list[Tensor] = []
    offset: int = 0
    for d in py_range(D):
        n = bins_seq[d] + 1
        edges_list.append(edges_flat[offset : offset + n])
        offset += n
    return counts, edges_list


def histogram(
    input: Tensor,
    bins: int | Sequence[float] = 10,
    range: tuple[float, float] | None = None,
    density: bool = False,
    weight: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute a histogram of ``input`` values.

    Returns ``(hist, bin_edges)`` — a 1-D count/density tensor and a
    1-D tensor of ``bins + 1`` edges.
    """
    # ``range`` is a function parameter that shadows the Python
    # builtin — capture the builtin once for use below.
    py_range = __builtins__["range"] if isinstance(__builtins__, dict) else __builtins__.range  # type: ignore[index, union-attr]

    flat = input.reshape(-1)
    n = int(flat.shape[0])
    vals = [float(flat[i].item()) for i in py_range(n)]

    if range is None:
        lo = min(vals) if vals else 0.0
        hi = max(vals) if vals else 1.0
        if lo == hi:
            hi = lo + 1.0
    else:
        lo, hi = float(range[0]), float(range[1])

    if isinstance(bins, int):
        n_bins = bins
        edges = [lo + (hi - lo) * i / n_bins for i in py_range(n_bins + 1)]
    else:
        edges = [float(b) for b in bins]
        n_bins = len(edges) - 1
        lo, hi = edges[0], edges[-1]

    w_list = None
    if weight is not None:
        wflat = weight.reshape(-1)
        w_list = [float(wflat[i].item()) for i in py_range(int(wflat.shape[0]))]

    if density:
        counts: list = [0.0] * n_bins
    else:
        counts = [0] * n_bins

    for i, v in enumerate(vals):
        if v < lo or v > hi:
            continue
        bin_idx = min(int((v - lo) / (hi - lo) * n_bins), n_bins - 1)
        w = w_list[i] if w_list is not None else 1
        counts[bin_idx] += w

    if density:
        total = sum(counts)
        if total > 0:
            counts = [
                c / (total * (edges[j + 1] - edges[j])) for j, c in enumerate(counts)
            ]
        hist_t = lucid.tensor(counts, dtype=lucid.float64)
    else:
        hist_t = lucid.tensor(counts, dtype=lucid.int64)

    edges_t = lucid.tensor(edges, dtype=lucid.float64)
    return hist_t, edges_t


__all__ = [
    "quantile",
    "nanquantile",
    "cov",
    "corrcoef",
    "cdist",
    "bincount",
    "histogram",
    "histogram2d",
    "histogramdd",
    "multinomial",
    "poisson",
]
