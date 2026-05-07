"""Statistical composite ops: quantile, cov, corrcoef, cdist."""

import math
import random

import lucid

# ── quantile helpers ──────────────────────────────────────────────────────────


def _quantile_sorted(sorted_x, q_list, dim, keepdim, interpolation, n):
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


def _parse_q(q):
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


def quantile(input, q, dim=None, keepdim: bool = False, interpolation: str = "linear"):
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
    input, q, dim=None, keepdim: bool = False, interpolation: str = "linear"
):
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


def cov(input, correction: int = 1):
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


def corrcoef(input):
    """Pearson correlation matrix of ``input`` (shape ``(N,)`` or ``(N, M)``)."""
    c = cov(input, correction=1)  # (N, N)
    d = lucid.sqrt(lucid.diagonal(c))  # (N,) — std deviations
    # Outer product: d_col * d_row gives the normalisation matrix
    d_col = d.reshape(-1, 1)  # (N, 1)
    d_row = d.reshape(1, -1)  # (1, N)
    return c / (d_col * d_row)


# ── pairwise distances ────────────────────────────────────────────────────────


def cdist(x1, x2, p: float = 2.0):
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


def bincount(input, weights=None, minlength: int = 0):
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


def multinomial(input, num_samples: int, replacement: bool = False, *, generator=None):
    """Draw ``num_samples`` indices from the categorical distribution defined by ``input``.

    ``input`` can be 1-D (single distribution) or 2-D (batch of distributions).
    Returns an integer tensor of shape ``(num_samples,)`` or
    ``(batch_size, num_samples)`` respectively.

    This op is not differentiable.
    """

    def _sample_row(probs_list: list, k: int, replace: bool) -> list:
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
                raise ValueError("multinomial: not enough elements for sampling without replacement")
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


def histogram(input, bins=10, range=None, density: bool = False, weight=None):
    """Compute a histogram of ``input`` values.

    Returns ``(hist, bin_edges)`` — a 1-D count/density tensor and a
    1-D tensor of ``bins + 1`` edges.
    """
    flat = input.reshape(-1)
    n = int(flat.shape[0])
    vals = [float(flat[i].item()) for i in range(n)]

    if range is None:
        lo = min(vals) if vals else 0.0
        hi = max(vals) if vals else 1.0
        if lo == hi:
            hi = lo + 1.0
    else:
        lo, hi = float(range[0]), float(range[1])

    if isinstance(bins, int):
        n_bins = bins
        edges = [lo + (hi - lo) * i / n_bins for i in range(n_bins + 1)]
    else:
        edges = [float(b) for b in bins]
        n_bins = len(edges) - 1
        lo, hi = edges[0], edges[-1]

    w_list = None
    if weight is not None:
        wflat = weight.reshape(-1)
        w_list = [float(wflat[i].item()) for i in range(int(wflat.shape[0]))]

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
            counts = [c / (total * (edges[j + 1] - edges[j])) for j, c in enumerate(counts)]
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
    "multinomial",
]
