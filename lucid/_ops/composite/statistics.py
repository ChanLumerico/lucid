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
            lo_t = lucid.tensor([lo], dtype=lucid.int64, device=sorted_x.device)
            hi_t = lucid.tensor([hi], dtype=lucid.int64, device=sorted_x.device)
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
                    dim,
                    lucid.tensor(
                        [round(idx_f)], dtype=lucid.int64, device=sorted_x.device
                    ),
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

    Parameters
    ----------
    input : Tensor
        Real-valued input.
    q : float, sequence of float, or Tensor
        Quantile(s) in :math:`[0, 1]`.  A single ``q`` returns a tensor
        without the leading quantile axis; a sequence/tensor adds a
        leading axis of length ``len(q)``.
    dim : int, optional
        Axis along which to reduce.  ``None`` (default) reduces over
        the flattened input.
    keepdim : bool, optional
        Retain the reduced axis as a size-1 dimension.  Default
        ``False``.
    interpolation : str, optional
        Rule for interpolating between the two samples bracketing the
        quantile.  One of ``'linear'`` (default), ``'lower'``,
        ``'higher'``, ``'midpoint'``, ``'nearest'``.

    Returns
    -------
    Tensor
        Same dtype as ``input``.  Shape depends on ``q`` / ``dim`` /
        ``keepdim``.

    See Also
    --------
    nanquantile : NaN-aware sibling.
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
    """NaN-aware quantile — same interface as :func:`quantile`, but
    every ``NaN`` is excluded from the sort and the per-axis quantile
    is computed over the remaining values.

    Parameters
    ----------
    input : Tensor
        Real-valued input.  ``NaN`` entries are masked out before
        sorting.
    q : float, sequence of float, or Tensor
        Quantile(s) in :math:`[0, 1]`.  A single ``q`` returns a tensor
        without the leading quantile axis; a sequence/tensor adds a
        leading axis of length ``len(q)``.
    dim : int, optional
        Axis along which to reduce.  ``None`` (default) reduces over
        the flattened input.
    keepdim : bool, optional
        Retain the reduced axis as a size-1 dimension.  Default
        ``False``.
    interpolation : str, optional
        Interpolation rule between two surrounding samples.  Currently
        ``"linear"`` is supported (default).

    Returns
    -------
    Tensor
        Same dtype as ``input``; shape depends on ``q`` / ``dim`` /
        ``keepdim``.  Returns ``NaN`` for slices that contain only
        ``NaN`` values.

    See Also
    --------
    quantile : NaN-unaware version (a single ``NaN`` poisons the slice).
    """
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
            nan_val = lucid.tensor(math.nan, dtype=input.dtype, device=input.device)
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
            nan_val = lucid.full(
                out_shape, math.nan, dtype=input.dtype, device=input.device
            )
            if scalar_q:
                return nan_val.squeeze(dim)
            return nan_val
        results = _quantile_sorted(sorted_x, q_list, dim, keepdim, interpolation, n)

    if scalar_q:
        return results[0]
    return lucid.stack(results)


# ── covariance & correlation ──────────────────────────────────────────────────


def cov(input: Tensor, correction: int = 1) -> Tensor:
    r"""Covariance matrix of ``input``.

    Parameters
    ----------
    input : Tensor
        Shape ``(N,)`` (single-variable degenerate case) or ``(N, M)``
        — *N* variables observed at *M* time points.
    correction : int, optional
        Bessel correction; the divisor is ``max(1, M - correction)``.
        Default ``1`` (unbiased sample covariance); pass ``0`` for the
        biased population covariance.

    Returns
    -------
    Tensor
        Shape ``(N, N)``.  Entry :math:`(i, j)` =
        :math:`\frac{1}{M - \text{correction}}\sum_k (x_{i,k} - \bar x_i)(x_{j,k} - \bar x_j)`.

    See Also
    --------
    corrcoef : normalised counterpart (Pearson correlation).
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
    r"""Pearson correlation-coefficient matrix.

    Composite of :func:`cov` divided by the outer product of the
    per-variable standard deviations:

    .. math::

        R_{ij} = \frac{C_{ij}}{\sqrt{C_{ii}} \sqrt{C_{jj}}}.

    Diagonal entries are ``1.0`` (up to floating-point); off-diagonal
    entries fall in :math:`[-1, 1]`.

    Parameters
    ----------
    input : Tensor
        Shape ``(N,)`` for a single-variable degenerate case, or
        ``(N, M)`` for ``N`` variables observed at ``M`` time points.

    Returns
    -------
    Tensor
        Shape ``(N, N)`` Pearson correlation matrix.

    See Also
    --------
    cov : raw covariance matrix without the std-normalisation.
    """
    c = cov(input, correction=1)  # (N, N)
    d = lucid.sqrt(lucid.diagonal(c))  # (N,) — std deviations
    # Outer product: d_col * d_row gives the normalisation matrix
    d_col = d.reshape(-1, 1)  # (N, 1)
    d_row = d.reshape(1, -1)  # (1, N)
    return c / (d_col * d_row)


# ── pairwise distances ────────────────────────────────────────────────────────


def cdist(x1: Tensor, x2: Tensor, p: float = 2.0) -> Tensor:
    r"""Pairwise :math:`L_p` distance matrix between rows of ``x1`` and ``x2``.

    Specialised paths exist for ``p`` in :math:`\{1, 2, \infty, 0\}`;
    other ``p`` values fall back to the general
    :math:`(\sum_k |a_k - b_k|^p)^{1/p}` formula.  For ``p=2`` the
    numerically stable expansion
    :math:`\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2 a \cdot b^\top` is used
    so large coordinates don't lose precision.

    Parameters
    ----------
    x1 : Tensor
        Shape ``(..., P, M)``.
    x2 : Tensor
        Shape ``(..., R, M)``.  Leading batch dimensions are
        broadcast-compatible with ``x1``.
    p : float, optional
        :math:`L_p` exponent.  Default ``2.0`` (Euclidean).  Use
        ``float("inf")`` for Chebyshev, ``0.0`` for the Hamming-style
        non-zero count.

    Returns
    -------
    Tensor
        Shape ``(..., P, R)``.  Entry :math:`(\ldots, i, j)` =
        :math:`\|x_1[\ldots, i, :] - x_2[\ldots, j, :]\|_p`.
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

    Parameters
    ----------
    input : Tensor
        1-D non-negative integer tensor.  Negative values raise.
    weights : Tensor, optional
        Same length as ``input``.  When supplied, ``output[i]`` is the
        *sum of weights* for entries with value ``i`` (rather than the
        plain count).  Default ``None``.
    minlength : int, optional
        Floor for the output length.  The output has length
        ``max(input.max() + 1, minlength)``.  Default ``0``.

    Returns
    -------
    Tensor
        1-D tensor; dtype is float when ``weights`` is given, otherwise
        int64.

    Raises
    ------
    ValueError
        If ``input`` contains negative values.
    """
    flat = input.reshape(-1)
    n = int(flat.shape[0])
    vals = [int(flat[i].item()) for i in range(n)]
    if vals and min(vals) < 0:
        raise ValueError("bincount: input must contain non-negative integers")
    length = max((max(vals) + 1 if vals else 0), minlength)
    if weights is not None:
        wflat = weights.reshape(-1)
        result: list[object] = [0.0] * length
        for i, v in enumerate(vals):
            result[v] += float(wflat[i].item())
        return lucid.tensor(result, dtype=lucid.float64)
    else:
        counts: list[object] = [0] * length
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

    Non-differentiable — gradients of ``input`` are not propagated
    through the sampling step.  Pass an explicit ``generator`` for an
    isolated PRNG stream; without one, draws come from Lucid's
    :func:`manual_seed`-controlled default Philox generator.

    Parameters
    ----------
    input : Tensor
        Probability weights.  1-D ``(K,)`` for a single categorical
        distribution, or 2-D ``(B, K)`` for a batch of ``B``
        distributions.  Weights are renormalised internally; they need
        not sum to 1.
    num_samples : int
        Number of draws per distribution.
    replacement : bool, optional
        When ``True`` each draw is independent.  When ``False``
        (default) each chosen index is removed before the next draw
        within the same row.
    generator : engine.Generator, optional
        Override Lucid's default Philox generator with an isolated
        stream.

    Returns
    -------
    Tensor
        Integer tensor.  Shape ``(num_samples,)`` for 1-D input,
        ``(B, num_samples)`` for 2-D input.
    """

    def _sample_row(probs_list: list[float], k: int, replace: bool) -> list[int]:
        """Draw ``k`` indices from a single categorical row with weights ``probs_list``.

        Weights are renormalised internally. When ``replace`` is ``False``,
        each chosen index is removed before the next draw.
        """
        total = sum(probs_list)
        probs_list = [p / total for p in probs_list]
        population = list(range(len(probs_list)))
        if replace:
            return random.choices(population, weights=probs_list, k=k)
        # Without replacement: repeated weighted choice, removing chosen item.
        chosen: list[int] = []
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
    r"""Sample element-wise from :math:`\mathrm{Poisson}(\text{rate} = \text{input}[i])`.

    Parameters
    ----------
    input : Tensor
        Non-negative rates :math:`\lambda_i` per output position.  Any
        shape.
    generator : engine.Generator, optional
        Override Lucid's default Philox generator with an isolated
        stream.  When ``None`` (default), draws are controlled by
        :func:`manual_seed`.

    Returns
    -------
    Tensor
        ``int64`` tensor with the same shape as ``input``.  Zero rates
        always return zero.

    Raises
    ------
    ValueError
        If any element of ``input`` is negative.

    Notes
    -----
    Two-branch implementation:

    * **Knuth's multiplication method** for ``rate < 30`` — exact but
      runtime grows with the rate.
    * **Normal approximation** :math:`\mathrm{Pois}(\lambda) \approx
      \lfloor \mathcal{N}(\lambda, \sqrt{\lambda}) + 0.5 \rfloor`
      for ``rate >= 30`` — bias < 0.05 standard deviations, orders of
      magnitude faster than Knuth in the tail.
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

    Parameters
    ----------
    x : Tensor
        1-D x-coordinates.
    y : Tensor
        1-D y-coordinates; must have the same length as ``x``.
    bins : int or tuple[int, int], optional
        Bin count.  A single int applies to both axes; a 2-tuple sets
        each axis independently.  Default ``10`` (i.e. 10×10).
    range : tuple of (float, float), optional
        ``((x_lo, x_hi), (y_lo, y_hi))``.  When ``None`` (default)
        each axis uses its own ``(min, max)``.
    density : bool, optional
        When ``True`` divide by total count × bin area to yield a
        probability density (sums × bin-area = 1).  Default ``False``.

    Returns
    -------
    tuple of Tensor
        ``(counts, x_edges, y_edges)``.  ``counts`` shape
        ``(bins_x, bins_y)`` int64 (or float64 when ``density``).
        ``x_edges`` length ``bins_x + 1``, ``y_edges`` length
        ``bins_y + 1``.
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
    """N-dimensional histogram for arbitrary sample dimension.

    Parameters
    ----------
    input : Tensor
        Shape ``(N, D)`` — :math:`N` samples in :math:`D`
        dimensions.  Other ranks raise.
    bins : int or sequence of int, optional
        Bin count per axis.  A single int applies to every axis; a
        sequence of length :math:`D` sets each axis independently.
        Default ``10``.
    range : sequence of (float, float), optional
        Per-axis ``(lo, hi)`` ranges.  When ``None`` (default) each
        axis uses its own ``(min, max)`` of the samples.
    density : bool, optional
        When ``True`` divide by total count × bin volume to yield a
        density that integrates to 1.  Default ``False``.

    Returns
    -------
    tuple
        ``(counts, edges)``.  ``counts`` shape
        ``(bins_0, …, bins_{D-1})``; ``edges`` is a length-:math:`D`
        list of 1-D tensors of size ``bins_i + 1``.
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
    py_range = (
        __builtins__["range"] if isinstance(__builtins__, dict) else __builtins__.range
    )

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
    """Compute a 1-D histogram of ``input`` values.

    Parameters
    ----------
    input : Tensor
        Values to bin.  Flattened internally — shape doesn't matter.
    bins : int or sequence of float, optional
        When an int, the number of equal-width bins (default ``10``).
        When a sequence, the explicit bin edges (length ``bins_n + 1``).
    range : tuple[float, float], optional
        ``(lo, hi)``.  Values outside the range are dropped.  Ignored
        when ``bins`` is a sequence of explicit edges.  ``None``
        (default) → ``(input.min(), input.max())``.
    density : bool, optional
        When ``True`` divide by total count × bin width to yield a
        probability density that integrates to 1.  Default ``False``.
    weight : Tensor, optional
        Same length as ``input``.  When supplied, sums weights per bin
        rather than counting.  Default ``None``.

    Returns
    -------
    tuple of Tensor
        ``(hist, bin_edges)``.  ``hist`` is a 1-D count/density tensor;
        ``bin_edges`` is 1-D of length ``bins + 1``.
    """
    # ``range`` is a function parameter that shadows the Python
    # builtin — capture the builtin once for use below.
    py_range = (
        __builtins__["range"] if isinstance(__builtins__, dict) else __builtins__.range
    )

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
        counts: list[float] = [0.0] * n_bins
    else:
        counts = [0.0] * n_bins

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


def std_mean(
    x: Tensor,
    dim: int | Sequence[int] | None = None,
    correction: int = 1,
    keepdim: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return ``(std, mean)`` along ``dim`` in a single call.

    Fused convenience over :func:`lucid.std` + :func:`lucid.mean` — the
    engine reduction still scans ``x`` twice, but the call site is
    shorter and the two outputs share an explicit ``correction`` /
    ``keepdim`` contract.

    Parameters
    ----------
    x : Tensor
        Input.
    dim : int, sequence of int, or None, optional
        Axis or axes to reduce.  ``None`` (default) reduces over every
        axis to a scalar pair.
    correction : int, optional
        Bessel correction passed to :func:`lucid.std`.  ``1`` is the
        unbiased sample std (default); ``0`` gives the biased
        population std.
    keepdim : bool, optional
        Retain the reduced axes as size-1 dimensions.  Default
        ``False``.

    Returns
    -------
    tuple of Tensor
        ``(std, mean)`` — same shape relationship as
        :func:`lucid.std` / :func:`lucid.mean` would produce
        individually.

    See Also
    --------
    var_mean : variance + mean variant.
    """
    if dim is not None:
        _dim = list(dim) if not isinstance(dim, int) else dim
        m = lucid.mean(x, _dim, keepdim)
        s = lucid.std(x, _dim, keepdim, correction=correction)
    else:
        m = lucid.mean(x)
        s = lucid.std(x)
    return s, m


def var_mean(
    x: Tensor,
    dim: int | Sequence[int] | None = None,
    correction: int = 1,
    keepdim: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return ``(var, mean)`` along ``dim`` in a single call.

    Variance counterpart to :func:`std_mean`.  Useful inside
    normalisation layers where the squared deviation is the quantity
    actually wanted (e.g. BatchNorm / LayerNorm forward computes
    ``(x - mean) / sqrt(var + eps)``).

    Parameters
    ----------
    x : Tensor
        Input.
    dim : int, sequence of int, or None, optional
        Axis or axes to reduce.  ``None`` (default) → full-tensor
        reduction.
    correction : int, optional
        Bessel correction passed to :func:`lucid.var`.  Default ``1``
        (unbiased sample variance).
    keepdim : bool, optional
        Retain the reduced axes as size-1 dimensions.  Default
        ``False``.

    Returns
    -------
    tuple of Tensor
        ``(var, mean)``.

    See Also
    --------
    std_mean : standard-deviation + mean variant.
    """
    if dim is not None:
        _dim = list(dim) if not isinstance(dim, int) else dim
        m = lucid.mean(x, _dim, keepdim)
        v = lucid.var(x, _dim, keepdim, correction=correction)
    else:
        m = lucid.mean(x)
        v = lucid.var(x)
    return v, m


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
    "std_mean",
    "var_mean",
]
