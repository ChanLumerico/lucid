"""Observers — collect activation / weight statistics to derive quant params.

An *observer* watches tensors flow through a model during a calibration
pass and accumulates the statistics (running min/max, or a histogram)
needed to choose ``(scale, zero_point)``.  Each observer is an
:class:`~lucid.nn.Module`, so its statistics live in **buffers** that
persist through ``state_dict`` and move with ``.to(device)``; its
``forward`` observes the input and returns it unchanged (identity), which
lets it be dropped into a model or wrapped by a :class:`FakeQuantize`.

Buffers are updated by **re-registering** them (``register_buffer`` with
the same name) rather than plain assignment: Lucid's
``Module.__setattr__`` removes a name from ``_buffers`` on reassignment,
so re-registration is the correct way to update a running statistic
(and it transparently handles the scalar→per-channel shape change on the
first observation).

All statistics are computed with the public ``lucid.*`` op surface — no
external libraries (H4); the :class:`HistogramObserver` uses the engine
``histc`` primitive rather than a numpy histogram.
"""

import functools
from typing import TYPE_CHECKING, cast, override

import lucid
import lucid.nn as nn
from lucid.quantization._qparams import calculate_qparams
from lucid.quantization._qscheme import (
    QDtype,
    QScheme,
    per_channel_symmetric,
    per_tensor_affine,
    qint8,
    quint8,
)

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


class ObserverBase(nn.Module):
    """Base class for statistics collectors that yield ``(scale, zero_point)``.

    Parameters
    ----------
    qscheme : QScheme
        Target quantization scheme.
    qdtype : QDtype
        Target quantized dtype.
    ch_axis : int, optional
        Channel axis; forced to ``None`` for per-tensor schemes.
    eps : float, default 1e-8
        Lower floor on ``scale`` passed to :func:`calculate_qparams`.
    """

    def __init__(
        self,
        qscheme: QScheme,
        qdtype: QDtype,
        ch_axis: int | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.qscheme = qscheme
        self.qdtype = qdtype
        self.ch_axis = ch_axis if qscheme.is_per_channel else None
        self.eps = eps

    def calculate_qparams(self) -> tuple[Tensor, Tensor]:
        """Return ``(scale, zero_point)`` from the accumulated statistics."""
        raise NotImplementedError

    @classmethod
    def with_args(cls, **kwargs: object) -> functools.partial[ObserverBase]:
        """Return a zero-arg factory that builds this observer with ``kwargs``.

        Mirrors the reference framework's ``Observer.with_args`` ergonomic so
        a :class:`~lucid.quantization.QConfig` can carry an observer *recipe*
        rather than a live instance.
        """
        return functools.partial(cls, **kwargs)  # type: ignore[arg-type]


class MinMaxObserver(ObserverBase):
    """Per-tensor observer tracking the global running min / max."""

    min_val: Tensor
    max_val: Tensor

    def __init__(
        self,
        qscheme: QScheme = per_tensor_affine,
        qdtype: QDtype = quint8,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(qscheme, qdtype, None, eps)
        self.register_buffer("min_val", lucid.tensor(float("inf")))
        self.register_buffer("max_val", lucid.tensor(float("-inf")))

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # observer forward is unary
        """Fold ``x``'s global min/max into the running statistics."""
        self.register_buffer("min_val", lucid.minimum(self.min_val, x.min()))
        self.register_buffer("max_val", lucid.maximum(self.max_val, x.max()))
        return x

    @override
    def calculate_qparams(self) -> tuple[Tensor, Tensor]:
        return calculate_qparams(
            self.min_val, self.max_val, self.qscheme, self.qdtype, self.eps
        )


class MovingAverageMinMaxObserver(MinMaxObserver):
    """Per-tensor observer with an exponential moving average of min / max.

    Smooths the running range across calibration batches, which is more
    robust to per-batch outliers than a hard running min/max.
    """

    def __init__(
        self,
        qscheme: QScheme = per_tensor_affine,
        qdtype: QDtype = quint8,
        averaging_constant: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(qscheme, qdtype, eps)
        self.averaging_constant = averaging_constant

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # observer forward is unary
        """EMA-update the running min/max (seeded on the first batch)."""
        c = self.averaging_constant
        cur_min, cur_max = x.min(), x.max()
        # On the first batch (min == +inf) seed directly; else EMA toward cur.
        first = lucid.isinf(self.min_val)
        new_min = lucid.where(
            first, cur_min, self.min_val + c * (cur_min - self.min_val)
        )
        new_max = lucid.where(
            first, cur_max, self.max_val + c * (cur_max - self.max_val)
        )
        self.register_buffer("min_val", new_min)
        self.register_buffer("max_val", new_max)
        return x


class PerChannelMinMaxObserver(ObserverBase):
    """Per-channel observer tracking running min / max along ``ch_axis``."""

    min_val: Tensor
    max_val: Tensor

    def __init__(
        self,
        ch_axis: int = 0,
        qscheme: QScheme = per_channel_symmetric,
        qdtype: QDtype = qint8,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(qscheme, qdtype, ch_axis, eps)
        self.register_buffer("min_val", lucid.tensor(float("inf")))
        self.register_buffer("max_val", lucid.tensor(float("-inf")))

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # observer forward is unary
        """Fold per-channel min/max (reduced over all non-channel axes)."""
        axis = self.ch_axis if self.ch_axis is not None else 0
        perm = lucid.moveaxis(x, axis, 0)
        num_channels = perm.shape[0]
        flat = perm.reshape(num_channels, -1)
        cur_min = lucid.amin(flat, dim=1)
        cur_max = lucid.amax(flat, dim=1)
        # A scalar +inf/-inf seed broadcasts to per-channel on the first call.
        self.register_buffer("min_val", lucid.minimum(self.min_val, cur_min))
        self.register_buffer("max_val", lucid.maximum(self.max_val, cur_max))
        return x

    @override
    def calculate_qparams(self) -> tuple[Tensor, Tensor]:
        return calculate_qparams(
            self.min_val, self.max_val, self.qscheme, self.qdtype, self.eps
        )


class HistogramObserver(ObserverBase):
    """Per-tensor observer that records a value histogram and picks a clip range.

    Accumulates a ``bins``-bucket histogram (via the engine ``histc``
    primitive) over the running ``[min, max]``, rebinning when the range
    grows.  ``calculate_qparams`` then selects the ``[new_min, new_max]``
    clip that minimises the expected L2 quantization error estimated from
    the histogram — trading a little clipping for much finer resolution on
    heavy-tailed activations.
    """

    min_val: Tensor
    max_val: Tensor
    histogram: Tensor

    def __init__(
        self,
        bins: int = 2048,
        qscheme: QScheme = per_tensor_affine,
        qdtype: QDtype = quint8,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(qscheme, qdtype, None, eps)
        self.bins = bins
        self.register_buffer("min_val", lucid.tensor(float("inf")))
        self.register_buffer("max_val", lucid.tensor(float("-inf")))
        self.register_buffer("histogram", lucid.zeros(bins))

    def _rebin(
        self,
        hist: Tensor,
        old_min: float,
        old_max: float,
        new_min: float,
        new_max: float,
    ) -> Tensor:
        """Redistribute an existing histogram onto a wider ``[new_min,new_max]``."""
        bins = self.bins
        old_width = (old_max - old_min) / bins
        new_width = (new_max - new_min) / bins
        centers = old_min + (lucid.arange(bins).to(lucid.float32) + 0.5) * old_width
        # Clip in the float domain (CPU has no integer clamp kernel) then cast.
        raw = lucid.clip((centers - new_min) / new_width, 0.0, float(bins - 1))
        dst = raw.to(lucid.int64)
        out = lucid.zeros(bins)
        return lucid.scatter_add(out, 0, dst, hist)

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # observer forward is unary
        """Update the running range and accumulate the histogram."""
        cur_min, cur_max = float(x.min().item()), float(x.max().item())
        old_min, old_max = float(self.min_val.item()), float(self.max_val.item())
        if old_min == float("inf"):
            new_min, new_max = cur_min, cur_max
            hist = lucid.histc(x, bins=self.bins, min=new_min, max=new_max)
        else:
            new_min, new_max = min(old_min, cur_min), max(old_max, cur_max)
            hist = lucid.histc(x, bins=self.bins, min=new_min, max=new_max)
            if new_min == old_min and new_max == old_max:
                hist = hist + self.histogram
            else:
                hist = hist + self._rebin(
                    self.histogram, old_min, old_max, new_min, new_max
                )
        self.register_buffer("min_val", lucid.tensor(float(new_min)))
        self.register_buffer("max_val", lucid.tensor(float(new_max)))
        self.register_buffer("histogram", hist)
        return x

    def _search_clip_range(self) -> tuple[float, float]:
        """Scan candidate clip ranges; return the L2-error-minimising one."""
        lo, hi = float(self.min_val.item()), float(self.max_val.item())
        # ``histogram`` is a 1-D float buffer, so ``tolist`` yields ``list[float]``.
        counts: list[float] = cast(list[float], self.histogram.tolist())
        total = sum(counts)
        if total <= 0.0 or hi <= lo:
            return lo, hi
        bins = self.bins
        width = (hi - lo) / bins
        centers = [lo + (i + 0.5) * width for i in range(bins)]
        levels = self.qdtype.quant_max - self.qdtype.quant_min
        best_err, best = float("inf"), (lo, hi)
        # Coarse symmetric-ish scan: clip the tails at a handful of quantiles.
        for frac in (0.0, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2):
            cmin = _quantile_from_hist(counts, centers, frac)
            cmax = _quantile_from_hist(counts, centers, 1.0 - frac)
            if cmax <= cmin:
                continue
            step = (cmax - cmin) / max(levels, 1)
            err = 0.0
            for c, n in zip(centers, counts):
                if n == 0.0:
                    continue
                cc = cmin if c < cmin else (cmax if c > cmax else c)
                # clip error + rounding error (uniform-quantization variance).
                err += n * ((c - cc) ** 2 + (step * step) / 12.0)
            if err < best_err:
                best_err, best = err, (cmin, cmax)
        return best

    @override
    def calculate_qparams(self) -> tuple[Tensor, Tensor]:
        new_min, new_max = self._search_clip_range()
        return calculate_qparams(
            lucid.tensor(float(new_min)),
            lucid.tensor(float(new_max)),
            self.qscheme,
            self.qdtype,
            self.eps,
        )


def _quantile_from_hist(counts: list[float], centers: list[float], q: float) -> float:
    """Return the bin center at cumulative mass fraction ``q``."""
    total = sum(counts)
    if total <= 0:
        return centers[0]
    target = q * total
    acc = 0.0
    for c, n in zip(centers, counts):
        acc += n
        if acc >= target:
            return c
    return centers[-1]


class MovingAveragePerChannelMinMaxObserver(PerChannelMinMaxObserver):
    """Per-channel observer with an EMA of the per-channel min / max.

    The per-channel analogue of :class:`MovingAverageMinMaxObserver` — smooths
    each channel's range across calibration batches (robust to per-batch
    outliers), which the reference framework ships as a distinct observer.
    """

    def __init__(
        self,
        ch_axis: int = 0,
        qscheme: QScheme = per_channel_symmetric,
        qdtype: QDtype = qint8,
        averaging_constant: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(ch_axis, qscheme, qdtype, eps)
        self.averaging_constant = averaging_constant

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # observer forward is unary
        """EMA-update the per-channel min/max (seeded on the first batch)."""
        c = self.averaging_constant
        axis = self.ch_axis if self.ch_axis is not None else 0
        perm = lucid.moveaxis(x, axis, 0)
        flat = perm.reshape(perm.shape[0], -1)
        cur_min, cur_max = lucid.amin(flat, dim=1), lucid.amax(flat, dim=1)
        # First batch: the ±inf seed is scalar, so seed the per-channel vectors
        # directly (a scalar `where` condition wouldn't broadcast to length C).
        if bool(lucid.isinf(self.min_val).all().item()):
            new_min, new_max = cur_min, cur_max
        else:
            new_min = self.min_val + c * (cur_min - self.min_val)
            new_max = self.max_val + c * (cur_max - self.max_val)
        self.register_buffer("min_val", new_min)
        self.register_buffer("max_val", new_max)
        return x


class FixedQParamsObserver(ObserverBase):
    """Observer with a fixed ``(scale, zero_point)`` — no statistics collected.

    For ops whose output range is known a priori (``sigmoid`` → ``[0, 1]``,
    ``tanh`` → ``[-1, 1]``, ``softmax`` → ``[0, 1]``): the grid is fixed, so
    calibration would be wasted.  ``forward`` is the identity.
    """

    scale: Tensor
    zero_point: Tensor

    def __init__(
        self,
        scale: float,
        zero_point: int,
        qscheme: QScheme = per_tensor_affine,
        qdtype: QDtype = quint8,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(qscheme, qdtype, None, eps)
        self.register_buffer("scale", lucid.tensor(float(scale)))
        self.register_buffer("zero_point", lucid.tensor(float(zero_point)))

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # observer forward is unary
        """Identity — the qparams are fixed, so no statistics are gathered."""
        return x

    @override
    def calculate_qparams(self) -> tuple[Tensor, Tensor]:
        return self.scale, self.zero_point


class PlaceholderObserver(ObserverBase):
    """Carries ``qdtype`` metadata but collects no statistics.

    Used where activation qparams are produced at *runtime* (dynamic quant) or
    deliberately absent — the observer only records the target dtype/scheme.
    """

    def __init__(
        self,
        qscheme: QScheme = per_tensor_affine,
        qdtype: QDtype = quint8,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(qscheme, qdtype, None, eps)

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # observer forward is unary
        """Identity — no statistics are gathered."""
        return x

    @override
    def calculate_qparams(self) -> tuple[Tensor, Tensor]:
        raise RuntimeError(
            f"{type(self).__name__} collects no statistics; qparams are computed "
            "at runtime (dynamic quantization) or supplied elsewhere."
        )


class NoopObserver(PlaceholderObserver):
    """A :class:`PlaceholderObserver` by another name — pure identity, no stats."""

