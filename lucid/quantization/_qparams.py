"""Quantization parameters and their derivation from an observed range.

:class:`QParams` bundles the four things a quantized tensor needs —
``scale``, ``zero_point``, the :class:`~lucid.quantization.QScheme`, and
the :class:`~lucid.quantization.QDtype` — plus the channel axis for
per-channel schemes.  :func:`calculate_qparams` turns an observed
``[min_val, max_val]`` range into the ``(scale, zero_point)`` pair, and
is the shared math behind every observer.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import lucid
from lucid._tensor.tensor import Tensor
from lucid.quantization._qscheme import QDtype, QScheme

if TYPE_CHECKING:
    _RangeLike = Tensor | float


@dataclass(frozen=True)
class QParams:
    """Immutable bundle of the parameters that define a quantized tensor.

    Parameters
    ----------
    scale : Tensor
        Quantization step; scalar for per-tensor, length-``C`` for
        per-channel.
    zero_point : Tensor
        Integer offset (stored as an integer-valued float tensor).
    qscheme : QScheme
        The mapping family.
    qdtype : QDtype
        The target integer grid.
    ch_axis : int, optional
        Channel axis for per-channel schemes; ``None`` for per-tensor.
    """

    scale: Tensor
    zero_point: Tensor
    qscheme: QScheme
    qdtype: QDtype
    ch_axis: int | None = None


def calculate_qparams(
    min_val: _RangeLike,
    max_val: _RangeLike,
    qscheme: QScheme,
    qdtype: QDtype,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    r"""Derive ``(scale, zero_point)`` from an observed value range.

    Implements the standard affine / symmetric calibration.  The observed
    range is first extended to include zero (so real 0 is always
    representable), then:

    * **affine** — :math:`s = (\max - \min)/(q_\max - q_\min)`,
      :math:`z = \operatorname{clip}(q_\min - \operatorname{round}(\min/s),
      q_\min, q_\max)`;
    * **symmetric** — :math:`s = \max(|\min|, |\max|) / ((q_\max -
      q_\min)/2)` with ``z`` pinned to 0 (signed) or mid-range
      (unsigned).

    Works elementwise, so scalar ``min_val`` / ``max_val`` yield per-tensor
    params and length-``C`` inputs yield per-channel params.

    Parameters
    ----------
    min_val, max_val : Tensor or float
        Observed minimum / maximum (per-tensor scalars or per-channel
        vectors).
    qscheme : QScheme
        Mapping family.
    qdtype : QDtype
        Target grid supplying ``quant_min`` / ``quant_max``.
    eps : float, default 1e-8
        Lower floor on ``scale`` to avoid division by zero on a degenerate
        (constant) range.

    Returns
    -------
    (Tensor, Tensor)
        ``scale`` and ``zero_point`` tensors.
    """
    min_t: Tensor = (
        min_val if isinstance(min_val, Tensor) else lucid.tensor(float(min_val))
    )
    max_t: Tensor = (
        max_val if isinstance(max_val, Tensor) else lucid.tensor(float(max_val))
    )

    quant_min, quant_max = qdtype.quant_min, qdtype.quant_max
    # Extend the range so that zero is always representable.
    min_t = lucid.minimum(min_t, lucid.zeros_like(min_t))
    max_t = lucid.maximum(max_t, lucid.zeros_like(max_t))
    eps_t = lucid.full_like(max_t, eps)

    if qscheme.is_symmetric:
        max_abs = lucid.maximum(lucid.abs(min_t), lucid.abs(max_t))
        scale = lucid.maximum(max_abs / ((quant_max - quant_min) / 2.0), eps_t)
        if qdtype.signed:
            zero_point = lucid.zeros_like(scale)
        else:
            zero_point = lucid.full_like(scale, float((quant_max + quant_min + 1) // 2))
    else:
        scale = lucid.maximum((max_t - min_t) / (quant_max - quant_min), eps_t)
        zero_point = lucid.clip(
            quant_min - lucid.round(min_t / scale), quant_min, quant_max
        )
    return scale, zero_point
