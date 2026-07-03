"""Quantization schemes and quantized-dtype descriptors.

A *quantization scheme* fixes how a real-valued tensor is mapped onto a
grid of integers: whether one ``(scale, zero_point)`` pair covers the
whole tensor (*per-tensor*) or one pair covers each output channel
(*per-channel*), and whether the mapping is *affine* (arbitrary
``zero_point``) or *symmetric* (``zero_point`` pinned so that zero maps
to zero).  A *quantized dtype* (:class:`QDtype`) pins the integer grid
itself — its bit width, signedness, representable ``[quant_min,
quant_max]`` range, and the on-device storage dtype that holds the
integer codes.

Lucid follows the *sidecar* representation (design B): a quantized
weight is an ordinary integer :class:`~lucid.Tensor` plus float
``scale`` / ``zero_point`` buffers held by the owning module, rather
than a dedicated quantized tensor subtype.  These descriptors are the
metadata that ties the three together.
"""

import enum
from dataclasses import dataclass
from typing import override


class QScheme(enum.Enum):
    """How ``scale`` / ``zero_point`` map real values onto the integer grid.

    Members
    -------
    PER_TENSOR_AFFINE
        One ``(scale, zero_point)`` for the whole tensor; ``zero_point``
        is free, so asymmetric ranges (e.g. post-ReLU activations) map
        without wasting codes.
    PER_TENSOR_SYMMETRIC
        One ``scale`` for the whole tensor with ``zero_point`` pinned
        (0 for signed, mid-range for unsigned); real 0 maps exactly to
        an integer code.
    PER_CHANNEL_AFFINE
        One ``(scale, zero_point)`` per channel along a chosen axis.
    PER_CHANNEL_SYMMETRIC
        One ``scale`` per channel with pinned ``zero_point`` — the usual
        choice for convolution / linear weights.
    """

    PER_TENSOR_AFFINE = "per_tensor_affine"
    PER_TENSOR_SYMMETRIC = "per_tensor_symmetric"
    PER_CHANNEL_AFFINE = "per_channel_affine"
    PER_CHANNEL_SYMMETRIC = "per_channel_symmetric"

    @property
    def is_per_channel(self) -> bool:
        """Return ``True`` for the two per-channel schemes."""
        return self in (QScheme.PER_CHANNEL_AFFINE, QScheme.PER_CHANNEL_SYMMETRIC)

    @property
    def is_symmetric(self) -> bool:
        """Return ``True`` when ``zero_point`` is pinned (symmetric schemes)."""
        return self in (QScheme.PER_TENSOR_SYMMETRIC, QScheme.PER_CHANNEL_SYMMETRIC)


# Module-level constants mirroring the conventional lowercase qscheme names,
# so call sites read `qscheme=lucid.quantization.per_channel_symmetric`.
per_tensor_affine: QScheme = QScheme.PER_TENSOR_AFFINE
per_tensor_symmetric: QScheme = QScheme.PER_TENSOR_SYMMETRIC
per_channel_affine: QScheme = QScheme.PER_CHANNEL_AFFINE
per_channel_symmetric: QScheme = QScheme.PER_CHANNEL_SYMMETRIC


@dataclass(frozen=True)
class QDtype:
    """Descriptor for a quantized integer grid.

    Unlike the reference framework, Lucid does not add quantized dtypes
    to the C++ engine enum; a :class:`QDtype` is a pure-Python descriptor
    and the integer codes live in an ordinary ``storage`` dtype.  This
    keeps the whole quantization subsystem off the engine's critical
    path until a real low-precision GEMM is wired in.

    Parameters
    ----------
    name : str
        Canonical name (``"qint8"`` …).
    bits : int
        Bit width of the grid.
    signed : bool
        Whether codes are signed.
    quant_min, quant_max : int
        Inclusive representable integer range.
    storage : str
        Name of the Lucid dtype that physically holds the codes.  ``quint8``
        maps to ``int16`` for now (the engine has no ``uint8``); ``qint4``
        maps to ``int8`` (unpacked) until 4-bit packing lands.
    """

    name: str
    bits: int
    signed: bool
    quant_min: int
    quant_max: int
    storage: str

    @override
    def __repr__(self) -> str:
        """Return ``"lucid.quantization.<name>"``."""
        return f"lucid.quantization.{self.name}"


qint8: QDtype = QDtype("qint8", 8, True, -128, 127, "int8")
quint8: QDtype = QDtype("quint8", 8, False, 0, 255, "int16")
qint32: QDtype = QDtype("qint32", 32, True, -(2**31), 2**31 - 1, "int32")
qint4: QDtype = QDtype("qint4", 4, True, -8, 7, "int8")
