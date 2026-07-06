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
    r"""How ``(scale, zero_point)`` map real values onto the integer grid.

    A *quantization scheme* fixes two independent choices about the affine map
    :math:`q = \operatorname{round}(x / s) + z` that turns a real value ``x`` into an
    integer code ``q``:

    1. **Granularity** — is there one ``(scale, zero_point)`` for the *whole tensor*
       (*per-tensor*), or one pair *per channel* along a chosen axis (*per-channel*)?
       Per-channel tracks layers whose channels span very different magnitudes far more
       tightly, at the cost of a small vector of parameters instead of a scalar.
    2. **Symmetry** — is ``zero_point`` *free* (*affine* / asymmetric), so the grid can
       straddle an arbitrary ``[min, max]``, or is it *pinned* (*symmetric*) so that
       real 0 maps exactly to a fixed code and the grid is centred on zero?

    The four members are the cross-product of the two useful granularities with the two
    symmetries. Four module-level aliases in lowercase mirror the members so call sites
    read naturally, e.g. ``qscheme=lucid.quantization.per_channel_symmetric``.

    Members
    -------
    PER_TENSOR_AFFINE
        One ``(scale, zero_point)`` for the whole tensor with a **free** ``zero_point``.
        The asymmetric grid fits ranges that do not straddle zero (e.g. post-ReLU
        activations, all :math:`\ge 0`) without wasting codes on an unused half. The
        default **activation** scheme. Exposed as ``per_tensor_affine``.
    PER_TENSOR_SYMMETRIC
        One ``scale`` for the whole tensor with ``zero_point`` **pinned** (0 for a
        signed grid, mid-range for unsigned), so real 0 maps exactly to a code and the
        grid is symmetric about zero. Exposed as ``per_tensor_symmetric``.
    PER_CHANNEL_AFFINE
        One ``(scale, zero_point)`` **per channel** along ``ch_axis`` with a free
        ``zero_point`` — rarely needed, for weights whose per-channel ranges are
        strongly skewed. Exposed as ``per_channel_affine``.
    PER_CHANNEL_SYMMETRIC
        One ``scale`` per channel with a pinned ``zero_point`` — the standard,
        accuracy-preserving choice for convolution / linear **weights**, and the form
        the low-precision GEMM kernels expect for their weight operand. Exposed as
        ``per_channel_symmetric``.

    Notes
    -----
    - **Affine vs symmetric, precisely.** Affine derives ``scale`` from the full
      ``max - min`` span and solves for ``zero_point`` so the endpoints land on grid
      bounds; symmetric derives ``scale`` from ``max(|min|, |max|)`` over a half-range
      and fixes ``zero_point`` — see :func:`~lucid.quantization.calculate_qparams`.
    - **Why symmetric weights.** Pinning ``zero_point = 0`` removes the cross-terms in
      an integer matmul (a ``w_q · x_q`` product needs no weight-offset correction),
      which is exactly what the packed low-precision GEMM assumes for its weight side.
    - The two boolean helpers :attr:`is_per_channel` / :attr:`is_symmetric` let callers
      branch on granularity / symmetry without matching individual members.

    Examples
    --------
    >>> import lucid.quantization as Q
    >>> Q.per_channel_symmetric is Q.QScheme.PER_CHANNEL_SYMMETRIC
    True
    >>> Q.per_channel_symmetric.is_per_channel
    True
    >>> Q.per_channel_symmetric.is_symmetric
    True
    >>> Q.per_tensor_affine.is_symmetric
    False

    See Also
    --------
    QDtype : The companion descriptor that fixes the integer grid (bits / range).
    lucid.quantization.calculate_qparams : Turns an observed range into
        ``(scale, zero_point)`` per this scheme.
    lucid.quantization.QParams : Bundles a scheme with its derived parameters.
    """

    PER_TENSOR_AFFINE = "per_tensor_affine"
    PER_TENSOR_SYMMETRIC = "per_tensor_symmetric"
    PER_CHANNEL_AFFINE = "per_channel_affine"
    PER_CHANNEL_SYMMETRIC = "per_channel_symmetric"

    @property
    def is_per_channel(self) -> bool:
        r"""``True`` for the two per-channel schemes, ``False`` for per-tensor.

        Lets calibration / quantization code branch on granularity — a per-channel
        scheme derives one ``(scale, zero_point)`` per channel along ``ch_axis``, so
        the qparams are length-``C`` vectors rather than scalars.
        """
        return self in (QScheme.PER_CHANNEL_AFFINE, QScheme.PER_CHANNEL_SYMMETRIC)

    @property
    def is_symmetric(self) -> bool:
        r"""``True`` when ``zero_point`` is pinned (the symmetric schemes).

        A symmetric scheme fixes ``zero_point`` (to 0 for a signed grid, or the grid
        midpoint for unsigned) and derives ``scale`` from ``max(|min|, |max|)``, so
        real 0 always maps exactly to a code; an affine scheme instead solves for a
        free ``zero_point``. See :func:`~lucid.quantization.calculate_qparams`.
        """
        return self in (QScheme.PER_TENSOR_SYMMETRIC, QScheme.PER_CHANNEL_SYMMETRIC)


# Module-level constants mirroring the conventional lowercase qscheme names,
# so call sites read `qscheme=lucid.quantization.per_channel_symmetric`.
per_tensor_affine: QScheme = QScheme.PER_TENSOR_AFFINE
per_tensor_symmetric: QScheme = QScheme.PER_TENSOR_SYMMETRIC
per_channel_affine: QScheme = QScheme.PER_CHANNEL_AFFINE
per_channel_symmetric: QScheme = QScheme.PER_CHANNEL_SYMMETRIC


@dataclass(frozen=True)
class QDtype:
    r"""Descriptor for a quantized integer grid — bits, signedness, range, storage.

    Where a :class:`QScheme` fixes the *geometry* of the affine map, a :class:`QDtype`
    pins the *target grid itself*: how many bits the codes use, whether they are
    signed, the inclusive ``[quant_min, quant_max]`` range they span, and the ordinary
    Lucid dtype that physically stores them. Every observer and fake-quant reads
    ``quant_min`` / ``quant_max`` off a :class:`QDtype` to clamp its codes.

    Unlike the reference framework, Lucid does **not** add quantized dtypes to the C++
    engine's dtype enum; a :class:`QDtype` is a pure-Python descriptor and the integer
    codes live in an ordinary ``storage`` dtype. This keeps the whole quantization
    subsystem off the engine's critical path until a real low-precision GEMM is wired
    in, and lets 4-bit codes ride inside ``int8`` until bit-packing lands.

    Four instances are exported as module-level constants, spanning the grids:

    .. math::

        \mathtt{qint8}:\ [-128,\ 127]\ (\text{8-bit signed}),\qquad
        \mathtt{quint8}:\ [0,\ 255]\ (\text{8-bit unsigned})

    .. math::

        \mathtt{qint4}:\ [-8,\ 7]\ (\text{4-bit signed}),\qquad
        \mathtt{qint32}:\ [-2^{31},\ 2^{31}-1]\ (\text{32-bit signed})

    * ``qint8`` — 8-bit **signed** (``[-128, 127]``), stored in ``int8``. The default
      **weight** grid: symmetric weights centre on 0, so a signed grid uses its full
      range. The workhorse of int8 inference.
    * ``quint8`` — 8-bit **unsigned** (``[0, 255]``), stored in ``int16`` (the engine
      has no ``uint8``, so codes ride in the next wider signed type). The default
      **activation** grid: post-ReLU activations are :math:`\ge 0`, so an unsigned grid
      spends all 256 codes on the range that actually occurs.
    * ``qint4`` — 4-bit **signed** (``[-8, 7]``), stored (unpacked) in ``int8`` until
      4-bit packing lands. The aggressive **weight** grid for the real MLX GEMM: about
      2x smaller than int8 for a modest accuracy cost, used for memory-bound decode.
    * ``qint32`` — 32-bit **signed** (``[-2**31, 2**31 - 1]``), stored in ``int32``.
      The **accumulator** grid: int8 × int8 partial sums accumulate in int32 before
      requantization, so it rarely appears as a weight / activation dtype.

    Parameters
    ----------
    name : str
        Canonical name of the grid (``"qint8"``, ``"quint8"``, …); also drives
        :meth:`__repr__` and appears in module ``extra_repr`` strings.
    bits : int
        Bit width of the grid (``4`` / ``8`` / ``32``).
    signed : bool
        Whether codes are signed. Governs where a symmetric ``zero_point`` is pinned
        (0 when signed, mid-range when unsigned).
    quant_min : int
        Inclusive lower bound of the representable integer range.
    quant_max : int
        Inclusive upper bound of the representable integer range.
    storage : str
        Name of the ordinary Lucid dtype that physically holds the codes. ``quint8``
        maps to ``int16`` (no engine ``uint8``); ``qint4`` maps to ``int8`` (unpacked)
        until 4-bit packing lands.

    Attributes
    ----------
    name, bits, signed, quant_min, quant_max, storage
        The frozen fields above; a :class:`QDtype` is an immutable value object
        (``@dataclass(frozen=True)``), so instances can be shared and compared by value.

    Notes
    -----
    - **Grid width.** ``quant_max - quant_min + 1 == 2 ** bits`` for every instance
      (``256`` for the 8-bit grids, ``16`` for ``qint4``).
    - **Signed / unsigned pairing with schemes.** A symmetric scheme + signed dtype
      pins ``zero_point`` to 0; a symmetric scheme + *unsigned* dtype pins it to the
      grid midpoint. Affine schemes solve for ``zero_point`` regardless.
    - The MLX low-precision GEMM only consumes ``bits`` in {4, 8}; ``qint32`` is an
      accumulator descriptor, not a kernel input.
    - ``repr(qint8)`` is ``"lucid.quantization.qint8"`` — the canonical import path,
      not the field dump — so the descriptor round-trips visually.

    Examples
    --------
    >>> import lucid.quantization as Q
    >>> Q.qint8.bits, Q.qint8.signed, (Q.qint8.quant_min, Q.qint8.quant_max)
    (8, True, (-128, 127))
    >>> Q.quint8.signed, (Q.quint8.quant_min, Q.quint8.quant_max)
    (False, (0, 255))
    >>> Q.qint4.quant_max - Q.qint4.quant_min + 1     # 2 ** 4
    16
    >>> repr(Q.qint8)
    'lucid.quantization.qint8'

    See Also
    --------
    QScheme : The companion descriptor fixing grid geometry (granularity / symmetry).
    lucid.quantization.calculate_qparams : Reads ``quant_min`` / ``quant_max`` to derive
        ``(scale, zero_point)``.
    lucid.quantization.QParams : Bundles a :class:`QDtype` with a scheme and params.
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
