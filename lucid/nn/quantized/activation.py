"""Quantized activation modules — float activation + output fake-quant.

Under the sidecar design (B) activations are carried as (fake-quantized)
``float32``, so a *quantized* activation is simply the float activation followed
by a fake-quantize to the calibrated output grid.  ``convert`` swaps a calibrated
float ``nn.Sigmoid`` / ``nn.Hardswish`` / … for these so a standalone activation
inside a quantized region is requantized rather than left at full precision.
"""

from typing import TYPE_CHECKING, cast, override

import lucid.nn as nn
import lucid.nn.functional as F
from lucid.nn.quantized._utils import activation_qparams
from lucid.quantization._functional import fake_quantize
from lucid.quantization._qscheme import QDtype, quint8

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


class _QuantizedActivation(nn.Module):
    """Float activation then fake-quant to the calibrated ``(scale, zero_point)``."""

    scale: Tensor
    zero_point: Tensor

    def __init__(
        self, scale: Tensor, zero_point: Tensor, qdtype: QDtype = quint8
    ) -> None:
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        self.qdtype = qdtype

    def _act(self, x: Tensor) -> Tensor:
        """Apply the float activation — overridden per subclass."""
        raise NotImplementedError

    def _configure(self, mod: nn.Module) -> None:
        """Copy activation-specific params (e.g. ``alpha``) from the float module."""

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary activation
        """Activate, then fake-quantize to the calibrated grid."""
        return fake_quantize(
            self._act(x),
            self.scale,
            self.zero_point,
            self.qdtype.quant_min,
            self.qdtype.quant_max,
        )

    @classmethod
    def from_float(cls, mod: nn.Module) -> "_QuantizedActivation":
        """Build from a calibrated float activation (reads its output observer)."""
        scale, zero_point, qdtype = activation_qparams(mod)
        obj = cls(scale, zero_point, qdtype)
        obj._configure(mod)
        return obj


class Sigmoid(_QuantizedActivation):
    r"""Quantized :class:`~lucid.nn.Sigmoid` — ``sigmoid(x)`` then output fake-quant.

    The inference-time replacement for a calibrated float ``nn.Sigmoid`` sitting
    inside a quantized region, installed by :func:`lucid.quantization.convert` /
    :meth:`from_float`. Under the sidecar representation (design B) a *quantized*
    activation is simply the float activation followed by a fake-quantize to the
    calibrated output grid, so a standalone sigmoid is requantized rather than
    left at full precision between two int8 layers.

    Because :math:`\sigma(x) \in [0, 1]` the output range is known a priori, this
    is the archetypal **fixed-qparams** activation: instead of calibrating the
    range from data, pair it with a
    :class:`~lucid.quantization.FixedQParamsObserver` in the qconfig, which pins
    the grid to the constant :math:`[0, 1]` mapping.

    .. math::

        y = \operatorname{fake\_quant}\bigl(\sigma(x)\bigr),
        \qquad
        S = \tfrac{1}{256},\quad Z = 0

    so the ``quint8`` codes ``0 .. 255`` tile :math:`[0, 1]` uniformly with step
    :math:`S`. :math:`\operatorname{fake\_quant}(t) = (\operatorname{clamp}(
    \operatorname{round}(t/S) + Z, q_{\min}, q_{\max}) - Z)\, S`.

    Parameters
    ----------
    scale : Tensor
        Per-tensor output scale — the fixed ``1/256`` when paired with a
        :class:`~lucid.quantization.FixedQParamsObserver`, else the calibrated
        value. Supplied by :meth:`from_float`; construct directly only with
        qparams in hand.
    zero_point : Tensor
        Per-tensor output zero-point (``0`` for the fixed ``[0, 1]`` grid).
    qdtype : QDtype, optional
        Quantized dtype bounding the grid. Defaults to
        :data:`~lucid.quantization.quint8` (``[0, 255]``).

    Notes
    -----
    - Built via ``convert`` / :meth:`from_float`, which reads the source module's
      output observer; a bare ``nn.Sigmoid`` is not one of these.
    - Pairing with :class:`~lucid.quantization.FixedQParamsObserver` skips a
      calibration pass for a range that is fixed by construction.
    - The forward is ``float sigmoid → fake-quant``; the sigmoid itself runs at
      full float precision, only the output is snapped to the grid.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> qsig = nn.quantized.Sigmoid(lucid.tensor(1 / 256), lucid.tensor(0.0))
    >>> y = qsig(lucid.randn(8))
    >>> bool(((y >= 0) & (y <= 1)).all().item())     # bounded to [0, 1]
    True

    See Also
    --------
    lucid.nn.quantized.Tanh : Bounded to ``[-1, 1]`` (also fixed-qparams).
    lucid.nn.quantized.Hardsigmoid : Piecewise-linear ``[0, 1]`` gate.
    lucid.quantization.FixedQParamsObserver : Pins the grid for bounded outputs.
    """

    @override
    def _act(self, x: Tensor) -> Tensor:
        return F.sigmoid(x)


class Hardswish(_QuantizedActivation):
    r"""Quantized :class:`~lucid.nn.Hardswish` — ``hardswish(x)`` then output fake-quant.

    The inference-time replacement for a calibrated float ``nn.Hardswish``,
    installed by :func:`lucid.quantization.convert` / :meth:`from_float`.
    Hardswish is the cheap piecewise-linear approximation of ``x·sigmoid(x)``
    used throughout MobileNetV3 / EfficientNet-lite, chosen precisely because its
    clamp-and-multiply form quantizes cleanly. Under the sidecar representation
    (design B) the quantized layer is the float activation followed by a
    fake-quantize to the calibrated output grid.

    Unlike :class:`Sigmoid` / :class:`Tanh`, hardswish is **unbounded above**
    (it grows like :math:`x` for large :math:`x`), so its output range is *not*
    known a priori — it must be calibrated with a real observer rather than a
    :class:`~lucid.quantization.FixedQParamsObserver`.

    .. math::

        y = \operatorname{fake\_quant}\bigl(\operatorname{hardswish}(x)\bigr),
        \qquad
        \operatorname{hardswish}(x) = x\,
            \frac{\operatorname{clamp}(x + 3,\ 0,\ 6)}{6}

    with :math:`\operatorname{fake\_quant}(t) = (\operatorname{clamp}(
    \operatorname{round}(t/S) + Z, q_{\min}, q_{\max}) - Z)\, S` and the
    calibrated output ``(scale, zero_point)`` :math:`S, Z`.

    Parameters
    ----------
    scale : Tensor
        Per-tensor output scale from the calibrating observer. Supplied by
        :meth:`from_float`; construct directly only with qparams in hand.
    zero_point : Tensor
        Per-tensor output zero-point from the calibrating observer.
    qdtype : QDtype, optional
        Quantized dtype bounding the grid. Defaults to
        :data:`~lucid.quantization.quint8` (``[0, 255]``).

    Notes
    -----
    - Built via ``convert`` / :meth:`from_float`, which reads the source module's
      output observer.
    - Its range is data-dependent, so calibrate through the prepared model before
      ``convert`` — a fixed-qparams grid would clip the unbounded upper tail.
    - The forward is ``float hardswish → fake-quant``; only the output is snapped.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> qhs = nn.quantized.Hardswish(lucid.tensor(0.05), lucid.tensor(64.0))
    >>> qhs(lucid.randn(8)).shape
    (8,)

    See Also
    --------
    lucid.nn.quantized.Sigmoid : Bounded ``[0, 1]`` (fixed-qparams-friendly).
    lucid.nn.quantized.Hardsigmoid : The gate paired with hardswish in SE blocks.
    lucid.quantization.convert : Installs this layer from a calibrated model.
    """

    @override
    def _act(self, x: Tensor) -> Tensor:
        return F.hardswish(x)


class Hardsigmoid(_QuantizedActivation):
    r"""Quantized :class:`~lucid.nn.Hardsigmoid` — ``hardsigmoid(x)`` then fake-quant.

    The inference-time replacement for a calibrated float ``nn.Hardsigmoid``,
    installed by :func:`lucid.quantization.convert` / :meth:`from_float`.
    Hardsigmoid is the piecewise-linear approximation of the sigmoid gate used in
    MobileNetV3's squeeze-and-excitation blocks; like :class:`Sigmoid` its output
    is bounded to :math:`[0, 1]`, so it is a natural **fixed-qparams** activation.
    Under the sidecar representation (design B) the quantized layer is the float
    activation followed by a fake-quantize to the calibrated output grid.

    Because the range is fixed to :math:`[0, 1]`, pair it with a
    :class:`~lucid.quantization.FixedQParamsObserver` (scale :math:`1/256`,
    zero-point :math:`0`) to skip calibrating a range known a priori.

    .. math::

        y = \operatorname{fake\_quant}\bigl(\operatorname{hardsigmoid}(x)\bigr),
        \qquad
        \operatorname{hardsigmoid}(x) =
            \frac{\operatorname{clamp}(x + 3,\ 0,\ 6)}{6}

    with :math:`\operatorname{fake\_quant}(t) = (\operatorname{clamp}(
    \operatorname{round}(t/S) + Z, q_{\min}, q_{\max}) - Z)\, S` and, for the
    fixed grid, :math:`S = 1/256,\ Z = 0`.

    Parameters
    ----------
    scale : Tensor
        Per-tensor output scale — the fixed ``1/256`` under a
        :class:`~lucid.quantization.FixedQParamsObserver`, else calibrated.
        Supplied by :meth:`from_float`.
    zero_point : Tensor
        Per-tensor output zero-point (``0`` for the fixed ``[0, 1]`` grid).
    qdtype : QDtype, optional
        Quantized dtype bounding the grid. Defaults to
        :data:`~lucid.quantization.quint8` (``[0, 255]``).

    Notes
    -----
    - Built via ``convert`` / :meth:`from_float`, which reads the source module's
      output observer.
    - Bounded to ``[0, 1]``, so pairing with
      :class:`~lucid.quantization.FixedQParamsObserver` avoids a calibration pass.
    - Often paired with :class:`Hardswish` inside the same SE block; both are the
      hardware-friendly piecewise-linear substitutes for their smooth originals.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> qhs = nn.quantized.Hardsigmoid(lucid.tensor(1 / 256), lucid.tensor(0.0))
    >>> y = qhs(lucid.randn(8))
    >>> bool(((y >= 0) & (y <= 1)).all().item())     # bounded to [0, 1]
    True

    See Also
    --------
    lucid.nn.quantized.Sigmoid : The smooth ``[0, 1]`` original.
    lucid.nn.quantized.Hardswish : Its partner activation in SE blocks.
    lucid.quantization.FixedQParamsObserver : Pins the grid for bounded outputs.
    """

    @override
    def _act(self, x: Tensor) -> Tensor:
        return F.hardsigmoid(x)


class Tanh(_QuantizedActivation):
    r"""Quantized :class:`~lucid.nn.Tanh` — ``tanh(x)`` then output fake-quant.

    The inference-time replacement for a calibrated float ``nn.Tanh``, installed
    by :func:`lucid.quantization.convert` / :meth:`from_float`. Under the sidecar
    representation (design B) the quantized layer is the float ``tanh`` followed
    by a fake-quantize to the calibrated output grid, so a standalone tanh inside
    a quantized region is requantized rather than left at full precision.

    Because :math:`\tanh(x) \in [-1, 1]` the range is known a priori, making this
    another **fixed-qparams** activation. Since the range is *signed*, the fixed
    grid centres the zero-point in the middle of the ``quint8`` codes so that both
    the ``-1`` and ``+1`` extremes are representable.

    .. math::

        y = \operatorname{fake\_quant}\bigl(\tanh(x)\bigr),
        \qquad
        S = \tfrac{2}{256},\quad Z = 128

    so codes ``0 .. 255`` tile :math:`[-1, 1]` with the real value ``0`` landing
    on code ``128``. :math:`\operatorname{fake\_quant}(t) = (\operatorname{clamp}(
    \operatorname{round}(t/S) + Z, q_{\min}, q_{\max}) - Z)\, S`.

    Parameters
    ----------
    scale : Tensor
        Per-tensor output scale — the fixed ``2/256`` under a
        :class:`~lucid.quantization.FixedQParamsObserver`, else calibrated.
        Supplied by :meth:`from_float`.
    zero_point : Tensor
        Per-tensor output zero-point (``128`` for the signed ``[-1, 1]`` grid).
    qdtype : QDtype, optional
        Quantized dtype bounding the grid. Defaults to
        :data:`~lucid.quantization.quint8` (``[0, 255]``).

    Notes
    -----
    - Built via ``convert`` / :meth:`from_float`, which reads the source module's
      output observer.
    - Bounded to ``[-1, 1]``; pairing with
      :class:`~lucid.quantization.FixedQParamsObserver` avoids a calibration pass.
    - The non-zero zero-point (``128``) is what lets a *signed* range live on the
      unsigned ``quint8`` grid without clipping either sign.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> qtanh = nn.quantized.Tanh(lucid.tensor(2 / 256), lucid.tensor(128.0))
    >>> y = qtanh(lucid.randn(8))
    >>> bool(((y >= -1) & (y <= 1)).all().item())    # bounded to [-1, 1]
    True

    See Also
    --------
    lucid.nn.quantized.Sigmoid : Bounded ``[0, 1]`` (unsigned fixed grid).
    lucid.quantization.FixedQParamsObserver : Pins the grid for bounded outputs.
    lucid.quantization.convert : Installs this layer from a calibrated model.
    """

    @override
    def _act(self, x: Tensor) -> Tensor:
        return F.tanh(x)


class ELU(_QuantizedActivation):
    r"""Quantized :class:`~lucid.nn.ELU` — ``elu(x, alpha)`` then output fake-quant.

    The inference-time replacement for a calibrated float ``nn.ELU``, installed
    by :func:`lucid.quantization.convert` / :meth:`from_float`. Under the sidecar
    representation (design B) the quantized layer is the float ELU followed by a
    fake-quantize to the calibrated output grid. Unlike the bounded activations,
    ELU is **unbounded above** and saturates to :math:`-\alpha` below, so its
    range is data-dependent and must be calibrated with a real observer rather
    than a :class:`~lucid.quantization.FixedQParamsObserver`.

    The negative-branch scale :math:`\alpha` is a parameter of the activation,
    not a learned weight; :meth:`from_float` copies it from the source ``nn.ELU``
    (via the internal ``_configure`` hook) so the quantized layer reproduces the
    exact float curve before snapping the output to the grid.

    .. math::

        y = \operatorname{fake\_quant}\bigl(\operatorname{elu}(x)\bigr),
        \qquad
        \operatorname{elu}(x) =
            \begin{cases} x & x > 0 \\ \alpha\,(e^{x} - 1) & x \le 0 \end{cases}

    with :math:`\operatorname{fake\_quant}(t) = (\operatorname{clamp}(
    \operatorname{round}(t/S) + Z, q_{\min}, q_{\max}) - Z)\, S` and the
    calibrated output ``(scale, zero_point)`` :math:`S, Z`.

    Parameters
    ----------
    scale : Tensor
        Per-tensor output scale from the calibrating observer. Supplied by
        :meth:`from_float`; construct directly only with qparams in hand.
    zero_point : Tensor
        Per-tensor output zero-point from the calibrating observer.
    qdtype : QDtype, optional
        Quantized dtype bounding the grid. Defaults to
        :data:`~lucid.quantization.quint8` (``[0, 255]``).

    Attributes
    ----------
    alpha : float
        The negative-branch scale :math:`\alpha` in :math:`\alpha\,(e^{x} - 1)`,
        copied from the source ``nn.ELU`` by :meth:`from_float`. Defaults to
        ``1.0`` on a bare instance.

    Notes
    -----
    - Built via ``convert`` / :meth:`from_float`, which reads the output observer
      and copies ``alpha`` from the float module.
    - Range is data-dependent (unbounded above), so calibrate through the
      prepared model before ``convert``; a fixed grid would clip the upper tail.
    - The forward is ``float elu(x, alpha) → fake-quant``; only the output is
      snapped to the grid.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> src = nn.ELU(alpha=1.5)
    >>> qelu = nn.quantized.ELU.from_float(src)   # (needs a calibrated observer)
    >>> qelu.alpha
    1.5

    See Also
    --------
    lucid.nn.quantized.LeakyReLU : Also carries a slope parameter across convert.
    lucid.nn.quantized.Hardswish : Another unbounded, calibration-only activation.
    lucid.quantization.convert : Installs this layer from a calibrated model.
    """

    alpha: float = 1.0

    @override
    def _configure(self, mod: nn.Module) -> None:
        self.alpha = float(cast("float", getattr(mod, "alpha", 1.0)))

    @override
    def _act(self, x: Tensor) -> Tensor:
        return F.elu(x, self.alpha)


class LeakyReLU(_QuantizedActivation):
    r"""Quantized :class:`~lucid.nn.LeakyReLU` — ``leaky_relu(x, slope)`` then fake-quant.

    The inference-time replacement for a calibrated float ``nn.LeakyReLU``,
    installed by :func:`lucid.quantization.convert` / :meth:`from_float`. Under
    the sidecar representation (design B) the quantized layer is the float
    leaky-ReLU followed by a fake-quantize to the calibrated output grid. Like
    :class:`ELU` its range is **unbounded** on both sides (linear far from the
    origin), so it is calibrated with a real observer, not a
    :class:`~lucid.quantization.FixedQParamsObserver`.

    The ``negative_slope`` — the small gradient applied to :math:`x < 0` that
    keeps dead-unit gradients alive — is a parameter of the activation;
    :meth:`from_float` copies it from the source ``nn.LeakyReLU`` (via the
    internal ``_configure`` hook) so the quantized curve matches the float one
    before the output is snapped to the grid.

    .. math::

        y = \operatorname{fake\_quant}\bigl(\operatorname{leaky\_relu}(x)\bigr),
        \qquad
        \operatorname{leaky\_relu}(x) =
            \begin{cases} x & x \ge 0 \\ \text{slope}\cdot x & x < 0 \end{cases}

    with :math:`\operatorname{fake\_quant}(t) = (\operatorname{clamp}(
    \operatorname{round}(t/S) + Z, q_{\min}, q_{\max}) - Z)\, S` and the
    calibrated output ``(scale, zero_point)`` :math:`S, Z`.

    Parameters
    ----------
    scale : Tensor
        Per-tensor output scale from the calibrating observer. Supplied by
        :meth:`from_float`; construct directly only with qparams in hand.
    zero_point : Tensor
        Per-tensor output zero-point from the calibrating observer.
    qdtype : QDtype, optional
        Quantized dtype bounding the grid. Defaults to
        :data:`~lucid.quantization.quint8` (``[0, 255]``).

    Attributes
    ----------
    negative_slope : float
        The slope applied to :math:`x < 0`, copied from the source
        ``nn.LeakyReLU`` by :meth:`from_float`. Defaults to ``0.01`` on a bare
        instance.

    Notes
    -----
    - Built via ``convert`` / :meth:`from_float`, which reads the output observer
      and copies ``negative_slope`` from the float module.
    - Range is unbounded on both sides, so calibrate through the prepared model
      before ``convert`` rather than using a fixed grid.
    - The forward is ``float leaky_relu(x, slope) → fake-quant``; only the output
      is snapped to the grid.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> src = nn.LeakyReLU(negative_slope=0.2)
    >>> qlr = nn.quantized.LeakyReLU.from_float(src)   # (needs a calibrated obs.)
    >>> qlr.negative_slope
    0.2

    See Also
    --------
    lucid.nn.quantized.ELU : Also carries a shape parameter across convert.
    lucid.nn.quantized.ConvReLU2d : Fused ReLU folded into a quantized conv.
    lucid.quantization.convert : Installs this layer from a calibrated model.
    """

    negative_slope: float = 0.01

    @override
    def _configure(self, mod: nn.Module) -> None:
        self.negative_slope = float(cast("float", getattr(mod, "negative_slope", 0.01)))

    @override
    def _act(self, x: Tensor) -> Tensor:
        return F.leaky_relu(x, self.negative_slope)
