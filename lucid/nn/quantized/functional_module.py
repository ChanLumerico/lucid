"""Quantizable functional merges — ``add`` / ``mul`` / ``cat`` / ``add_relu``.

Element-wise merges (residual skip-adds, concatenations) carry no weight, so
the module-swap machinery never sees them — yet a residual network cannot be
quantized end-to-end unless the merge output is observed and requantized to a
consistent grid.  A model uses :class:`FloatFunctional` in place of a bare
``x + y`` / :func:`lucid.cat` so ``prepare`` can attach an observer to the
result; ``convert`` then swaps it for :class:`QFunctional`, which fake-quantizes
each merge output to the calibrated activation grid (design B — float-carried).
"""

from typing import TYPE_CHECKING, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.nn.quantized._utils import activation_qparams
from lucid.quantization._functional import fake_quantize
from lucid.quantization._qscheme import QDtype, quint8

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


class FloatFunctional(nn.Module):
    """Float-domain quantizable merge ops whose output ``prepare`` observes.

    Element-wise merges — residual skip-adds, gated multiplies, concatenations —
    carry no learnable weight, so the module-swap machinery that quantizes
    ``Linear`` / ``Conv`` never sees a bare ``x + y`` or :func:`lucid.cat`. Yet a
    residual network cannot be quantized end-to-end unless the *merge result* is
    observed and later requantized to a consistent grid: otherwise the two
    branches of an add arrive on different scales and the sum leaks precision.
    ``FloatFunctional`` is the fix — a stateless module you call in place of the
    raw op (``ff.add(x, y)`` instead of ``x + y``) so the merge gets a named site
    the tooling can attach an observer to.

    It is the calibration-time half of the pair. Until
    :func:`lucid.quantization.prepare` installs an ``activation_post_process``
    observer, every method is a plain float op routed through the ``_obs``
    passthrough — the numerics are identical to writing the operator by hand.
    Once prepared, each call additionally feeds its result to the observer, which
    records the merged activation's dynamic range. :func:`lucid.quantization.convert`
    then swaps the whole module for a :class:`QFunctional` carrying those qparams.

    The exposed merges are ``add`` (residual skip-add), ``mul``, ``add_relu``
    (the fused add-then-ReLU of a ResNet block), ``cat`` (concatenation along a
    dim), and the scalar variants ``add_scalar`` / ``mul_scalar``.

    Parameters
    ----------
    None
        Constructed directly with no arguments (``nn.quantized.FloatFunctional()``)
        and placed inside a float model. It gains its observer from ``prepare``
        rather than from constructor arguments, and is replaced — not converted
        in place — by :class:`QFunctional` at ``convert`` time.

    Notes
    -----
    - Reuse **one** ``FloatFunctional`` per merge site; do not share a single
      instance across unrelated adds, since they would then share one observer
      and be forced onto a common, looser grid.
    - Before ``prepare`` it is a transparent passthrough, so inserting it into a
      float model never changes float-mode numerics — safe to add pre-emptively.
    - Only the *output* is observed; the two operands keep whatever grids their
      producing layers assigned.

    Examples
    --------
    Residual skip-add inside a block — call ``add`` instead of ``+``:

    >>> import lucid.nn as nn
    >>> class Block(nn.Module):
    ...     def __init__(self) -> None:
    ...         super().__init__()
    ...         self.conv = nn.Conv2d(8, 8, 3, padding=1)
    ...         self.skip = nn.quantized.FloatFunctional()
    ...     def forward(self, x):
    ...         return self.skip.add(self.conv(x), x)   # not `self.conv(x) + x`

    The common mistake is leaving the raw operator in place — the sum then has no
    observer and cannot be requantized to a single grid:

    >>> ff = nn.quantized.FloatFunctional()
    >>> import lucid
    >>> a, b = lucid.randn(4), lucid.randn(4)
    >>> bool((ff.add(a, b) == a + b).all().item())      # identity pre-prepare
    True

    See Also
    --------
    lucid.nn.quantized.QFunctional : The converted, requantizing counterpart.
    lucid.quantization.prepare : Attaches the observer these merges feed.
    lucid.quantization.convert : Swaps this module for :class:`QFunctional`.
    """

    def _obs(self, x: Tensor) -> Tensor:
        """Feed the result to the attached observer (identity before ``prepare``)."""
        app = getattr(self, "activation_post_process", None)
        return cast("Tensor", app(x)) if app is not None else x

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        """Observed ``x + y`` (a residual skip-add)."""
        return self._obs(x + y)

    def mul(self, x: Tensor, y: Tensor) -> Tensor:
        """Observed ``x * y``."""
        return self._obs(x * y)

    def add_relu(self, x: Tensor, y: Tensor) -> Tensor:
        """Observed ``relu(x + y)`` — the fused residual-add of a ResNet block."""
        return self._obs(F.relu(x + y))

    def cat(self, tensors: list[Tensor], dim: int = 0) -> Tensor:
        """Observed concatenation along ``dim``."""
        return self._obs(lucid.cat(tensors, dim=dim))

    def add_scalar(self, x: Tensor, scalar: float) -> Tensor:
        """Observed ``x + scalar``."""
        return self._obs(x + scalar)

    def mul_scalar(self, x: Tensor, scalar: float) -> Tensor:
        """Observed ``x * scalar``."""
        return self._obs(x * scalar)


class QFunctional(nn.Module):
    r"""Quantized element-wise merges — each result fake-quantized to one grid.

    The converted, inference-time form of :class:`FloatFunctional`, installed by
    :func:`lucid.quantization.convert` (or built via :meth:`from_float`). It
    exposes the identical surface — ``add``, ``mul``, ``add_relu``, ``cat``,
    ``add_scalar``, ``mul_scalar`` — but every call now *fake-quantizes* its
    output onto the single ``(scale, zero_point)`` grid the source
    :class:`FloatFunctional` observed during calibration.

    This is what makes residual quantization correct. A skip-add feeds two
    branches that were quantized on their own scales; requantizing the sum to one
    calibrated grid re-aligns them, so the block's output matches int8 numerics
    instead of drifting off at full precision. Under the sidecar representation
    (design B) the requantized result is carried as ``float32`` with int8
    numerics — no packed-integer container is produced.

    Every merge routes its raw float result :math:`t` through

    .. math::

        \operatorname{fake\_quant}(t) = \bigl(
            \operatorname{clamp}(\operatorname{round}(t/S) + Z,\ q_{\min},\ q_{\max})
            - Z\bigr)\, S

    with the calibrated ``(scale, zero_point)`` :math:`S, Z` and grid bounds
    :math:`q_{\min}, q_{\max}` (``0, 255`` for the default ``quint8``). Only the
    output is snapped; ``add_relu`` applies the ReLU *before* this step so the
    grid matches the post-ReLU range calibration saw.

    Parameters
    ----------
    scale : Tensor
        Per-tensor output scale, from the source ``FloatFunctional``'s observer.
    zero_point : Tensor
        Per-tensor output zero-point, from that same observer.
    qdtype : QDtype, optional
        Quantized dtype whose ``quant_min`` / ``quant_max`` bound the grid.
        Defaults to :data:`~lucid.quantization.quint8` (``[0, 255]``).

    Attributes
    ----------
    scale : Tensor
        Scalar output-scale buffer shared by every merge method.
    zero_point : Tensor
        Scalar output zero-point buffer shared by every merge method.

    Notes
    -----
    - Normally produced by ``convert`` / :meth:`from_float`, which reads the
      calibrated ``FloatFunctional``'s observer; construct directly only when you
      already hold the qparams.
    - All merge methods share one grid — the module is bound to a single merge
      site, mirroring the one-instance-per-site rule for ``FloatFunctional``.
    - ``add_relu`` fuses add + ReLU into one requantization; prefer it over
      ``F.relu(qf.add(x, y))`` so the observed and executed ranges agree.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> qf = nn.quantized.QFunctional(lucid.tensor(0.1), lucid.tensor(0.0))
    >>> a = lucid.randn(4)
    >>> qf.add(a, a).shape                       # residual add, requantized
    (4,)

    The fused residual-add of a ResNet block clamps negatives *before* the grid:

    >>> out = qf.add_relu(lucid.randn(4), lucid.randn(4))
    >>> bool((out >= 0).all().item())
    True

    See Also
    --------
    lucid.nn.quantized.FloatFunctional : The calibration-time counterpart.
    lucid.quantization.fake_quantize : The op applied to each merge output.
    lucid.quantization.convert : Installs this module from a calibrated model.
    """

    scale: Tensor
    zero_point: Tensor

    def __init__(
        self, scale: Tensor, zero_point: Tensor, qdtype: QDtype = quint8
    ) -> None:
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        self.qdtype = qdtype

    def _q(self, x: Tensor) -> Tensor:
        """Fake-quantize ``x`` to the calibrated output grid."""
        return fake_quantize(
            x, self.scale, self.zero_point, self.qdtype.quant_min, self.qdtype.quant_max
        )

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        """Quantized ``x + y``."""
        return self._q(x + y)

    def mul(self, x: Tensor, y: Tensor) -> Tensor:
        """Quantized ``x * y``."""
        return self._q(x * y)

    def add_relu(self, x: Tensor, y: Tensor) -> Tensor:
        """Quantized ``relu(x + y)``."""
        return self._q(F.relu(x + y))

    def cat(self, tensors: list[Tensor], dim: int = 0) -> Tensor:
        """Quantized concatenation along ``dim``."""
        return self._q(lucid.cat(tensors, dim=dim))

    def add_scalar(self, x: Tensor, scalar: float) -> Tensor:
        """Quantized ``x + scalar``."""
        return self._q(x + scalar)

    def mul_scalar(self, x: Tensor, scalar: float) -> Tensor:
        """Quantized ``x * scalar``."""
        return self._q(x * scalar)

    @classmethod
    def from_float(cls, mod: nn.Module) -> QFunctional:
        """Build from a calibrated :class:`FloatFunctional` (reads its observer)."""
        scale, zero_point, qdtype = activation_qparams(mod)
        return cls(scale, zero_point, qdtype)
