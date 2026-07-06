"""Quantization boundary markers and their converted forms.

``QuantStub`` / ``DeQuantStub`` are placed in a **float** model to mark
where the quantized region begins and ends.  ``prepare`` attaches an
activation observer to the ``QuantStub``; ``convert`` then swaps the stubs
for their runtime forms:

* :class:`Quantize` — fake-quantizes its input to the calibrated
  activation ``(scale, zero_point)``, i.e. the entry into the quantized
  region.
* :class:`DeQuantize` — the exit.  Under Lucid's sidecar representation
  (design B) activations are carried as (fake-quantized) ``float32``
  throughout, so dequantization is the identity.
"""

from typing import TYPE_CHECKING, cast, override

import lucid
import lucid.nn as nn
from lucid.quantization._fake_quantize import FakeQuantize
from lucid.quantization._functional import fake_quantize
from lucid.quantization._qscheme import QDtype, quint8

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.quantization.observer import ObserverBase


class QuantStub(nn.Module):
    r"""Float→quantized boundary marker — the entry stub of a quantizable model.

    Placed in a **float** model at the point where the quantized region should
    begin. On its own it does nothing to the numerics; it exists so the
    quantization tooling has a named site to attach an activation observer and,
    later, to swap in a runtime quantizer. It is the mirror image of
    :class:`DeQuantStub`, which marks the exit.

    Its behaviour depends on which workflow prepared the model:

    * **PTQ** (:func:`lucid.quantization.prepare`) attaches a bare observer as a
      forward hook and leaves the stub an identity — the observer passively
      records the activation range while calibration data flows through.
    * **QAT** (:func:`lucid.quantization.prepare_qat`) attaches a
      :class:`~lucid.quantization.FakeQuantize` as ``activation_post_process``;
      the stub then fake-quantizes its (floating-point) input on every forward so
      the network trains against the same rounding it will meet at inference.

    Once calibrated, :func:`lucid.quantization.convert` replaces the stub with a
    runtime :class:`Quantize`. In the QAT path the fake-quantize applied to a
    real-valued input is

    .. math::

        \operatorname{fake\_quant}(x) = \bigl(
            \operatorname{clamp}(\operatorname{round}(x/S) + Z,\ q_{\min},\ q_{\max})
            - Z\bigr)\, S

    with the observed activation ``(scale, zero_point)`` :math:`S, Z` and grid
    bounds :math:`q_{\min}, q_{\max}` (``0, 255`` for the default ``quint8``).
    Integer inputs (e.g. token indices) are passed through untouched.

    Parameters
    ----------
    qconfig : object, optional
        Quantization config to attach to this boundary. When ``None`` (the
        default) no ``qconfig`` attribute is set and the stub inherits the
        surrounding model's config during ``prepare`` / ``convert``.

    Notes
    -----
    - A raw ``QuantStub()`` is a no-op identity; it only acquires an
      ``activation_post_process`` after ``prepare`` / ``prepare_qat`` runs.
    - Use :class:`QuantWrapper` to bolt a ``QuantStub`` / ``DeQuantStub`` pair
      onto a model that has no explicit boundaries of its own.
    - It converts to :class:`Quantize`, which carries the calibrated qparams as
      buffers; the stub itself holds no scale / zero-point.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> class Net(nn.Module):
    ...     def __init__(self) -> None:
    ...         super().__init__()
    ...         self.quant = nn.quantized.QuantStub()
    ...         self.fc = nn.Linear(4, 2)
    ...         self.dequant = nn.quantized.DeQuantStub()
    ...     def forward(self, x):
    ...         return self.dequant(self.fc(self.quant(x)))

    A stub used *before* ``prepare`` is a plain identity — the common mistake is
    to expect it to quantize on its own:

    >>> stub = nn.quantized.QuantStub()      # no observer attached yet
    >>> x = lucid.randn(3)
    >>> bool((stub(x) == x).all().item())    # identity until prepare() runs
    True

    See Also
    --------
    lucid.nn.quantized.DeQuantStub : The matching exit-side boundary marker.
    lucid.nn.quantized.Quantize : The runtime form this stub converts into.
    lucid.nn.quantized.QuantWrapper : Wraps a model in a stub pair.
    lucid.quantization.prepare : Attaches the observer this stub feeds.
    """

    def __init__(self, qconfig: object = None) -> None:
        super().__init__()
        if qconfig is not None:
            self.qconfig = qconfig

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary boundary marker
        """Identity for PTQ (a hook observes); fake-quant for QAT.

        In QAT ``prepare_qat`` attaches a :class:`FakeQuantize` as
        ``activation_post_process`` — apply it so the input is fake-quantized
        during training.  In PTQ it's a bare observer fed by a forward hook,
        so the stub stays an identity.
        """
        app = getattr(self, "activation_post_process", None)
        if isinstance(app, FakeQuantize) and lucid.is_floating_point(x):
            return cast("Tensor", app(x))
        return x


class DeQuantStub(nn.Module):
    """Quantized→float boundary marker — the exit stub of a quantizable model.

    The exit-side counterpart of :class:`QuantStub`: place it where the
    quantized region ends and ordinary float compute resumes. Like its
    entry-side twin it is a placeholder — an identity at float / calibration
    time whose only job is to give the tooling a named site — and
    :func:`lucid.quantization.convert` swaps it for a runtime
    :class:`DeQuantize`.

    Under Lucid's sidecar representation (design B) activations are carried as
    (fake-quantized) ``float32`` throughout the quantized region rather than as
    packed integer tensors, so "dequantizing" the exit is a no-op: there is no
    integer buffer to unpack back to float. The marker therefore never changes a
    tensor's values in either its stub or its converted form — it exists purely
    so the quantized graph is symmetric and walkable by tooling.

    Parameters
    ----------
    qconfig : object, optional
        Quantization config attached to this boundary. When ``None`` (the
        default) no ``qconfig`` attribute is set and the module inherits the
        surrounding model's config during ``prepare`` / ``convert``.

    Notes
    -----
    - Holds no state and takes no runtime tensor arguments beyond the input.
    - Because activations stay float-carried, a model may omit ``DeQuantStub``
      entirely and still produce correct numerics; it is kept for graph
      symmetry and to mark intent.
    - Converts to :class:`DeQuantize`, which is likewise a pure identity — so a
      quantized region is bracketed by exactly one *active* boundary
      (:class:`Quantize` on entry).

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> dq = nn.quantized.DeQuantStub()
    >>> x = lucid.randn(2, 3)
    >>> bool((dq(x) == x).all().item())      # identity in every mode
    True

    See Also
    --------
    lucid.nn.quantized.QuantStub : The matching entry-side boundary marker.
    lucid.nn.quantized.DeQuantize : The runtime form this stub converts into.
    lucid.nn.quantized.QuantWrapper : Wraps a model in a stub pair.
    """

    def __init__(self, qconfig: object = None) -> None:
        super().__init__()
        if qconfig is not None:
            self.qconfig = qconfig

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary boundary marker
        """Identity."""
        return x


class Quantize(nn.Module):
    r"""Runtime entry into the quantized region — fake-quantizes the input.

    The converted, inference-time form of a calibrated :class:`QuantStub`,
    installed by :func:`lucid.quantization.convert` (or built directly via
    :meth:`from_float`). Where the stub was a passive marker, ``Quantize`` is
    active: every forward maps a real-valued activation onto the calibrated
    ``(scale, zero_point)`` grid the observer recorded, which is the numerical
    "entry" into the quantized region.

    Under the sidecar representation (design B) the quantized activation is
    *carried as float* — the result has int8 numerics (values snapped to the
    grid) but remains an ordinary ``float32`` tensor, so it flows through the
    rest of the quantized graph without any packed-integer container. Integer
    inputs such as token indices are recognised and passed through unchanged,
    since only real-valued activations have a meaningful quantization grid.

    .. math::

        \operatorname{fake\_quant}(x) = \bigl(
            \operatorname{clamp}(\operatorname{round}(x/S) + Z,\ q_{\min},\ q_{\max})
            - Z\bigr)\, S

    where :math:`S, Z` are the calibrated activation ``(scale, zero_point)`` and
    :math:`q_{\min}, q_{\max}` the grid bounds (``0, 255`` for the default
    ``quint8``). The inner round-and-clamp is what quantizes; the outer
    :math:`(\cdot - Z)\, S` re-expresses the integer code as the nearest
    representable float.

    Parameters
    ----------
    scale : Tensor
        Per-tensor activation scale from the calibrating observer.
    zero_point : Tensor
        Per-tensor activation zero-point from the calibrating observer.
    qdtype : QDtype, optional
        Quantized dtype whose ``quant_min`` / ``quant_max`` bound the grid.
        Defaults to :data:`~lucid.quantization.quint8` (``[0, 255]``).

    Attributes
    ----------
    scale : Tensor
        Scalar activation-scale buffer, set from the calibrated observer.
    zero_point : Tensor
        Scalar activation zero-point buffer, set from the calibrated observer.

    Notes
    -----
    - Normally produced by ``convert`` / :meth:`from_float`, which reads the
      calibrated ``QuantStub``'s observer via ``calculate_qparams``; construct
      directly only when you already hold the qparams.
    - The input's dtype decides the path: floating-point tensors are
      fake-quantized, integer tensors are returned verbatim.
    - The exit-side :class:`DeQuantize` is an identity, so a quantized region is
      bracketed by exactly one active boundary (this one).

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> q = nn.quantized.Quantize(lucid.tensor(0.05), lucid.tensor(128.0))
    >>> q(lucid.randn(4)).shape              # snapped to the 0.05 grid
    (4,)

    Integer inputs bypass quantization entirely (a common surprise):

    >>> idx = lucid.tensor([1, 2, 3])
    >>> bool((q(idx) == idx).all().item())
    True

    See Also
    --------
    lucid.nn.quantized.QuantStub : The float-model marker this converts from.
    lucid.nn.quantized.DeQuantize : The matching (identity) exit marker.
    lucid.quantization.fake_quantize : The op applied on each forward.
    lucid.quantization.convert : Installs this layer from a calibrated model.
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

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary boundary marker
        """Quantize ``x`` to the calibrated activation grid (float-carried).

        Integer inputs (e.g. token indices) are passed through unchanged —
        only real-valued activations are quantized.
        """
        if not lucid.is_floating_point(x):
            return x
        return fake_quantize(
            x, self.scale, self.zero_point, self.qdtype.quant_min, self.qdtype.quant_max
        )

    @classmethod
    def from_float(cls, stub: nn.Module) -> Quantize:
        """Build from a calibrated :class:`QuantStub` (reads its observer)."""
        obs = cast("ObserverBase", stub.activation_post_process)
        scale, zero_point = obs.calculate_qparams()
        return cls(scale, zero_point, obs.qdtype)


class DeQuantize(nn.Module):
    """Runtime exit from the quantized region — identity under design B.

    The converted, inference-time form of a :class:`DeQuantStub`, produced by
    :func:`lucid.quantization.convert` / :meth:`from_float`. It marks where the
    quantized region ends and float compute resumes. Because Lucid's sidecar
    representation (design B) carries activations as (fake-quantized) ``float32``
    throughout — never as packed integers — there is nothing to unpack: the
    forward is a plain identity and the module holds no state.

    It is the passive twin of the active :class:`Quantize` entry marker. Keeping
    it in the graph (rather than deleting it) makes the quantized module tree
    symmetric, so tooling that walks entry/exit pairs sees a balanced boundary.

    Parameters
    ----------
    None
        The class takes no constructor arguments; instances are created by
        :meth:`from_float` from a :class:`DeQuantStub` (there is no state to
        carry over) or by calling ``DeQuantize()`` directly.

    Notes
    -----
    - Pure identity in every mode — it never changes a tensor's dtype, shape, or
      values. Its role is documentary / structural, not numerical.
    - Holds no buffers and no qparams; contrast with :class:`Quantize`, which
      carries the calibrated ``scale`` / ``zero_point``.
    - A model may legally omit the dequant marker; it is retained for graph
      symmetry and to signal where float compute resumes.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> dq = nn.quantized.DeQuantize()
    >>> x = lucid.randn(2, 3)
    >>> bool((dq(x) == x).all().item())      # identity — nothing to unpack
    True

    See Also
    --------
    lucid.nn.quantized.DeQuantStub : The float-model marker this converts from.
    lucid.nn.quantized.Quantize : The active entry marker it mirrors.
    lucid.quantization.convert : Installs this layer from a calibrated model.
    """

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary boundary marker
        """Identity (activations are already float-carried)."""
        return x

    @classmethod
    def from_float(cls, stub: nn.Module) -> DeQuantize:
        """Build from a :class:`DeQuantStub` (no state to carry)."""
        return cls()


class QuantWrapper(nn.Module):
    """Wrap an arbitrary float model with ``QuantStub`` / ``DeQuantStub``.

    A convenience container that bolts a quantization boundary pair onto a model
    that was never written with quantization in mind (e.g. a stock zoo model
    with no ``QuantStub`` of its own). The wrapper inserts a :class:`QuantStub`
    before the model and a :class:`DeQuantStub` after it, so ``prepare`` can
    attach an entry-side activation observer and ``convert`` can install a
    runtime :class:`Quantize` at the boundary — all without editing the wrapped
    module's ``forward``.

    The wrapped module is stored as the ``module`` submodule, flanked by
    ``quant`` and ``dequant``. Each forward runs ``quant → module → dequant``:
    the input is quantized on entry, the model runs (its own weighted layers get
    swapped to their quantized forms during ``convert``), and the output passes
    through the identity dequant marker on the way out.

    Parameters
    ----------
    module : nn.Module
        The float model to wrap. It is stored verbatim as ``self.module`` and is
        itself quantized in place when the wrapper is prepared and converted.

    Attributes
    ----------
    quant : QuantStub
        The entry-side boundary marker inserted before ``module``.
    module : nn.Module
        The wrapped float (later quantized) model.
    dequant : DeQuantStub
        The exit-side boundary marker inserted after ``module``.

    Notes
    -----
    - Only the entry boundary is numerically active after ``convert`` — the
      dequant side is an identity under the sidecar representation (design B).
    - The wrapped model still needs a ``qconfig`` (set it on the wrapper or the
      inner module) before ``prepare`` can attach observers.
    - Prefer explicit in-model ``QuantStub`` / ``DeQuantStub`` placement when you
      want the boundary somewhere other than the model's outermost input/output.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> net = nn.Sequential(nn.Linear(16, 8))
    >>> wrapped = nn.quantized.QuantWrapper(net)
    >>> wrapped.qconfig = Q.get_default_qconfig()
    >>> prepared = Q.prepare(wrapped)
    >>> _ = prepared(lucid.randn(4, 16))         # calibrate
    >>> qmodel = Q.convert(prepared)
    >>> type(qmodel.quant).__name__              # stub swapped to runtime form
    'Quantize'

    See Also
    --------
    lucid.nn.quantized.QuantStub : The entry marker it inserts.
    lucid.nn.quantized.DeQuantStub : The exit marker it inserts.
    lucid.quantization.prepare : Attaches observers to the wrapped boundaries.
    lucid.quantization.convert : Swaps the stubs for their runtime forms.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.quant = QuantStub()
        self.module = module
        self.dequant = DeQuantStub()

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary wrapper
        """Quantize the input, run the wrapped model, dequantize the output."""
        x = cast("Tensor", self.quant(x))
        x = cast("Tensor", self.module(x))
        return cast("Tensor", self.dequant(x))
