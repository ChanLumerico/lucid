"""Fused (*intrinsic*) float modules produced by ``fuse_modules``.

Fusing a ``Conv``/``Linear`` with the activation (and, for conv, the
BatchNorm folded into the weight) that follows it does two things for
quantization: it lets a **single** activation observer see the *post*-ReLU
output — so the quantized layer's output grid is chosen for the actual
(non-negative) inference range — and it removes the intermediate float
tensors.  Each intrinsic module is a thin :class:`~lucid.nn.Sequential`
subclass; ``convert`` maps it to the matching ``nn.quantized`` fused layer.
"""

import lucid.nn as nn


class _FusedModule(nn.Sequential):
    """Base tag for fused float modules (a :class:`~lucid.nn.Sequential`)."""


class ConvReLU1d(_FusedModule):
    r"""Fused float ``Conv1d`` + ``ReLU`` marker — a plain-float :class:`~lucid.nn.Sequential`.

    A thin :class:`~lucid.nn.Sequential` of ``[conv, relu]`` emitted by
    :func:`lucid.quantization.fuse_modules`. It runs as **plain float** — it carries no
    observers and no fake-quant, and computing through it is identical to running the conv
    then the ReLU. Its whole purpose is *structural*: by packaging the two ops as one unit,
    it tells the quantization pipeline to place a **single** activation observer *after* the
    ReLU rather than between the conv and the ReLU.

    **Why fuse for quantization.** If an observer sat on the raw conv output it would
    calibrate a symmetric range that includes values the ReLU is about to discard, wasting
    half the int8 codes on negatives that never reach inference. Observing the *post*-ReLU
    range instead lets the eventual quantized layer pick a grid over :math:`[0, \max]`,
    roughly doubling resolution. Fusion also drops the intermediate float tensor between
    conv and ReLU, so the converted kernel does conv → ReLU → requantize in one step.

    **Where it sits.** This marker is the hand-off point between float and quantized worlds.
    :func:`lucid.quantization.prepare_qat` replaces it with a trainable
    :class:`~lucid.nn.qat.ConvReLU1d` (weight + post-ReLU fake-quant under an STE), and
    :func:`lucid.quantization.convert` maps it — or the QAT layer it became — to a fused
    inference :class:`~lucid.nn.quantized.ConvReLU1d`.

    As a plain-float module it simply computes, with :math:`\star` the 1-D
    cross-correlation:

    .. math::

        y = \operatorname{ReLU}(W \star x + b)

    Downstream, once an observer is attached, that fused output is fake-quantized under a
    straight-through estimator — rounded in the forward pass, identity in the backward:

    .. math::

        \operatorname{fake\_quant}(t) = \bigl(
            \operatorname{clamp}(\operatorname{round}(t / s) + z,\ q_{\min},\ q_{\max})
            - z\bigr)\, s,
        \qquad
        \frac{\partial\, \operatorname{fake\_quant}(t)}{\partial t} = 1

    Parameters
    ----------
    conv : nn.Conv1d
        The 1-D convolution to run first; kept as the ``[0]`` child of the sequential.
    relu : nn.ReLU
        The rectifier applied to the convolution output; the ``[1]`` child.

    Notes
    -----
    - **Structural tag only.** It carries no quantization state — no ``weight_fake_quant``,
      no ``activation_post_process``. Those are attached later by
      :func:`lucid.quantization.prepare_qat` (or by :func:`lucid.quantization.prepare` for
      the post-training path).
    - **Post-ReLU observer placement.** The single reason to fuse is so the downstream
      activation observer sees the non-negative post-ReLU range.
    - **Both directions are wired for you.** :func:`lucid.quantization.fuse_modules` emits
      this marker *in*; :func:`lucid.quantization.prepare_qat` /
      :func:`lucid.quantization.convert` map it *out* to the QAT / quantized fused layer.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.intrinsic as nni
    >>> m = nn.Sequential(nn.Conv1d(3, 8, 3, padding=1), nn.ReLU())
    >>> fused = Q.fuse_modules(m, [["0", "1"]])
    >>> isinstance(fused[0], nni.ConvReLU1d)     # [conv, relu] collapsed to one marker
    True
    >>> fused(lucid.randn(2, 3, 8)).shape         # runs as plain float
    (2, 8, 8)

    See Also
    --------
    lucid.nn.qat.ConvReLU1d : The trainable QAT layer ``prepare_qat`` swaps this for.
    lucid.nn.quantized.ConvReLU1d : The fused int8 inference conv ``convert`` maps this to.
    lucid.quantization.fuse_modules : Emits this marker from ``[conv, relu]``.
    lucid.quantization.fuse_modules_qat : Fuses straight to the QAT layer.
    """

    def __init__(self, conv: nn.Conv1d, relu: nn.ReLU) -> None:
        super().__init__(conv, relu)


class ConvReLU2d(_FusedModule):
    r"""Fused float ``Conv2d`` + ``ReLU`` marker — a plain-float :class:`~lucid.nn.Sequential`.

    A thin :class:`~lucid.nn.Sequential` of ``[conv, relu]`` (with any BatchNorm already
    folded into the conv weight) emitted by :func:`lucid.quantization.fuse_modules` — the
    dominant fused pattern in quantized vision backbones. It runs as **plain float** — it
    carries no observers and no fake-quant, and computing through it is identical to running
    the conv then the ReLU. Its whole purpose is *structural*: by packaging the two ops as
    one unit it tells the quantization pipeline to place a **single** activation observer
    *after* the ReLU rather than between the conv and the ReLU.

    **Why fuse for quantization.** If an observer sat on the raw conv output it would
    calibrate a symmetric range that includes values the ReLU is about to discard, wasting
    half the int8 codes on negatives that never reach inference. Observing the *post*-ReLU
    range instead lets the eventual quantized layer pick a grid over :math:`[0, \max]`,
    roughly doubling resolution. Fusion also drops the intermediate float tensor between
    conv and ReLU, so the converted kernel does conv → ReLU → requantize in one step.

    **Where it sits.** This marker is the hand-off point between float and quantized worlds.
    :func:`lucid.quantization.prepare_qat` replaces it with a trainable
    :class:`~lucid.nn.qat.ConvReLU2d` (weight + post-ReLU fake-quant under an STE), and
    :func:`lucid.quantization.convert` maps it — or the QAT layer it became — to a fused
    inference :class:`~lucid.nn.quantized.ConvReLU2d`.

    As a plain-float module it simply computes, with :math:`\star` the 2-D
    cross-correlation:

    .. math::

        y = \operatorname{ReLU}(W \star x + b)

    Downstream, once an observer is attached, that fused output is fake-quantized under a
    straight-through estimator — rounded in the forward pass, identity in the backward:

    .. math::

        \operatorname{fake\_quant}(t) = \bigl(
            \operatorname{clamp}(\operatorname{round}(t / s) + z,\ q_{\min},\ q_{\max})
            - z\bigr)\, s,
        \qquad
        \frac{\partial\, \operatorname{fake\_quant}(t)}{\partial t} = 1

    Parameters
    ----------
    conv : nn.Conv2d
        The 2-D convolution to run first (BatchNorm already folded in); the ``[0]`` child.
    relu : nn.ReLU
        The rectifier applied to the convolution output; the ``[1]`` child.

    Notes
    -----
    - **Structural tag only.** It carries no quantization state — no ``weight_fake_quant``,
      no ``activation_post_process``. Those are attached later by
      :func:`lucid.quantization.prepare_qat` (or by :func:`lucid.quantization.prepare` for
      the post-training path).
    - **Post-ReLU observer placement.** The single reason to fuse is so the downstream
      activation observer sees the non-negative post-ReLU range.
    - **Both directions are wired for you.** :func:`lucid.quantization.fuse_modules` emits
      this marker *in*; :func:`lucid.quantization.prepare_qat` /
      :func:`lucid.quantization.convert` map it *out* to the QAT / quantized fused layer.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.intrinsic as nni
    >>> m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
    >>> fused = Q.fuse_modules(m, [["0", "1"]])
    >>> isinstance(fused[0], nni.ConvReLU2d)     # [conv, relu] collapsed to one marker
    True
    >>> fused(lucid.randn(2, 3, 8, 8)).shape       # runs as plain float
    (2, 8, 8, 8)

    See Also
    --------
    lucid.nn.qat.ConvReLU2d : The trainable QAT layer ``prepare_qat`` swaps this for.
    lucid.nn.quantized.ConvReLU2d : The fused int8 inference conv ``convert`` maps this to.
    lucid.quantization.fuse_modules : Emits this marker from ``[conv, relu]``.
    lucid.quantization.fuse_modules_qat : Fuses straight to the QAT layer.
    """

    def __init__(self, conv: nn.Conv2d, relu: nn.ReLU) -> None:
        super().__init__(conv, relu)


class ConvReLU3d(_FusedModule):
    r"""Fused float ``Conv3d`` + ``ReLU`` marker — a plain-float :class:`~lucid.nn.Sequential`.

    A thin :class:`~lucid.nn.Sequential` of ``[conv, relu]`` emitted by
    :func:`lucid.quantization.fuse_modules`. It runs as **plain float** — it carries no
    observers and no fake-quant, and computing through it is identical to running the conv
    then the ReLU. Its whole purpose is *structural*: by packaging the two ops as one unit
    it tells the quantization pipeline to place a **single** activation observer *after* the
    ReLU rather than between the conv and the ReLU.

    **Why fuse for quantization.** If an observer sat on the raw conv output it would
    calibrate a symmetric range that includes values the ReLU is about to discard, wasting
    half the int8 codes on negatives that never reach inference. Observing the *post*-ReLU
    range instead lets the eventual quantized layer pick a grid over :math:`[0, \max]`,
    roughly doubling resolution. Fusion also drops the intermediate float tensor between
    conv and ReLU, so the converted kernel does conv → ReLU → requantize in one step.

    **Where it sits.** This marker is the hand-off point between float and quantized worlds.
    :func:`lucid.quantization.prepare_qat` replaces it with a trainable
    :class:`~lucid.nn.qat.ConvReLU3d` (weight + post-ReLU fake-quant under an STE), and
    :func:`lucid.quantization.convert` maps it — or the QAT layer it became — to a fused
    inference :class:`~lucid.nn.quantized.ConvReLU3d`.

    As a plain-float module it simply computes, with :math:`\star` the 3-D
    cross-correlation:

    .. math::

        y = \operatorname{ReLU}(W \star x + b)

    Downstream, once an observer is attached, that fused output is fake-quantized under a
    straight-through estimator — rounded in the forward pass, identity in the backward:

    .. math::

        \operatorname{fake\_quant}(t) = \bigl(
            \operatorname{clamp}(\operatorname{round}(t / s) + z,\ q_{\min},\ q_{\max})
            - z\bigr)\, s,
        \qquad
        \frac{\partial\, \operatorname{fake\_quant}(t)}{\partial t} = 1

    Parameters
    ----------
    conv : nn.Conv3d
        The 3-D convolution to run first; kept as the ``[0]`` child of the sequential.
    relu : nn.ReLU
        The rectifier applied to the convolution output; the ``[1]`` child.

    Notes
    -----
    - **Structural tag only.** It carries no quantization state — no ``weight_fake_quant``,
      no ``activation_post_process``. Those are attached later by
      :func:`lucid.quantization.prepare_qat` (or by :func:`lucid.quantization.prepare` for
      the post-training path).
    - **Post-ReLU observer placement.** The single reason to fuse is so the downstream
      activation observer sees the non-negative post-ReLU range.
    - **Both directions are wired for you.** :func:`lucid.quantization.fuse_modules` emits
      this marker *in*; :func:`lucid.quantization.prepare_qat` /
      :func:`lucid.quantization.convert` map it *out* to the QAT / quantized fused layer.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.intrinsic as nni
    >>> m = nn.Sequential(nn.Conv3d(3, 8, 3, padding=1), nn.ReLU())
    >>> fused = Q.fuse_modules(m, [["0", "1"]])
    >>> isinstance(fused[0], nni.ConvReLU3d)     # [conv, relu] collapsed to one marker
    True
    >>> fused(lucid.randn(2, 3, 4, 4, 4)).shape    # runs as plain float
    (2, 8, 4, 4, 4)

    See Also
    --------
    lucid.nn.qat.ConvReLU3d : The trainable QAT layer ``prepare_qat`` swaps this for.
    lucid.nn.quantized.ConvReLU3d : The fused int8 inference conv ``convert`` maps this to.
    lucid.quantization.fuse_modules : Emits this marker from ``[conv, relu]``.
    lucid.quantization.fuse_modules_qat : Fuses straight to the QAT layer.
    """

    def __init__(self, conv: nn.Conv3d, relu: nn.ReLU) -> None:
        super().__init__(conv, relu)


class LinearReLU(_FusedModule):
    r"""Fused float ``Linear`` + ``ReLU`` marker — a plain-float :class:`~lucid.nn.Sequential`.

    A thin :class:`~lucid.nn.Sequential` of ``[linear, relu]`` emitted by
    :func:`lucid.quantization.fuse_modules`. It runs as **plain float** — it carries no
    observers and no fake-quant, and computing through it is identical to running the linear
    then the ReLU. Its whole purpose is *structural*: by packaging the two ops as one unit
    it tells the quantization pipeline to place a **single** activation observer *after* the
    ReLU rather than between the linear and the ReLU.

    **Why fuse for quantization.** If an observer sat on the raw linear output it would
    calibrate a symmetric range that includes values the ReLU is about to discard, wasting
    half the int8 codes on negatives that never reach inference. Observing the *post*-ReLU
    range instead lets the eventual quantized layer pick a grid over :math:`[0, \max]`,
    roughly doubling resolution. Fusion also drops the intermediate float tensor between
    linear and ReLU, so the converted kernel does matmul → ReLU → requantize in one step.

    **Where it sits.** This marker is the hand-off point between float and quantized worlds.
    :func:`lucid.quantization.prepare_qat` replaces it with a trainable
    :class:`~lucid.nn.qat.LinearReLU` (weight + post-ReLU fake-quant under an STE), and
    :func:`lucid.quantization.convert` maps it — or the QAT layer it became — to a fused
    inference :class:`~lucid.nn.quantized.LinearReLU`.

    As a plain-float module it simply computes:

    .. math::

        y = \operatorname{ReLU}(W x^{\top} + b)

    Downstream, once an observer is attached, that fused output is fake-quantized under a
    straight-through estimator — rounded in the forward pass, identity in the backward:

    .. math::

        \operatorname{fake\_quant}(t) = \bigl(
            \operatorname{clamp}(\operatorname{round}(t / s) + z,\ q_{\min},\ q_{\max})
            - z\bigr)\, s,
        \qquad
        \frac{\partial\, \operatorname{fake\_quant}(t)}{\partial t} = 1

    Parameters
    ----------
    linear : nn.Linear
        The linear (fully-connected) layer to run first; the ``[0]`` child of the
        sequential.
    relu : nn.ReLU
        The rectifier applied to the linear output; the ``[1]`` child.

    Notes
    -----
    - **Structural tag only.** It carries no quantization state — no ``weight_fake_quant``,
      no ``activation_post_process``. Those are attached later by
      :func:`lucid.quantization.prepare_qat` (or by :func:`lucid.quantization.prepare` for
      the post-training path).
    - **Post-ReLU observer placement.** The single reason to fuse is so the downstream
      activation observer sees the non-negative post-ReLU range.
    - **Both directions are wired for you.** :func:`lucid.quantization.fuse_modules` emits
      this marker *in*; :func:`lucid.quantization.prepare_qat` /
      :func:`lucid.quantization.convert` map it *out* to the QAT / quantized fused layer.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.intrinsic as nni
    >>> m = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
    >>> fused = Q.fuse_modules(m, [["0", "1"]])
    >>> isinstance(fused[0], nni.LinearReLU)     # [linear, relu] collapsed to one marker
    True
    >>> fused(lucid.randn(8, 32)).shape           # runs as plain float
    (8, 16)

    See Also
    --------
    lucid.nn.qat.LinearReLU : The trainable QAT layer ``prepare_qat`` swaps this for.
    lucid.nn.quantized.LinearReLU : The fused int8 inference layer ``convert`` maps this to.
    lucid.quantization.fuse_modules : Emits this marker from ``[linear, relu]``.
    lucid.quantization.fuse_modules_qat : Fuses straight to the QAT layer.
    """

    def __init__(self, linear: nn.Linear, relu: nn.ReLU) -> None:
        super().__init__(linear, relu)
