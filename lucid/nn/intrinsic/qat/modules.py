"""QAT fused Conv+BN(+ReLU) — BatchNorm folded into the conv during training.

The delicate QAT primitive: a trainable conv + BatchNorm that, on every forward,
folds BN's (running) affine into the conv weight, fake-quantizes the **folded**
weight, convolves, optionally applies ReLU, and fake-quantizes the output.
Gradients flow (via STE) to the conv weight and BN parameters, so the network
learns weights that survive the eventual folded-and-quantized inference.
:func:`convert` bakes the folded int8 weight into a quantized conv.

The fold is rank-generic (1d / 2d / 3d): the only rank-specific piece is the
per-output-channel reshape and the ``F.convNd`` call, both derived from the
conv weight's rank.
"""

from typing import TYPE_CHECKING, cast, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

    from lucid.quantization._fake_quantize import FakeQuantize
    from lucid.quantization.qconfig import QConfig

_CONV_FNS = (F.conv1d, F.conv2d, F.conv3d)


class _ConvBnNd(nn.Module):
    """Rank-generic fused Conv+BN(+ReLU) with fold-and-fake-quant (trainable)."""

    weight_fake_quant: FakeQuantize
    activation_post_process: FakeQuantize

    def __init__(
        self,
        conv: nn.Module,
        bn: nn.Module,
        relu: bool,
        qconfig: QConfig | None,
    ) -> None:
        super().__init__()
        if qconfig is None:
            raise ValueError(f"{type(self).__name__} requires a qconfig")
        self.conv = conv
        self.bn = bn
        self.relu = relu
        self.qconfig = qconfig
        self.weight_fake_quant = cast("FakeQuantize", qconfig.weight())
        self.activation_post_process = cast("FakeQuantize", qconfig.activation())

    def _fold(self) -> tuple[Tensor, Tensor]:
        """Return the BN-folded ``(weight, bias)`` for the conv (any rank)."""
        conv = cast("nn.Conv2d", self.conv)  # structural: all conv ranks share these
        bn = cast("nn.BatchNorm2d", self.bn)
        running_var = cast("Tensor", bn.running_var)
        running_mean = cast("Tensor", bn.running_mean)
        inv_std = (running_var + bn.eps).rsqrt()
        gamma = cast("Tensor", bn.weight) if bn.affine else lucid.ones_like(inv_std)
        scale = gamma * inv_std
        out_channels = conv.weight.shape[0]
        # (C, 1, …) with one trailing 1 per non-output weight axis → 1d/2d/3d.
        w = conv.weight * scale.reshape(
            (out_channels,) + (1,) * (len(conv.weight.shape) - 1)
        )
        conv_bias = conv.bias if conv.bias is not None else lucid.zeros_like(scale)
        bias = (conv_bias - running_mean) * scale
        if bn.affine:
            bias = bias + cast("Tensor", bn.bias)
        return w, bias

    def _conv(self, x: Tensor, w: Tensor, b: Tensor) -> Tensor:
        """Dispatch to ``F.conv{1,2,3}d`` by the conv weight's rank."""
        conv = cast("nn.Conv2d", self.conv)
        fn = _CONV_FNS[len(conv.weight.shape) - 3]
        # ``conv`` is cast to Conv2d structurally; at runtime its rank matches the
        # selected ``fn``, so the (int|tuple) stride/padding are valid for it.
        return fn(x, w, b, conv.stride, conv.padding, conv.dilation, conv.groups)  # type: ignore[arg-type]

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary layer
        """Fold BN → fake-quant folded weight → conv → (ReLU) → fake-quant output."""
        weight, bias = self._fold()
        w_q = cast("Tensor", self.weight_fake_quant(weight))
        y = self._conv(x, w_q, bias)
        if self.relu:
            y = F.relu(y)
        return cast("Tensor", self.activation_post_process(y))


class ConvBn1d(_ConvBnNd):
    r"""QAT fused ``Conv1d`` + ``BatchNorm1d`` — BN folded into the weight every forward.

    The training-time stand-in for a ``Conv1d`` immediately followed by a ``BatchNorm1d``.
    At inference the BN scale-and-shift can be absorbed into the conv weight and bias so the
    pair runs as a *single* convolution; a naive QAT layer that folded BN once up front,
    however, would freeze BN's running statistics and stop it training. This class instead
    **re-folds BN into the conv weight on every forward**, fake-quantizes the *folded*
    weight, convolves, and fake-quantizes the output — all differentiably — so BN keeps
    learning while the network already experiences the folded-and-quantized numerics it will
    see at inference.

    **Straight-through estimator (STE).** The weight fake-quant rounds the folded kernel in
    the forward pass but passes gradients *straight through* in the backward pass, so both
    the conv weight and the BN affine / running statistics receive clean gradients despite
    the non-differentiable ``round``. Folding inside the forward — rather than once, ahead
    of training — is what keeps BN trainable: the fake-quantized weight always reflects the
    *current* running statistics.

    **Where it sits.** :func:`lucid.quantization.prepare_qat` builds this from a fused
    conv+BN pattern. After training, :func:`lucid.quantization.convert` folds BN one final
    time, quantizes the folded weight to int8, and bakes it — together with the calibrated
    activation qparams — into an inference :class:`~lucid.nn.quantized.Conv1d`. No separate
    BN layer survives conversion; it lives entirely inside the conv weight and bias.

    The per-output-channel BN fold (:math:`c` indexes output channels) is

    .. math::

        w'_{c} = \gamma_c\,\frac{w_c}{\sqrt{\sigma_c^2 + \epsilon}},
        \qquad
        b'_{c} = \beta_c + \gamma_c\,\frac{b_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}

    with BN affine :math:`(\gamma, \beta)`, running statistics
    :math:`(\mu, \sigma^2)`, numerical floor :math:`\epsilon`, and conv weight / bias
    :math:`(w_c, b_c)`. When the conv has no bias (the usual fused case, :math:`b_c = 0`)
    the shift reduces to :math:`b'_c = \beta_c - \gamma_c\,\mu_c/\sqrt{\sigma_c^2+\epsilon}`.
    The folded weight is then fake-quantized under the STE before the convolution:

    .. math::

        y = \operatorname{fake\_quant}_{a}\!\bigl(
            \operatorname{fake\_quant}_{w}(w') \star x + b'\bigr),
        \qquad
        \frac{\partial\, \operatorname{fake\_quant}(t)}{\partial t} = 1

    where :math:`\star` is the 1-D cross-correlation and
    :math:`\operatorname{fake\_quant}(t) = (\operatorname{clamp}(\operatorname{round}(t/s)
    + z,\ q_{\min}, q_{\max}) - z)\, s`.

    Parameters
    ----------
    conv : nn.Conv1d
        The float 1-D convolution whose weight receives the folded BN affine.
    bn : nn.BatchNorm1d
        The batch-norm layer folded into ``conv``; its running stats and affine parameters
        keep training under the STE.
    qconfig : QConfig
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
        Required — constructing the layer without one raises ``ValueError``.

    Attributes
    ----------
    weight_fake_quant : FakeQuantize
        The weight observer + fake-quant from ``qconfig.weight()``; rounds the *folded*
        kernel each forward and tracks its range so ``convert`` can pick the int8 qparams.
    activation_post_process : FakeQuantize
        The output observer + fake-quant from ``qconfig.activation()``; calibrates the
        activation grid whose ``(scale, zero_point)`` ``convert`` later bakes in.

    Notes
    -----
    - **BN re-folded every forward.** The fold happens inside ``forward`` rather than once
      up front, so the BN parameters stay trainable and the fake-quantized weight always
      reflects the current running statistics.
    - **STE differentiability.** The folded weight is rounded forward but the gradient is
      passed through as the identity, so the conv weight *and* the BN parameters train.
    - **Both directions are wired for you.** :func:`lucid.quantization.prepare_qat` builds
      this fused layer *in*; :func:`lucid.quantization.convert` folds it *out* into the
      matching :class:`~lucid.nn.quantized.Conv1d` (BN absorbed, no BN layer left).

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.intrinsic.qat as nniqat
    >>> cbn = nniqat.ConvBn1d(
    ...     nn.Conv1d(3, 8, 3, padding=1, bias=False),
    ...     nn.BatchNorm1d(8),
    ...     qconfig=Q.get_default_qat_qconfig(),
    ... )
    >>> w, b = cbn._fold()                 # BN folded into (weight, bias)
    >>> w.shape, b.shape
    ((8, 3, 3), (8,))
    >>> loss = (cbn(lucid.randn(2, 3, 8)) ** 2).mean()
    >>> loss.backward()                    # STE trains conv weight AND BN params

    A ``qconfig`` is mandatory — constructing the layer without one raises:

    >>> nniqat.ConvBn1d(nn.Conv1d(3, 8, 3), nn.BatchNorm1d(8))
    Traceback (most recent call last):
        ...
    ValueError: ConvBn1d requires a qconfig

    See Also
    --------
    lucid.nn.quantized.Conv1d : The int8 inference conv ``convert`` bakes this into.
    lucid.nn.intrinsic.qat.ConvBnReLU1d : Same fold, with a fused ReLU on the output.
    lucid.quantization.prepare_qat : Builds this fused QAT layer from conv + BN.
    lucid.quantization.convert : Folds BN a final time and bakes the int8 conv.
    """

    def __init__(
        self, conv: nn.Module, bn: nn.Module, qconfig: QConfig | None = None
    ) -> None:
        super().__init__(conv, bn, False, qconfig)


class ConvBn2d(_ConvBnNd):
    r"""QAT fused ``Conv2d`` + ``BatchNorm2d`` — BN folded into the weight every forward.

    The training-time stand-in for a ``Conv2d`` immediately followed by a ``BatchNorm2d``
    — the single most common fused block in quantized vision backbones. At inference the
    BN scale-and-shift can be absorbed into the conv weight and bias so the pair runs as a
    *single* convolution; a naive QAT layer that folded BN once up front, however, would
    freeze BN's running statistics and stop it training. This class instead **re-folds BN
    into the conv weight on every forward**, fake-quantizes the *folded* weight, convolves,
    and fake-quantizes the output — all differentiably — so BN keeps learning while the
    network already experiences the folded-and-quantized numerics it will see at inference.

    **Straight-through estimator (STE).** The weight fake-quant rounds the folded kernel in
    the forward pass but passes gradients *straight through* in the backward pass, so both
    the conv weight and the BN affine / running statistics receive clean gradients despite
    the non-differentiable ``round``. Folding inside the forward — rather than once, ahead
    of training — is what keeps BN trainable: the fake-quantized weight always reflects the
    *current* running statistics.

    **Where it sits.** :func:`lucid.quantization.prepare_qat` builds this from a fused
    conv+BN pattern. After training, :func:`lucid.quantization.convert` folds BN one final
    time, quantizes the folded weight to int8, and bakes it — together with the calibrated
    activation qparams — into an inference :class:`~lucid.nn.quantized.Conv2d`. No separate
    BN layer survives conversion; it lives entirely inside the conv weight and bias.

    The per-output-channel BN fold (:math:`c` indexes output channels) is

    .. math::

        w'_{c} = \gamma_c\,\frac{w_c}{\sqrt{\sigma_c^2 + \epsilon}},
        \qquad
        b'_{c} = \beta_c + \gamma_c\,\frac{b_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}

    with BN affine :math:`(\gamma, \beta)`, running statistics
    :math:`(\mu, \sigma^2)`, numerical floor :math:`\epsilon`, and conv weight / bias
    :math:`(w_c, b_c)`. When the conv has no bias (the usual fused case, :math:`b_c = 0`)
    the shift reduces to :math:`b'_c = \beta_c - \gamma_c\,\mu_c/\sqrt{\sigma_c^2+\epsilon}`.
    The folded weight is then fake-quantized under the STE before the convolution:

    .. math::

        y = \operatorname{fake\_quant}_{a}\!\bigl(
            \operatorname{fake\_quant}_{w}(w') \star x + b'\bigr),
        \qquad
        \frac{\partial\, \operatorname{fake\_quant}(t)}{\partial t} = 1

    where :math:`\star` is the 2-D cross-correlation and
    :math:`\operatorname{fake\_quant}(t) = (\operatorname{clamp}(\operatorname{round}(t/s)
    + z,\ q_{\min}, q_{\max}) - z)\, s`.

    Parameters
    ----------
    conv : nn.Conv2d
        The float 2-D convolution whose weight receives the folded BN affine.
    bn : nn.BatchNorm2d
        The batch-norm layer folded into ``conv``; its running stats and affine parameters
        keep training under the STE.
    qconfig : QConfig
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
        Required — constructing the layer without one raises ``ValueError``.

    Attributes
    ----------
    weight_fake_quant : FakeQuantize
        The weight observer + fake-quant from ``qconfig.weight()``; rounds the *folded*
        kernel each forward and tracks its range so ``convert`` can pick the int8 qparams.
    activation_post_process : FakeQuantize
        The output observer + fake-quant from ``qconfig.activation()``; calibrates the
        activation grid whose ``(scale, zero_point)`` ``convert`` later bakes in.

    Notes
    -----
    - **BN re-folded every forward.** The fold happens inside ``forward`` rather than once
      up front, so the BN parameters stay trainable and the fake-quantized weight always
      reflects the current running statistics.
    - **STE differentiability.** The folded weight is rounded forward but the gradient is
      passed through as the identity, so the conv weight *and* the BN parameters train.
    - **Both directions are wired for you.** :func:`lucid.quantization.prepare_qat` builds
      this fused layer *in*; :func:`lucid.quantization.convert` folds it *out* into the
      matching :class:`~lucid.nn.quantized.Conv2d` (BN absorbed, no BN layer left).

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.intrinsic.qat as nniqat
    >>> cbn = nniqat.ConvBn2d(
    ...     nn.Conv2d(3, 8, 3, padding=1, bias=False),
    ...     nn.BatchNorm2d(8),
    ...     qconfig=Q.get_default_qat_qconfig(),
    ... )
    >>> w, b = cbn._fold()                 # BN folded into (weight, bias)
    >>> w.shape, b.shape
    ((8, 3, 3, 3), (8,))
    >>> loss = (cbn(lucid.randn(2, 3, 8, 8)) ** 2).mean()
    >>> loss.backward()                    # STE trains conv weight AND BN params

    A ``qconfig`` is mandatory — constructing the layer without one raises:

    >>> nniqat.ConvBn2d(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8))
    Traceback (most recent call last):
        ...
    ValueError: ConvBn2d requires a qconfig

    See Also
    --------
    lucid.nn.quantized.Conv2d : The int8 inference conv ``convert`` bakes this into.
    lucid.nn.intrinsic.qat.ConvBnReLU2d : Same fold, with a fused ReLU on the output.
    lucid.quantization.prepare_qat : Builds this fused QAT layer from conv + BN.
    lucid.quantization.convert : Folds BN a final time and bakes the int8 conv.
    """

    def __init__(
        self, conv: nn.Module, bn: nn.Module, qconfig: QConfig | None = None
    ) -> None:
        super().__init__(conv, bn, False, qconfig)


class ConvBn3d(_ConvBnNd):
    r"""QAT fused ``Conv3d`` + ``BatchNorm3d`` — BN folded into the weight every forward.

    The training-time stand-in for a ``Conv3d`` immediately followed by a ``BatchNorm3d``
    (volumetric / video backbones). At inference the BN scale-and-shift can be absorbed
    into the conv weight and bias so the pair runs as a *single* convolution; a naive QAT
    layer that folded BN once up front, however, would freeze BN's running statistics and
    stop it training. This class instead **re-folds BN into the conv weight on every
    forward**, fake-quantizes the *folded* weight, convolves, and fake-quantizes the output
    — all differentiably — so BN keeps learning while the network already experiences the
    folded-and-quantized numerics it will see at inference.

    **Straight-through estimator (STE).** The weight fake-quant rounds the folded kernel in
    the forward pass but passes gradients *straight through* in the backward pass, so both
    the conv weight and the BN affine / running statistics receive clean gradients despite
    the non-differentiable ``round``. Folding inside the forward — rather than once, ahead
    of training — is what keeps BN trainable: the fake-quantized weight always reflects the
    *current* running statistics.

    **Where it sits.** :func:`lucid.quantization.prepare_qat` builds this from a fused
    conv+BN pattern. After training, :func:`lucid.quantization.convert` folds BN one final
    time, quantizes the folded weight to int8, and bakes it — together with the calibrated
    activation qparams — into an inference :class:`~lucid.nn.quantized.Conv3d`. No separate
    BN layer survives conversion; it lives entirely inside the conv weight and bias.

    The per-output-channel BN fold (:math:`c` indexes output channels) is

    .. math::

        w'_{c} = \gamma_c\,\frac{w_c}{\sqrt{\sigma_c^2 + \epsilon}},
        \qquad
        b'_{c} = \beta_c + \gamma_c\,\frac{b_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}

    with BN affine :math:`(\gamma, \beta)`, running statistics
    :math:`(\mu, \sigma^2)`, numerical floor :math:`\epsilon`, and conv weight / bias
    :math:`(w_c, b_c)`. When the conv has no bias (the usual fused case, :math:`b_c = 0`)
    the shift reduces to :math:`b'_c = \beta_c - \gamma_c\,\mu_c/\sqrt{\sigma_c^2+\epsilon}`.
    The folded weight is then fake-quantized under the STE before the convolution:

    .. math::

        y = \operatorname{fake\_quant}_{a}\!\bigl(
            \operatorname{fake\_quant}_{w}(w') \star x + b'\bigr),
        \qquad
        \frac{\partial\, \operatorname{fake\_quant}(t)}{\partial t} = 1

    where :math:`\star` is the 3-D cross-correlation and
    :math:`\operatorname{fake\_quant}(t) = (\operatorname{clamp}(\operatorname{round}(t/s)
    + z,\ q_{\min}, q_{\max}) - z)\, s`.

    Parameters
    ----------
    conv : nn.Conv3d
        The float 3-D convolution whose weight receives the folded BN affine.
    bn : nn.BatchNorm3d
        The batch-norm layer folded into ``conv``; its running stats and affine parameters
        keep training under the STE.
    qconfig : QConfig
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
        Required — constructing the layer without one raises ``ValueError``.

    Attributes
    ----------
    weight_fake_quant : FakeQuantize
        The weight observer + fake-quant from ``qconfig.weight()``; rounds the *folded*
        kernel each forward and tracks its range so ``convert`` can pick the int8 qparams.
    activation_post_process : FakeQuantize
        The output observer + fake-quant from ``qconfig.activation()``; calibrates the
        activation grid whose ``(scale, zero_point)`` ``convert`` later bakes in.

    Notes
    -----
    - **BN re-folded every forward.** The fold happens inside ``forward`` rather than once
      up front, so the BN parameters stay trainable and the fake-quantized weight always
      reflects the current running statistics.
    - **STE differentiability.** The folded weight is rounded forward but the gradient is
      passed through as the identity, so the conv weight *and* the BN parameters train.
    - **Both directions are wired for you.** :func:`lucid.quantization.prepare_qat` builds
      this fused layer *in*; :func:`lucid.quantization.convert` folds it *out* into the
      matching :class:`~lucid.nn.quantized.Conv3d` (BN absorbed, no BN layer left).

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.intrinsic.qat as nniqat
    >>> cbn = nniqat.ConvBn3d(
    ...     nn.Conv3d(3, 8, 3, padding=1, bias=False),
    ...     nn.BatchNorm3d(8),
    ...     qconfig=Q.get_default_qat_qconfig(),
    ... )
    >>> w, b = cbn._fold()                 # BN folded into (weight, bias)
    >>> w.shape, b.shape
    ((8, 3, 3, 3, 3), (8,))
    >>> loss = (cbn(lucid.randn(2, 3, 4, 4, 4)) ** 2).mean()
    >>> loss.backward()                    # STE trains conv weight AND BN params

    A ``qconfig`` is mandatory — constructing the layer without one raises:

    >>> nniqat.ConvBn3d(nn.Conv3d(3, 8, 3), nn.BatchNorm3d(8))
    Traceback (most recent call last):
        ...
    ValueError: ConvBn3d requires a qconfig

    See Also
    --------
    lucid.nn.quantized.Conv3d : The int8 inference conv ``convert`` bakes this into.
    lucid.nn.intrinsic.qat.ConvBnReLU3d : Same fold, with a fused ReLU on the output.
    lucid.quantization.prepare_qat : Builds this fused QAT layer from conv + BN.
    lucid.quantization.convert : Folds BN a final time and bakes the int8 conv.
    """

    def __init__(
        self, conv: nn.Module, bn: nn.Module, qconfig: QConfig | None = None
    ) -> None:
        super().__init__(conv, bn, False, qconfig)


class ConvBnReLU1d(_ConvBnNd):
    r"""QAT fused ``Conv1d`` + ``BatchNorm1d`` + ``ReLU`` — BN folded every forward.

    Like :class:`ConvBn1d`, but a ReLU follows the (BN-folded) convolution and the *output*
    fake-quant observes the **post**-ReLU range, so the calibrated activation grid covers
    only the non-negative values inference can actually produce. This is the fully-fused
    conv→BN→ReLU block that becomes a single quantized kernel after conversion.

    As in :class:`ConvBn1d`, BN is **re-folded into the conv weight on every forward** and
    the folded weight is fake-quantized under a straight-through estimator (STE): rounding
    happens forward, gradients pass straight through, so the conv weight *and* the BN affine
    / running statistics keep training. Built by :func:`lucid.quantization.prepare_qat`;
    :func:`lucid.quantization.convert` folds BN a final time, quantizes the folded weight to
    int8, and bakes it into a fused inference :class:`~lucid.nn.quantized.ConvReLU1d`.

    The per-output-channel BN fold (:math:`c` indexes output channels) is

    .. math::

        w'_{c} = \gamma_c\,\frac{w_c}{\sqrt{\sigma_c^2 + \epsilon}},
        \qquad
        b'_{c} = \beta_c + \gamma_c\,\frac{b_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}

    with BN affine :math:`(\gamma, \beta)`, running statistics
    :math:`(\mu, \sigma^2)`, numerical floor :math:`\epsilon`, and conv weight / bias
    :math:`(w_c, b_c)` (reducing to
    :math:`b'_c = \beta_c - \gamma_c\,\mu_c/\sqrt{\sigma_c^2+\epsilon}` for a bias-free
    conv). The folded weight is fake-quantized, convolved, passed through ReLU, then the
    output is fake-quantized on the observed post-ReLU grid:

    .. math::

        y = \operatorname{fake\_quant}_{a}\!\Bigl(
            \operatorname{ReLU}\bigl(
                \operatorname{fake\_quant}_{w}(w') \star x + b'\bigr)\Bigr),
        \qquad
        \frac{\partial\, \operatorname{fake\_quant}(t)}{\partial t} = 1

    where :math:`\star` is the 1-D cross-correlation and
    :math:`\operatorname{fake\_quant}(t) = (\operatorname{clamp}(\operatorname{round}(t/s)
    + z,\ q_{\min}, q_{\max}) - z)\, s`.

    Parameters
    ----------
    conv : nn.Conv1d
        The float 1-D convolution whose weight receives the folded BN affine.
    bn : nn.BatchNorm1d
        The batch-norm layer folded into ``conv``; its running stats and affine parameters
        keep training under the STE.
    relu : bool, default=True
        Whether the fused ReLU is applied before the output fake-quant.
    qconfig : QConfig
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
        Required — constructing the layer without one raises ``ValueError``.

    Attributes
    ----------
    weight_fake_quant : FakeQuantize
        The weight observer + fake-quant from ``qconfig.weight()``; rounds the *folded*
        kernel each forward and tracks its range so ``convert`` can pick the int8 qparams.
    activation_post_process : FakeQuantize
        The output observer + fake-quant from ``qconfig.activation()``; here it observes
        the **post-ReLU** range so the baked-in grid matches inference.

    Notes
    -----
    - **BN re-folded every forward.** The fold happens inside ``forward`` rather than once
      up front, so the BN parameters stay trainable and the fake-quantized weight always
      reflects the current running statistics.
    - **Post-ReLU calibration.** The activation observer sits after the ReLU, so its grid
      covers only :math:`[0, \max]`, roughly doubling resolution versus a symmetric range.
    - **STE differentiability.** The folded weight is rounded forward but the gradient is
      passed through as the identity, so the conv weight *and* the BN parameters train.
    - **Both directions are wired for you.** :func:`lucid.quantization.prepare_qat` builds
      this fused layer *in*; :func:`lucid.quantization.convert` folds it *out* into the
      matching :class:`~lucid.nn.quantized.ConvReLU1d`.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.intrinsic.qat as nniqat
    >>> cbr = nniqat.ConvBnReLU1d(
    ...     nn.Conv1d(3, 8, 3, padding=1, bias=False),
    ...     nn.BatchNorm1d(8),
    ...     relu=True,
    ...     qconfig=Q.get_default_qat_qconfig(),
    ... )
    >>> loss = (cbr(lucid.randn(2, 3, 8)) ** 2).mean()
    >>> loss.backward()                    # STE trains conv weight AND BN params
    >>> cbr.eval()
    >>> from lucid.nn.intrinsic.qat.modules import convbnrelu2d_to_quantized
    >>> qc = convbnrelu2d_to_quantized(cbr)     # rank-generic bake
    >>> type(qc).__name__
    'ConvReLU1d'

    See Also
    --------
    lucid.nn.quantized.ConvReLU1d : The fused int8 inference conv ``convert`` bakes into.
    lucid.nn.intrinsic.qat.ConvBn1d : Same fold, without the fused ReLU.
    lucid.quantization.prepare_qat : Builds this fused QAT layer from conv + BN + ReLU.
    lucid.quantization.convert : Folds BN a final time and bakes the fused int8 conv.
    """

    def __init__(
        self,
        conv: nn.Module,
        bn: nn.Module,
        relu: bool = True,
        qconfig: QConfig | None = None,
    ) -> None:
        super().__init__(conv, bn, relu, qconfig)


class ConvBnReLU2d(_ConvBnNd):
    r"""QAT fused ``Conv2d`` + ``BatchNorm2d`` + ``ReLU`` — BN folded every forward.

    Like :class:`ConvBn2d`, but a ReLU follows the (BN-folded) convolution and the *output*
    fake-quant observes the **post**-ReLU range, so the calibrated activation grid covers
    only the non-negative values inference can actually produce. This conv→BN→ReLU triple
    is *the* canonical fused block of quantized vision backbones and becomes a single
    quantized kernel after conversion.

    As in :class:`ConvBn2d`, BN is **re-folded into the conv weight on every forward** and
    the folded weight is fake-quantized under a straight-through estimator (STE): rounding
    happens forward, gradients pass straight through, so the conv weight *and* the BN affine
    / running statistics keep training. Built by :func:`lucid.quantization.prepare_qat`;
    :func:`lucid.quantization.convert` folds BN a final time, quantizes the folded weight to
    int8, and bakes it into a fused inference :class:`~lucid.nn.quantized.ConvReLU2d`.

    The per-output-channel BN fold (:math:`c` indexes output channels) is

    .. math::

        w'_{c} = \gamma_c\,\frac{w_c}{\sqrt{\sigma_c^2 + \epsilon}},
        \qquad
        b'_{c} = \beta_c + \gamma_c\,\frac{b_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}

    with BN affine :math:`(\gamma, \beta)`, running statistics
    :math:`(\mu, \sigma^2)`, numerical floor :math:`\epsilon`, and conv weight / bias
    :math:`(w_c, b_c)` (reducing to
    :math:`b'_c = \beta_c - \gamma_c\,\mu_c/\sqrt{\sigma_c^2+\epsilon}` for a bias-free
    conv). The folded weight is fake-quantized, convolved, passed through ReLU, then the
    output is fake-quantized on the observed post-ReLU grid:

    .. math::

        y = \operatorname{fake\_quant}_{a}\!\Bigl(
            \operatorname{ReLU}\bigl(
                \operatorname{fake\_quant}_{w}(w') \star x + b'\bigr)\Bigr),
        \qquad
        \frac{\partial\, \operatorname{fake\_quant}(t)}{\partial t} = 1

    where :math:`\star` is the 2-D cross-correlation and
    :math:`\operatorname{fake\_quant}(t) = (\operatorname{clamp}(\operatorname{round}(t/s)
    + z,\ q_{\min}, q_{\max}) - z)\, s`.

    Parameters
    ----------
    conv : nn.Conv2d
        The float 2-D convolution whose weight receives the folded BN affine.
    bn : nn.BatchNorm2d
        The batch-norm layer folded into ``conv``; its running stats and affine parameters
        keep training under the STE.
    relu : bool, default=True
        Whether the fused ReLU is applied before the output fake-quant.
    qconfig : QConfig
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
        Required — constructing the layer without one raises ``ValueError``.

    Attributes
    ----------
    weight_fake_quant : FakeQuantize
        The weight observer + fake-quant from ``qconfig.weight()``; rounds the *folded*
        kernel each forward and tracks its range so ``convert`` can pick the int8 qparams.
    activation_post_process : FakeQuantize
        The output observer + fake-quant from ``qconfig.activation()``; here it observes
        the **post-ReLU** range so the baked-in grid matches inference.

    Notes
    -----
    - **BN re-folded every forward.** The fold happens inside ``forward`` rather than once
      up front, so the BN parameters stay trainable and the fake-quantized weight always
      reflects the current running statistics.
    - **Post-ReLU calibration.** The activation observer sits after the ReLU, so its grid
      covers only :math:`[0, \max]`, roughly doubling resolution versus a symmetric range.
    - **STE differentiability.** The folded weight is rounded forward but the gradient is
      passed through as the identity, so the conv weight *and* the BN parameters train.
    - **Both directions are wired for you.** :func:`lucid.quantization.prepare_qat` builds
      this fused layer *in*; :func:`lucid.quantization.convert` folds it *out* into the
      matching :class:`~lucid.nn.quantized.ConvReLU2d`.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.intrinsic.qat as nniqat
    >>> cbr = nniqat.ConvBnReLU2d(
    ...     nn.Conv2d(3, 8, 3, padding=1, bias=False),
    ...     nn.BatchNorm2d(8),
    ...     relu=True,
    ...     qconfig=Q.get_default_qat_qconfig(),
    ... )
    >>> loss = (cbr(lucid.randn(2, 3, 8, 8)) ** 2).mean()
    >>> loss.backward()                    # STE trains conv weight AND BN params
    >>> cbr.eval()
    >>> from lucid.nn.intrinsic.qat.modules import convbnrelu2d_to_quantized
    >>> qc = convbnrelu2d_to_quantized(cbr)     # BN folded + baked to int8
    >>> type(qc).__name__
    'ConvReLU2d'

    See Also
    --------
    lucid.nn.quantized.ConvReLU2d : The fused int8 inference conv ``convert`` bakes into.
    lucid.nn.intrinsic.qat.ConvBn2d : Same fold, without the fused ReLU.
    lucid.quantization.prepare_qat : Builds this fused QAT layer from conv + BN + ReLU.
    lucid.quantization.convert : Folds BN a final time and bakes the fused int8 conv.
    """

    def __init__(
        self,
        conv: nn.Module,
        bn: nn.Module,
        relu: bool = True,
        qconfig: QConfig | None = None,
    ) -> None:
        super().__init__(conv, bn, relu, qconfig)


class ConvBnReLU3d(_ConvBnNd):
    r"""QAT fused ``Conv3d`` + ``BatchNorm3d`` + ``ReLU`` — BN folded every forward.

    Like :class:`ConvBn3d`, but a ReLU follows the (BN-folded) convolution and the *output*
    fake-quant observes the **post**-ReLU range, so the calibrated activation grid covers
    only the non-negative values inference can actually produce. This is the fully-fused
    conv→BN→ReLU block for volumetric / video backbones and becomes a single quantized
    kernel after conversion.

    As in :class:`ConvBn3d`, BN is **re-folded into the conv weight on every forward** and
    the folded weight is fake-quantized under a straight-through estimator (STE): rounding
    happens forward, gradients pass straight through, so the conv weight *and* the BN affine
    / running statistics keep training. Built by :func:`lucid.quantization.prepare_qat`;
    :func:`lucid.quantization.convert` folds BN a final time, quantizes the folded weight to
    int8, and bakes it into a fused inference :class:`~lucid.nn.quantized.ConvReLU3d`.

    The per-output-channel BN fold (:math:`c` indexes output channels) is

    .. math::

        w'_{c} = \gamma_c\,\frac{w_c}{\sqrt{\sigma_c^2 + \epsilon}},
        \qquad
        b'_{c} = \beta_c + \gamma_c\,\frac{b_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}

    with BN affine :math:`(\gamma, \beta)`, running statistics
    :math:`(\mu, \sigma^2)`, numerical floor :math:`\epsilon`, and conv weight / bias
    :math:`(w_c, b_c)` (reducing to
    :math:`b'_c = \beta_c - \gamma_c\,\mu_c/\sqrt{\sigma_c^2+\epsilon}` for a bias-free
    conv). The folded weight is fake-quantized, convolved, passed through ReLU, then the
    output is fake-quantized on the observed post-ReLU grid:

    .. math::

        y = \operatorname{fake\_quant}_{a}\!\Bigl(
            \operatorname{ReLU}\bigl(
                \operatorname{fake\_quant}_{w}(w') \star x + b'\bigr)\Bigr),
        \qquad
        \frac{\partial\, \operatorname{fake\_quant}(t)}{\partial t} = 1

    where :math:`\star` is the 3-D cross-correlation and
    :math:`\operatorname{fake\_quant}(t) = (\operatorname{clamp}(\operatorname{round}(t/s)
    + z,\ q_{\min}, q_{\max}) - z)\, s`.

    Parameters
    ----------
    conv : nn.Conv3d
        The float 3-D convolution whose weight receives the folded BN affine.
    bn : nn.BatchNorm3d
        The batch-norm layer folded into ``conv``; its running stats and affine parameters
        keep training under the STE.
    relu : bool, default=True
        Whether the fused ReLU is applied before the output fake-quant.
    qconfig : QConfig
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
        Required — constructing the layer without one raises ``ValueError``.

    Attributes
    ----------
    weight_fake_quant : FakeQuantize
        The weight observer + fake-quant from ``qconfig.weight()``; rounds the *folded*
        kernel each forward and tracks its range so ``convert`` can pick the int8 qparams.
    activation_post_process : FakeQuantize
        The output observer + fake-quant from ``qconfig.activation()``; here it observes
        the **post-ReLU** range so the baked-in grid matches inference.

    Notes
    -----
    - **BN re-folded every forward.** The fold happens inside ``forward`` rather than once
      up front, so the BN parameters stay trainable and the fake-quantized weight always
      reflects the current running statistics.
    - **Post-ReLU calibration.** The activation observer sits after the ReLU, so its grid
      covers only :math:`[0, \max]`, roughly doubling resolution versus a symmetric range.
    - **STE differentiability.** The folded weight is rounded forward but the gradient is
      passed through as the identity, so the conv weight *and* the BN parameters train.
    - **Both directions are wired for you.** :func:`lucid.quantization.prepare_qat` builds
      this fused layer *in*; :func:`lucid.quantization.convert` folds it *out* into the
      matching :class:`~lucid.nn.quantized.ConvReLU3d`.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.intrinsic.qat as nniqat
    >>> cbr = nniqat.ConvBnReLU3d(
    ...     nn.Conv3d(3, 8, 3, padding=1, bias=False),
    ...     nn.BatchNorm3d(8),
    ...     relu=True,
    ...     qconfig=Q.get_default_qat_qconfig(),
    ... )
    >>> loss = (cbr(lucid.randn(2, 3, 4, 4, 4)) ** 2).mean()
    >>> loss.backward()                    # STE trains conv weight AND BN params
    >>> cbr.eval()
    >>> from lucid.nn.intrinsic.qat.modules import convbnrelu2d_to_quantized
    >>> qc = convbnrelu2d_to_quantized(cbr)     # rank-generic bake
    >>> type(qc).__name__
    'ConvReLU3d'

    See Also
    --------
    lucid.nn.quantized.ConvReLU3d : The fused int8 inference conv ``convert`` bakes into.
    lucid.nn.intrinsic.qat.ConvBn3d : Same fold, without the fused ReLU.
    lucid.quantization.prepare_qat : Builds this fused QAT layer from conv + BN + ReLU.
    lucid.quantization.convert : Folds BN a final time and bakes the fused int8 conv.
    """

    def __init__(
        self,
        conv: nn.Module,
        bn: nn.Module,
        relu: bool = True,
        qconfig: QConfig | None = None,
    ) -> None:
        super().__init__(conv, bn, relu, qconfig)


def _convbn_to_quantized(mod: nn.Module) -> nn.Module:
    """Bake a trained fused Conv+BN(+ReLU) into a quantized inference conv (any rank)."""
    import lucid.nn.quantized as nnq
    from lucid.quantization._functional import quantize

    cbr = cast("_ConvBnNd", mod)
    weight, bias = cbr._fold()
    wfq = cbr.weight_fake_quant
    wfq(weight)  # observe the folded weight
    w_scale, w_zp = wfq.calculate_qparams()
    ch_axis = wfq.ch_axis if wfq.ch_axis is not None else 0
    codes = quantize(weight, w_scale, w_zp, wfq.qdtype, ch_axis=wfq.ch_axis)

    conv = cast("nn.Conv2d", cbr.conv)
    rank = len(conv.weight.shape) - 2  # 1 / 2 / 3
    plain = (nnq.Conv1d, nnq.Conv2d, nnq.Conv3d)[rank - 1]
    fused = (nnq.ConvReLU1d, nnq.ConvReLU2d, nnq.ConvReLU3d)[rank - 1]
    q_cls = fused if cbr.relu else plain
    q = q_cls(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True,
    )
    q.register_buffer("weight_int8", codes)
    q.register_buffer("weight_scale", w_scale)
    q.register_buffer("weight_zero_point", w_zp)
    q.weight_ch_axis = ch_axis
    q.register_buffer("bias", bias)
    a_scale, a_zp = cbr.activation_post_process.calculate_qparams()
    q.register_buffer("scale", a_scale)
    q.register_buffer("zero_point", a_zp)
    q.out_qdtype = cbr.activation_post_process.qdtype
    return q


# Back-compat alias — the original 2d-only name (still imported by convert).
convbnrelu2d_to_quantized = _convbn_to_quantized
