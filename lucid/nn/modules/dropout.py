"""
Dropout modules.
"""

from lucid._tensor.tensor import Tensor
from lucid.nn.module import Module
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.nn.functional.dropout import dropout, dropout2d, feature_alpha_dropout


def _check_dropout_prob(p: float) -> None:
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability must be in [0, 1], got {p!r}")


class Dropout(Module):
    r"""Randomly zero individual tensor elements during training (inverted dropout).

    During training, each scalar element of the input is independently
    set to zero with probability :math:`p`.  The remaining elements are
    rescaled by :math:`\frac{1}{1-p}` so that the expected value of
    every element is preserved — this is the *inverted* dropout
    convention, which means **no rescaling is needed at inference time**:

    .. math::

        y_i =
        \begin{cases}
            \dfrac{x_i}{1 - p} & \text{with probability } 1 - p \\[6pt]
            0                  & \text{with probability } p
        \end{cases}
        \quad \text{(training)}

    .. math::

        y_i = x_i \quad \text{(eval)}

    In eval mode the layer is the identity and the ``p`` parameter has
    no effect.

    **Why dropout works.** By randomly disabling units, dropout prevents
    co-adaptation — individual neurons cannot rely on the presence of
    specific peers, so they are forced to learn more robust features.
    Dropout is approximately equivalent to averaging the predictions of
    an ensemble of :math:`2^n` sub-networks (one per binary mask).

    Parameters
    ----------
    p : float, optional
        Probability of zeroing each element.  Must be in ``[0, 1]``.
        ``p=0`` disables dropout; ``p=1`` zeros the entire tensor.
        Default: ``0.5``.
    inplace : bool, optional
        If ``True``, modify the input tensor in place.  Use with care
        when the input participates in the autograd graph.
        Default: ``False``.

    Shape
    -----
    * **Input**: any shape ``(*)``.
    * **Output**: same shape ``(*)``.

    Notes
    -----
    Dropout should only be applied during training.  Call
    ``model.eval()`` before inference to switch all dropout layers to
    pass-through mode; call ``model.train()`` to re-enable them.

    For convolutional feature maps where adjacent spatial positions are
    highly correlated, per-element dropout is ineffective — consider
    :class:`Dropout2d` instead.

    Examples
    --------
    Basic usage in a linear classifier head:

    >>> import lucid, lucid.nn as nn
    >>> drop = nn.Dropout(p=0.3)
    >>> drop.train()
    >>> x = lucid.ones(4, 8)
    >>> y = drop(x)
    >>> # Approximately 30 % of elements are zero; rest scaled by 1/0.7
    >>> y.shape
    (4, 8)

    Disabled in eval mode:

    >>> drop.eval()
    >>> y_eval = drop(lucid.ones(4, 8))
    >>> # All elements equal 1.0 — no masking
    >>> float(y_eval.sum()) == 32.0
    True

    See Also
    --------
    Dropout1d : Channel-wise dropout for 3-D inputs.
    Dropout2d : Channel-wise dropout for 4-D (image) inputs.
    Dropout3d : Channel-wise dropout for 5-D (volumetric) inputs.
    AlphaDropout : Dropout variant that preserves SELU statistics.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        _check_dropout_prob(p)
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        return dropout(x, self.p, self.training, self.inplace)

    def extra_repr(self) -> str:
        s: str = f"p={self.p}"
        if self.inplace:
            s += ", inplace=True"
        return s


class Dropout1d(Module):
    r"""Randomly zero entire channels during training for 3-D inputs.

    Extends the dropout idea from individual elements to entire **channels**
    (feature maps) for sequence data of shape ``(N, C, L)``.  A single
    Bernoulli draw per channel per sample determines whether that channel
    is zeroed across its entire length:

    .. math::

        y_{n,c,:} =
        \begin{cases}
            \dfrac{x_{n,c,:}}{1 - p} & \text{with probability } 1 - p \\[6pt]
            \mathbf{0}               & \text{with probability } p
        \end{cases}
        \quad \text{(training)}

    This is more aggressive than element-wise :class:`Dropout` and is
    appropriate when neighbouring positions along the ``L`` axis are highly
    correlated (as they are in the output of a 1-D convolution), so that
    zeroing a single element would not provide enough regularisation signal.

    Parameters
    ----------
    p : float, optional
        Probability of zeroing an entire channel.  Must be in ``[0, 1]``.
        Default: ``0.5``.
    inplace : bool, optional
        If ``True``, modify the input in place.  Default: ``False``.

    Shape
    -----
    * **Input**: ``(N, C, L)`` — batch of 1-D feature maps.
    * **Output**: ``(N, C, L)`` — same shape; zeroed channels are
      uniformly zero along the entire ``L`` dimension.

    Notes
    -----
    The mask is sampled along the ``(N, C)`` axes only, then broadcast
    over ``L``.  The forward pass delegates to the same engine kernel
    as :class:`Dropout2d`.

    Examples
    --------
    Channel-wise dropout on a batch of 1-D feature maps:

    >>> import lucid, lucid.nn as nn
    >>> drop1d = nn.Dropout1d(p=0.2)
    >>> drop1d.train()
    >>> x = lucid.ones(2, 8, 16)    # (N=2, C=8, L=16)
    >>> y = drop1d(x)
    >>> y.shape
    (2, 8, 16)
    >>> # Each of the 8 channels is either all-zero or all-scaled

    Verify pass-through in eval mode:

    >>> drop1d.eval()
    >>> y_eval = drop1d(lucid.ones(2, 8, 16))
    >>> float(y_eval.sum()) == 256.0
    True

    See Also
    --------
    Dropout : Element-wise dropout.
    Dropout2d : Channel-wise dropout for 4-D (spatial) inputs.
    Dropout3d : Channel-wise dropout for 5-D (volumetric) inputs.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        _check_dropout_prob(p)
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        # The engine's ``dropoutnd`` kernel handles 3-D / 4-D / 5-D inputs by
        # building the mask along the channel axis, so the same call works
        # here as for ``Dropout2d``.
        return dropout2d(x, self.p, self.training)

    def extra_repr(self) -> str:
        s: str = f"p={self.p}"
        if self.inplace:
            s += ", inplace=True"
        return s


class Dropout2d(Module):
    r"""Randomly zero entire feature-map channels during training for 4-D inputs.

    For image-like inputs of shape ``(N, C, H, W)`` a single Bernoulli
    draw per channel per sample determines whether that entire spatial
    map is zeroed:

    .. math::

        y_{n,c,:,:} =
        \begin{cases}
            \dfrac{x_{n,c,:,:}}{1 - p} & \text{with probability } 1 - p \\[6pt]
            \mathbf{0}                  & \text{with probability } p
        \end{cases}
        \quad \text{(training)}

    Because adjacent pixels in a convolutional feature map are strongly
    correlated, zeroing individual pixels (as standard :class:`Dropout`
    does) has little regularisation effect.  Zeroing the entire channel
    forces the network to not rely on any single feature map.

    Parameters
    ----------
    p : float, optional
        Probability of zeroing an entire channel.  Must be in ``[0, 1]``.
        Default: ``0.5``.
    inplace : bool, optional
        If ``True``, modify the input in place.  Default: ``False``.

    Shape
    -----
    * **Input**: ``(N, C, H, W)`` — batch of 2-D feature maps.
    * **Output**: ``(N, C, H, W)`` — same shape; zeroed channels are
      zero across the full ``H × W`` spatial extent.

    Notes
    -----
    The Bernoulli mask is sampled on the ``(N, C)`` axes and broadcast
    over ``(H, W)``.  Spatial structure within a channel is therefore
    fully preserved — only the decision of *which* channels survive
    varies.

    Examples
    --------
    Typical use after a convolutional layer:

    >>> import lucid, lucid.nn as nn
    >>> drop2d = nn.Dropout2d(p=0.25)
    >>> drop2d.train()
    >>> x = lucid.ones(2, 16, 8, 8)    # (N=2, C=16, H=8, W=8)
    >>> y = drop2d(x)
    >>> y.shape
    (2, 16, 8, 8)
    >>> # Roughly 25 % of the 16 channels are entirely zero per sample

    No-op in eval mode:

    >>> drop2d.eval()
    >>> y_eval = drop2d(lucid.ones(2, 4, 4, 4))
    >>> float(y_eval.sum()) == 128.0
    True

    See Also
    --------
    Dropout : Element-wise scalar dropout.
    Dropout1d : Channel-wise dropout for 3-D (sequence) inputs.
    Dropout3d : Channel-wise dropout for 5-D (volumetric) inputs.
    FeatureAlphaDropout : Channel-wise variant that preserves SELU statistics.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        _check_dropout_prob(p)
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        return dropout2d(x, self.p, self.training)

    def extra_repr(self) -> str:
        s: str = f"p={self.p}"
        if self.inplace:
            s += ", inplace=True"
        return s


class AlphaDropout(Module):
    r"""Alpha dropout — element-wise dropout that preserves SELU self-normalisation.

    Standard dropout breaks the zero-mean / unit-variance property of
    `SELU <https://arxiv.org/abs/1706.02515>`_ activations because it
    sets units to zero, shifting the mean.  Alpha dropout fixes this by
    replacing dropped units with a learned negative saturation value
    :math:`\alpha'` and then applying an affine correction:

    .. math::

        y =
        \begin{cases}
            x & \text{with probability } 1 - p \\
            \alpha' & \text{with probability } p
        \end{cases}

        \tilde{y} = a \cdot y + b

    where :math:`\alpha' = -\lambda\alpha` with
    :math:`\lambda \approx 1.0507` and :math:`\alpha \approx 1.6733`
    (the SELU fixed-point constants), and the affine coefficients
    :math:`a, b` are chosen so that :math:`\mathbb{E}[\tilde{y}] = 0`
    and :math:`\operatorname{Var}[\tilde{y}] = 1` after the mask is
    applied.  The result is that the output distribution of each
    alpha-dropout layer is approximately standard normal, preserving the
    self-normalising property that makes deep SELU networks trainable
    without batch normalisation.

    Parameters
    ----------
    p : float, optional
        Probability of replacing an element with :math:`\alpha'`.
        Must be in ``[0, 1]``.  Default: ``0.5``.
    inplace : bool, optional
        Currently accepted for API compatibility but has no effect
        (the affine correction always produces a new tensor).
        Default: ``False``.

    Shape
    -----
    * **Input**: any shape ``(*)``.
    * **Output**: same shape ``(*)``.

    Notes
    -----
    Alpha dropout should be used **exclusively** with :class:`lucid.nn.SELU`
    activations.  Using it after other activations (ReLU, tanh, etc.)
    will not preserve any statistical invariant and is likely harmful.

    In eval mode the layer is the identity (no masking, no affine
    correction).

    Examples
    --------
    In a self-normalising MLP (SELU + AlphaDropout):

    >>> import lucid, lucid.nn as nn
    >>> mlp = nn.Sequential(
    ...     nn.Linear(32, 64),
    ...     nn.SELU(),
    ...     nn.AlphaDropout(p=0.05),
    ...     nn.Linear(64, 10),
    ... )
    >>> mlp.train()
    >>> y = mlp(lucid.randn(8, 32))
    >>> y.shape
    (8, 10)

    Verify that eval mode is a no-op:

    >>> drop = nn.AlphaDropout(p=0.5)
    >>> drop.eval()
    >>> x = lucid.randn(4, 16)
    >>> import lucid.linalg
    >>> # Output should equal input exactly in eval mode
    >>> out = drop(x)
    >>> out.shape
    (4, 16)

    See Also
    --------
    Dropout : Standard element-wise dropout.
    FeatureAlphaDropout : Channel-wise variant of alpha dropout.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        _check_dropout_prob(p)
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        return _wrap(_C_engine.nn.alpha_dropout(_unwrap(x), self.p, self.training))

    def extra_repr(self) -> str:
        s: str = f"p={self.p}"
        if self.inplace:
            s += ", inplace=True"
        return s


class Dropout3d(Module):
    r"""Randomly zero entire volumetric feature-map channels during training.

    Extends channel-wise dropout to 5-D inputs of shape
    ``(N, C, D, H, W)``.  A single Bernoulli draw per channel per sample
    determines whether that entire volumetric feature map is zeroed:

    .. math::

        y_{n,c,:,:,:} =
        \begin{cases}
            \dfrac{x_{n,c,:,:,:}}{1 - p} & \text{with probability } 1 - p \\[6pt]
            \mathbf{0}                    & \text{with probability } p
        \end{cases}
        \quad \text{(training)}

    This is the natural generalisation of :class:`Dropout2d` to 3-D
    convolutional networks (e.g. video models or volumetric medical
    imaging).  Because adjacent voxels within a feature map are strongly
    correlated, element-wise dropout would have little regularisation
    effect; zeroing the entire channel is a stronger signal.

    Parameters
    ----------
    p : float, optional
        Probability of zeroing an entire channel.  Must be in ``[0, 1]``.
        Default: ``0.5``.
    inplace : bool, optional
        If ``True``, modify the input in place.  Default: ``False``.

    Shape
    -----
    * **Input**: ``(N, C, D, H, W)`` — batch of 3-D feature volumes.
    * **Output**: ``(N, C, D, H, W)`` — same shape; zeroed channels are
      zero across the full ``D × H × W`` spatial extent.

    Notes
    -----
    The mask is sampled on the ``(N, C)`` axes and broadcast over
    ``(D, H, W)``.  In eval mode the layer is the identity.

    Examples
    --------
    After a 3-D convolutional layer:

    >>> import lucid, lucid.nn as nn
    >>> drop3d = nn.Dropout3d(p=0.1)
    >>> drop3d.train()
    >>> x = lucid.ones(2, 8, 4, 4, 4)    # (N=2, C=8, D=4, H=4, W=4)
    >>> y = drop3d(x)
    >>> y.shape
    (2, 8, 4, 4, 4)

    No-op in eval mode:

    >>> drop3d.eval()
    >>> y_eval = drop3d(lucid.ones(1, 4, 2, 2, 2))
    >>> float(y_eval.sum()) == 32.0
    True

    See Also
    --------
    Dropout : Element-wise scalar dropout.
    Dropout1d : Channel-wise dropout for 3-D inputs.
    Dropout2d : Channel-wise dropout for 4-D inputs.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        _check_dropout_prob(p)
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        from lucid.nn.functional.dropout import dropout3d

        return dropout3d(x, self.p, self.training)

    def extra_repr(self) -> str:
        s: str = f"p={self.p}"
        if self.inplace:
            s += ", inplace=True"
        return s


class FeatureAlphaDropout(Module):
    r"""Channel-wise alpha dropout that preserves SELU self-normalisation.

    Combines the ideas of :class:`Dropout2d` (zero entire channels) and
    :class:`AlphaDropout` (maintain zero mean / unit variance after
    masking).  A single Bernoulli draw per channel per sample decides
    whether that entire feature map is replaced by the SELU saturation
    value :math:`\alpha'`; an affine correction is then applied to
    restore the distributional invariants:

    .. math::

        y_{n,c} =
        \begin{cases}
            x_{n,c} & \text{with probability } 1 - p \\
            \alpha' \cdot \mathbf{1} & \text{with probability } p
        \end{cases}

        \tilde{y}_{n,c} = a \cdot y_{n,c} + b

    where the spatial dimensions are suppressed for clarity,
    :math:`\alpha' = -\lambda\alpha \approx -1.7581`, and the affine
    coefficients :math:`a, b` restore
    :math:`\mathbb{E}[\tilde{y}] = 0`,
    :math:`\operatorname{Var}[\tilde{y}] = 1` across the channel.

    This variant is appropriate for SELU-activated **convolutional**
    networks, where using :class:`AlphaDropout` on individual pixels
    would be too local (adjacent pixels are correlated) while
    :class:`Dropout2d` would break the self-normalising statistics.

    Parameters
    ----------
    p : float, optional
        Probability of replacing an entire channel with :math:`\alpha'`.
        Must be in ``[0, 1]``.  Default: ``0.5``.
    inplace : bool, optional
        Accepted for API compatibility; has no effect.
        Default: ``False``.

    Shape
    -----
    * **Input**: ``(N, C, *)`` — any number of spatial dimensions.
    * **Output**: same shape ``(N, C, *)``.

    Notes
    -----
    The mask is sampled on the ``(N, C)`` axes and broadcast over all
    remaining dimensions, so the full spatial volume of each channel is
    either kept intact or replaced uniformly.

    In eval mode the layer is the identity.

    Use this module **only** after :class:`lucid.nn.SELU` activations.
    Applying it after non-SELU activations yields no statistical benefit.

    Examples
    --------
    In a self-normalising convolutional network:

    >>> import lucid, lucid.nn as nn
    >>> block = nn.Sequential(
    ...     nn.Conv2d(16, 32, kernel_size=3, padding=1),
    ...     nn.SELU(),
    ...     nn.FeatureAlphaDropout(p=0.05),
    ... )
    >>> block.train()
    >>> y = block(lucid.randn(2, 16, 8, 8))
    >>> y.shape
    (2, 32, 8, 8)

    Compare channel mask vs. element mask for SELU conv features:

    >>> fad = nn.FeatureAlphaDropout(p=0.3)
    >>> fad.train()
    >>> x = lucid.randn(4, 8, 6, 6)
    >>> out = fad(x)
    >>> out.shape
    (4, 8, 6, 6)

    See Also
    --------
    AlphaDropout : Element-wise variant for SELU networks.
    Dropout2d : Channel-wise dropout without the SELU correction.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        _check_dropout_prob(p)
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        return feature_alpha_dropout(x, self.p, self.training, self.inplace)

    def extra_repr(self) -> str:
        s: str = f"p={self.p}"
        if self.inplace:
            s += ", inplace=True"
        return s
