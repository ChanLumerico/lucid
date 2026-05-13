r"""
Transformer modules: TransformerEncoderLayer, TransformerEncoder,
TransformerDecoderLayer, TransformerDecoder, Transformer.
"""

from typing import cast

from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike

from lucid.nn.module import Module
from lucid.nn.modules.attention import MultiheadAttention
from lucid.nn.modules.normalization import LayerNorm

from lucid.nn.modules.linear import Linear
from lucid.nn.modules.dropout import Dropout
from lucid.nn.functional.activations import gelu, relu


class TransformerEncoderLayer(Module):
    r"""Single transformer encoder layer: self-attention followed by a
    position-wise feed-forward network, with residual connections and
    layer normalisation.

    This is one building block of the encoder stack described in
    "Attention Is All You Need" (Vaswani et al., 2017).  A full encoder
    is formed by stacking :math:`N` copies of this layer (see
    :class:`TransformerEncoder`).

    **Post-LN (original paper, default ``norm_first=False``)**:

    .. math::

        x = \text{LayerNorm}\!\left(x + \text{Dropout}(\text{SelfAttn}(x))\right)

    .. math::

        x = \text{LayerNorm}\!\left(x + \text{Dropout}(\text{FFN}(x))\right)

    Normalisation *after* the residual addition keeps the residual
    stream unnormalised, which can cause instability at the start of
    training for very deep models.

    **Pre-LN (``norm_first=True``)**:

    .. math::

        x = x + \text{Dropout}(\text{SelfAttn}(\text{LayerNorm}(x)))

    .. math::

        x = x + \text{Dropout}(\text{FFN}(\text{LayerNorm}(x)))

    Normalising *before* the sub-layer keeps the residual stream on the
    identity path, which substantially improves gradient flow and allows
    training without learning-rate warm-up.  Pre-LN is the default in
    most modern large-scale transformers (GPT-2, GPT-3, LLaMA, etc.).

    **Feed-forward network (FFN)**:

    .. math::

        \text{FFN}(x) = \text{Linear}_2\!\left(
            \text{Dropout}(\sigma(\text{Linear}_1(x)))
        \right)

    where :math:`\sigma` is either ReLU or GELU depending on
    ``activation``.  The inner dimension ``dim_feedforward`` is
    typically set to :math:`4 \times d_{\text{model}}` as in the
    original paper.

    Parameters
    ----------
    d_model : int
        Dimensionality of the model's hidden representations
        :math:`d_{\text{model}}`.  All sub-layers produce outputs of
        this width.
    nhead : int
        Number of attention heads in the self-attention sub-layer.
        Must divide ``d_model`` evenly.
    dim_feedforward : int, optional
        Inner (hidden) width of the two-layer FFN.  Default: ``2048``.
    dropout : float, optional
        Dropout probability applied after each sub-layer output and
        inside the FFN activation.  Default: ``0.1``.
    activation : str, optional
        Non-linearity used inside the FFN.  Supported values:
        ``"relu"`` (default) and ``"gelu"``.
    batch_first : bool, optional
        If ``True``, inputs and outputs are ``(batch, seq, feature)``.
        If ``False`` (default), they are ``(seq, batch, feature)``.
    norm_first : bool, optional
        If ``True``, applies Pre-LN (layer norm *before* each
        sub-layer).  If ``False`` (default), applies Post-LN (layer
        norm *after* the residual addition).
    device : DeviceLike, optional
        Device for all sub-module parameters.
    dtype : DTypeLike, optional
        Data type for all sub-module parameters.

    Attributes
    ----------
    self_attn : MultiheadAttention
        Multi-head self-attention sub-layer.
    linear1 : Linear
        First linear layer of the FFN: ``d_model → dim_feedforward``.
    linear2 : Linear
        Second linear layer of the FFN: ``dim_feedforward → d_model``.
    norm1 : LayerNorm
        Layer normalisation applied around the self-attention sub-layer.
    norm2 : LayerNorm
        Layer normalisation applied around the FFN sub-layer.
    dropout1 : Dropout
        Dropout applied to the self-attention output before the
        residual addition.
    dropout2 : Dropout
        Dropout applied inside the FFN (after the activation).
    dropout3 : Dropout
        Dropout applied to the FFN output before the residual addition.

    Shape
    -----
    * Input ``src``: :math:`(S, N, E)` when ``batch_first=False``,
      or :math:`(N, S, E)` when ``batch_first=True``.
    * Output: same shape as ``src``.

    where :math:`S` is the source sequence length, :math:`N` is the
    batch size, and :math:`E` = ``d_model``.

    Notes
    -----
    The three ``Dropout`` modules (``dropout1``, ``dropout2``,
    ``dropout3``) are each initialised with the same probability but
    are distinct instances.  This ensures that each sub-layer's dropout
    mask is sampled independently, giving the model more regularisation
    diversity.

    For inference, call ``model.eval()`` to disable all three dropout
    layers simultaneously through Lucid's ``Module.training`` flag.

    Examples
    --------
    **Basic Post-LN encoder layer** (default):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    >>> # Sequence-first layout: (seq_len, batch, d_model)
    >>> src = lucid.randn(20, 4, 512)
    >>> out = layer(src)
    >>> out.shape
    (20, 4, 512)

    **Pre-LN variant with GELU and batch_first layout**:

    >>> layer = nn.TransformerEncoderLayer(
    ...     d_model=256,
    ...     nhead=4,
    ...     dim_feedforward=1024,
    ...     dropout=0.0,
    ...     activation="gelu",
    ...     batch_first=True,
    ...     norm_first=True,
    ... )
    >>> src = lucid.randn(2, 15, 256)       # (batch, seq, d_model)
    >>> out = layer(src)
    >>> out.shape
    (2, 15, 256)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = False,
        norm_first: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_val = dropout
        self.activation = activation
        self.batch_first = batch_first
        self.norm_first = norm_first

        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        self.linear1 = Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.linear2 = Linear(dim_feedforward, d_model, device=device, dtype=dtype)
        self.norm1 = LayerNorm(d_model, device=device, dtype=dtype)
        self.norm2 = LayerNorm(d_model, device=device, dtype=dtype)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def _ff(self, x: Tensor) -> Tensor:
        act = gelu if self.activation == "gelu" else relu
        h = cast(Tensor, self.linear1(x))
        return self.linear2(cast(Tensor, self.dropout2(act(h))))  # type: ignore[return-value]

    def forward(  # type: ignore[override]  # narrower signature than Function/Module base by design
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        if self.norm_first:
            normed: Tensor = cast(Tensor, self.norm1(src))
            src2, _ = self.self_attn(
                normed,
                normed,
                normed,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
                is_causal=is_causal,
            )
            src = src + cast(Tensor, self.dropout1(src2))
            src = src + cast(
                Tensor, self.dropout3(self._ff(cast(Tensor, self.norm2(src))))
            )
        else:
            src2, _ = self.self_attn(
                src,
                src,
                src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
                is_causal=is_causal,
            )
            src = cast(Tensor, self.norm1(src + cast(Tensor, self.dropout1(src2))))
            src = cast(
                Tensor, self.norm2(src + cast(Tensor, self.dropout3(self._ff(src))))
            )
        return src

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, nhead={self.nhead}, "
            f"dim_feedforward={self.dim_feedforward}, dropout={self.dropout_val}"
        )


class TransformerEncoder(Module):
    r"""A stack of :math:`N` identical :class:`TransformerEncoderLayer` modules.

    The encoder maps a source sequence of continuous representations
    :math:`(x_1, \ldots, x_S)` into another sequence of the same shape.
    Each layer refines the representation by attending to all positions
    simultaneously (self-attention has no ordering constraint), allowing
    every position to gather context from the entire sequence in a single
    forward pass.

    Formally, letting :math:`\text{Layer}_i` denote the :math:`i`-th
    encoder layer:

    .. math::

        h^{(0)} = \text{src}

    .. math::

        h^{(i)} = \text{Layer}_i\!\left(h^{(i-1)},\;
            \text{mask},\; \text{src\_key\_padding\_mask}\right),
        \quad i = 1, \ldots, N

    .. math::

        \text{output} =
        \begin{cases}
            \text{LayerNorm}\!\left(h^{(N)}\right) & \text{if norm is set} \\
            h^{(N)} & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    encoder_layer : TransformerEncoderLayer
        A single configured encoder layer whose hyperparameters
        (``d_model``, ``nhead``, ``dim_feedforward``, ``dropout``,
        ``activation``, ``batch_first``, ``norm_first``) are used
        to construct ``num_layers`` independent copies.  The provided
        instance itself is **not** reused as one of the copies; a
        fresh layer is always instantiated per index.
    num_layers : int
        Number of encoder layers :math:`N` to stack.
    norm : Module or None, optional
        Optional final normalisation module applied to the output of
        the last layer.  Commonly a :class:`LayerNorm` of size
        ``d_model``.  Default: ``None``.

    Attributes
    ----------
    layers : list[TransformerEncoderLayer]
        The :math:`N` encoder layer instances, also registered as
        sub-modules ``"0"``, ``"1"``, …, ``"N-1"`` for proper
        parameter tracking and serialisation.
    norm : Module or None
        The optional post-stack normalisation module.
    num_layers : int
        The number of stacked layers.

    Shape
    -----
    * Input ``src``: :math:`(S, N, E)` when ``batch_first=False``,
      or :math:`(N, S, E)` when ``batch_first=True``.
    * Output: same shape as ``src``.

    where :math:`S` is the source sequence length, :math:`N` is the
    batch size, and :math:`E` = ``d_model``.

    Notes
    -----
    The ``mask`` and ``src_key_padding_mask`` arguments are propagated
    unchanged to every layer in the stack.  A boolean mask follows the
    convention that ``True`` indicates positions to **ignore** (mask
    out), matching the additive ``-inf`` semantics used internally.

    When ``norm`` is provided it is registered as a sub-module named
    ``"norm"``, so its parameters appear in ``state_dict()`` and are
    updated by the optimiser.

    Examples
    --------
    **Six-layer encoder with final LayerNorm** (standard transformer):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> layer = nn.TransformerEncoderLayer(d_model=512, nhead=8,
    ...                                    dim_feedforward=2048)
    >>> norm = nn.LayerNorm(512)
    >>> encoder = nn.TransformerEncoder(layer, num_layers=6, norm=norm)
    >>> src = lucid.randn(30, 4, 512)       # (seq, batch, d_model)
    >>> memory = encoder(src)
    >>> memory.shape
    (30, 4, 512)

    **Encoder with source padding mask** (variable-length sequences):

    >>> encoder = nn.TransformerEncoder(
    ...     nn.TransformerEncoderLayer(d_model=128, nhead=4,
    ...                                batch_first=True),
    ...     num_layers=3,
    ... )
    >>> src = lucid.randn(2, 10, 128)
    >>> # True = padding position to be ignored
    >>> pad_mask = lucid.tensor([[False] * 7 + [True] * 3,
    ...                          [False] * 10])
    >>> out = encoder(src, src_key_padding_mask=pad_mask)
    >>> out.shape
    (2, 10, 128)
    """

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: Module | None = None,
    ) -> None:
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(
                encoder_layer.d_model,
                encoder_layer.nhead,
                encoder_layer.dim_feedforward,
                encoder_layer.dropout_val,
                encoder_layer.activation,
                encoder_layer.batch_first,
                encoder_layer.norm_first,
            )
            for _ in range(num_layers)
        ]
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)
        self.norm = norm
        if norm is not None:
            self.add_module("norm", norm)
        self.num_layers = num_layers

    def forward(  # type: ignore[override]  # narrower signature than Function/Module base by design
        self,
        src: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        output = src
        for layer in self.layers:
            output = cast(
                Tensor,
                layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask),
            )
        if self.norm is not None:
            output = cast(Tensor, self.norm(output))
        return output

    def extra_repr(self) -> str:
        return f"num_layers={self.num_layers}"


class TransformerDecoderLayer(Module):
    r"""Single transformer decoder layer: masked self-attention,
    cross-attention over encoder memory, and a position-wise FFN —
    each with residual connections and layer normalisation.

    The decoder layer introduces a second attention sub-layer that
    attends to the encoder output (``memory``), connecting the
    generation process to the source context.  The three sub-layers
    and their residual connections are:

    **Post-LN (default, ``norm_first=False``)**:

    .. math::

        x = \text{LayerNorm}\!\left(
            x + \text{Dropout}(\text{SelfAttn}(x,\, x,\, x))
        \right)
        \quad \text{[masked / causal self-attention]}

    .. math::

        x = \text{LayerNorm}\!\left(
            x + \text{Dropout}(\text{CrossAttn}(x,\, m,\, m))
        \right)
        \quad \text{[cross-attention to encoder memory } m \text{]}

    .. math::

        x = \text{LayerNorm}\!\left(
            x + \text{Dropout}(\text{FFN}(x))
        \right)

    **Pre-LN (``norm_first=True``)**:

    .. math::

        x = x + \text{Dropout}(\text{SelfAttn}(\text{LayerNorm}(x)))

    .. math::

        x = x + \text{Dropout}(\text{CrossAttn}(\text{LayerNorm}(x),\, m,\, m))

    .. math::

        x = x + \text{Dropout}(\text{FFN}(\text{LayerNorm}(x)))

    where :math:`m` is the ``memory`` tensor produced by the encoder.

    **Feed-forward network (FFN)**:

    .. math::

        \text{FFN}(x) = \text{Linear}_2\!\left(
            \text{Dropout}(\sigma(\text{Linear}_1(x)))
        \right)

    Parameters
    ----------
    d_model : int
        Dimensionality of the model's hidden representations.
    nhead : int
        Number of attention heads in both the self-attention and the
        cross-attention sub-layers.
    dim_feedforward : int, optional
        Inner width of the FFN.  Default: ``2048``.
    dropout : float, optional
        Dropout probability for all sub-layer outputs and the FFN
        activation.  Default: ``0.1``.
    activation : str, optional
        Non-linearity inside the FFN: ``"relu"`` (default) or
        ``"gelu"``.
    batch_first : bool, optional
        If ``True``, all tensors use ``(batch, seq, feature)`` layout.
        Default: ``False`` (sequence-first).
    norm_first : bool, optional
        If ``True``, uses Pre-LN order.  Default: ``False`` (Post-LN).
    device : DeviceLike, optional
        Device for all sub-module parameters.
    dtype : DTypeLike, optional
        Data type for all sub-module parameters.

    Attributes
    ----------
    self_attn : MultiheadAttention
        Masked (causal) self-attention over the target sequence.
    multihead_attn : MultiheadAttention
        Cross-attention: queries from target, keys/values from memory.
    linear1 : Linear
        First FFN layer: ``d_model → dim_feedforward``.
    linear2 : Linear
        Second FFN layer: ``dim_feedforward → d_model``.
    norm1 : LayerNorm
        Normalisation for the self-attention sub-layer.
    norm2 : LayerNorm
        Normalisation for the cross-attention sub-layer.
    norm3 : LayerNorm
        Normalisation for the FFN sub-layer.
    dropout1 : Dropout
        Dropout after the self-attention output.
    dropout2 : Dropout
        Dropout inside the FFN (after activation).
    dropout3 : Dropout
        Dropout after the cross-attention output.
    dropout4 : Dropout
        Dropout after the FFN output.

    Shape
    -----
    * ``tgt``: :math:`(T, N, E)` when ``batch_first=False``,
      or :math:`(N, T, E)` when ``batch_first=True``.
    * ``memory``: :math:`(S, N, E)` when ``batch_first=False``,
      or :math:`(N, S, E)` when ``batch_first=True``.
    * Output: same shape as ``tgt``.

    where :math:`T` is the target sequence length, :math:`S` is the
    source sequence length, :math:`N` is the batch size, and
    :math:`E` = ``d_model``.

    Notes
    -----
    The self-attention sub-layer is typically used with
    ``tgt_is_causal=True`` during autoregressive decoding so that
    position :math:`i` cannot attend to any future position
    :math:`j > i`.  This prevents information leakage from future
    tokens during teacher-forced training.

    The cross-attention sub-layer uses the encoder output ``memory``
    as both keys and values, while the decoder's current hidden state
    provides the queries.  This is the mechanism by which the decoder
    conditions its generation on the source sequence.

    Examples
    --------
    **Basic decoder layer** (sequence-first layout):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
    >>> tgt = lucid.randn(10, 2, 512)       # (tgt_len, batch, d_model)
    >>> memory = lucid.randn(20, 2, 512)    # (src_len, batch, d_model)
    >>> out = layer(tgt, memory)
    >>> out.shape
    (10, 2, 512)

    **Pre-LN decoder layer with causal self-attention**:

    >>> layer = nn.TransformerDecoderLayer(
    ...     d_model=256, nhead=4, activation="gelu",
    ...     batch_first=True, norm_first=True,
    ... )
    >>> tgt = lucid.randn(2, 8, 256)
    >>> memory = lucid.randn(2, 12, 256)
    >>> out = layer(tgt, memory, tgt_is_causal=True)
    >>> out.shape
    (2, 8, 256)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = False,
        norm_first: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_val = dropout
        self.activation = activation
        self.batch_first = batch_first
        self.norm_first = norm_first

        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        self.multihead_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        self.linear1 = Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.linear2 = Linear(dim_feedforward, d_model, device=device, dtype=dtype)
        self.norm1 = LayerNorm(d_model, device=device, dtype=dtype)
        self.norm2 = LayerNorm(d_model, device=device, dtype=dtype)
        self.norm3 = LayerNorm(d_model, device=device, dtype=dtype)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.dropout4 = Dropout(dropout)

    def _ff(self, x: Tensor) -> Tensor:
        act = gelu if self.activation == "gelu" else relu
        h = cast(Tensor, self.linear1(x))
        return self.linear2(cast(Tensor, self.dropout2(act(h))))  # type: ignore[return-value]

    def forward(  # type: ignore[override]  # narrower signature than Function/Module base by design
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        if self.norm_first:
            normed: Tensor = cast(Tensor, self.norm1(tgt))
            tgt2, _ = self.self_attn(
                normed,
                normed,
                normed,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=False,
                is_causal=tgt_is_causal,
            )
            tgt = tgt + cast(Tensor, self.dropout1(tgt2))
            tgt2, _ = self.multihead_attn(
                cast(Tensor, self.norm2(tgt)),
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,
                is_causal=memory_is_causal,
            )
            tgt = tgt + cast(Tensor, self.dropout3(tgt2))
            tgt = tgt + cast(
                Tensor, self.dropout4(self._ff(cast(Tensor, self.norm3(tgt))))
            )
        else:
            tgt2, _ = self.self_attn(
                tgt,
                tgt,
                tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=False,
                is_causal=tgt_is_causal,
            )
            tgt = cast(Tensor, self.norm1(tgt + cast(Tensor, self.dropout1(tgt2))))
            tgt2, _ = self.multihead_attn(
                tgt,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,
                is_causal=memory_is_causal,
            )
            tgt = cast(Tensor, self.norm2(tgt + cast(Tensor, self.dropout3(tgt2))))
            tgt = cast(
                Tensor, self.norm3(tgt + cast(Tensor, self.dropout4(self._ff(tgt))))
            )
        return tgt

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, nhead={self.nhead}, "
            f"dim_feedforward={self.dim_feedforward}, dropout={self.dropout_val}"
        )


class TransformerDecoder(Module):
    r"""A stack of :math:`N` identical :class:`TransformerDecoderLayer` modules.

    The decoder takes a target sequence and a ``memory`` tensor produced
    by the encoder and generates a sequence of hidden representations
    that are conditioned on both.  Every decoder layer receives the
    same ``memory`` as its cross-attention context, allowing the
    decoder to attend to all encoder positions at each decoding step.

    Formally, letting :math:`\text{Layer}_i` denote the :math:`i`-th
    decoder layer:

    .. math::

        h^{(0)} = \text{tgt}

    .. math::

        h^{(i)} = \text{Layer}_i\!\left(h^{(i-1)},\; \text{memory},\;
            \text{tgt\_mask},\; \text{memory\_mask}\right),
        \quad i = 1, \ldots, N

    .. math::

        \text{output} =
        \begin{cases}
            \text{LayerNorm}\!\left(h^{(N)}\right) & \text{if norm is set} \\
            h^{(N)} & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    decoder_layer : TransformerDecoderLayer
        A single configured decoder layer.  Its hyperparameters
        are copied to construct ``num_layers`` independent instances.
        The provided layer itself is not reused.
    num_layers : int
        Number of decoder layers :math:`N` to stack.
    norm : Module or None, optional
        Optional final normalisation applied to the last layer's
        output.  Default: ``None``.

    Attributes
    ----------
    layers : list[TransformerDecoderLayer]
        The :math:`N` decoder layer instances, registered as
        sub-modules ``"0"``, ``"1"``, …, ``"N-1"``.
    norm : Module or None
        The optional post-stack normalisation module.
    num_layers : int
        Number of stacked layers.

    Shape
    -----
    * ``tgt``: :math:`(T, N, E)` when ``batch_first=False``,
      or :math:`(N, T, E)` when ``batch_first=True``.
    * ``memory``: :math:`(S, N, E)` when ``batch_first=False``,
      or :math:`(N, S, E)` when ``batch_first=True``.
    * Output: same shape as ``tgt``.

    where :math:`T` is the target sequence length, :math:`S` is the
    source sequence length, :math:`N` is the batch size, and
    :math:`E` = ``d_model``.

    Notes
    -----
    The ``TransformerDecoder`` is the natural building block for
    sequence-to-sequence tasks (machine translation, summarisation,
    speech synthesis) and for autoregressive language modelling when
    combined with an encoder that processes the conditioning context.

    For autoregressive generation at inference time, the decoder is
    invoked step-by-step: at each step, the ``tgt`` tensor grows by
    one token (the previously generated token is appended), and the
    ``memory`` tensor remains fixed as the encoded source.  A causal
    mask passed through ``tgt_mask`` ensures each position only
    attends to previously generated tokens.

    Examples
    --------
    **Six-layer decoder** (standard seq2seq decoder):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> d_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
    >>> norm = nn.LayerNorm(512)
    >>> decoder = nn.TransformerDecoder(d_layer, num_layers=6, norm=norm)
    >>> tgt = lucid.randn(15, 2, 512)       # (tgt_len, batch, d_model)
    >>> memory = lucid.randn(30, 2, 512)    # (src_len, batch, d_model)
    >>> out = decoder(tgt, memory)
    >>> out.shape
    (15, 2, 512)

    **Decoder for autoregressive generation** (batch_first):

    >>> d_layer = nn.TransformerDecoderLayer(
    ...     d_model=256, nhead=4, batch_first=True
    ... )
    >>> decoder = nn.TransformerDecoder(d_layer, num_layers=4)
    >>> memory = lucid.randn(2, 20, 256)    # encoder output
    >>> # At step t, tgt contains the t tokens generated so far
    >>> tgt = lucid.randn(2, 5, 256)
    >>> out = decoder(tgt, memory, tgt_mask=None)
    >>> out.shape
    (2, 5, 256)
    """

    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: Module | None = None,
    ) -> None:
        super().__init__()
        self.layers = [
            TransformerDecoderLayer(
                decoder_layer.d_model,
                decoder_layer.nhead,
                decoder_layer.dim_feedforward,
                decoder_layer.dropout_val,
                decoder_layer.activation,
                decoder_layer.batch_first,
                decoder_layer.norm_first,
            )
            for _ in range(num_layers)
        ]
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)
        self.norm = norm
        if norm is not None:
            self.add_module("norm", norm)
        self.num_layers = num_layers

    def forward(  # type: ignore[override]  # narrower signature than Function/Module base by design
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        output = tgt
        for layer in self.layers:
            output = cast(
                Tensor,
                layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask),
            )
        if self.norm is not None:
            output = cast(Tensor, self.norm(output))
        return output

    def extra_repr(self) -> str:
        return f"num_layers={self.num_layers}"


class Transformer(Module):
    r"""Full encoder-decoder Transformer architecture.

    Implements the complete sequence-to-sequence model introduced in
    "Attention Is All You Need" (Vaswani et al., 2017, arXiv:1706.03762).
    The model consists of:

    1. An :class:`TransformerEncoder` that encodes the source sequence
       into a continuous memory representation.
    2. A :class:`TransformerDecoder` that generates the target sequence
       conditioned on the encoder memory.

    High-level data flow:

    .. math::

        \text{memory} = \text{Encoder}(\text{src},\; \text{src\_mask},\;
            \text{src\_key\_padding\_mask})

    .. math::

        \text{output} = \text{Decoder}(\text{tgt},\; \text{memory},\;
            \text{tgt\_mask},\; \text{memory\_mask},\;
            \text{tgt\_key\_padding\_mask},\;
            \text{memory\_key\_padding\_mask})

    Both the encoder and decoder are built with a final
    :class:`LayerNorm` applied after the last layer.

    Parameters
    ----------
    d_model : int, optional
        Dimensionality of all model representations
        :math:`d_{\text{model}}`.  Default: ``512``.
    nhead : int, optional
        Number of attention heads in every multi-head attention
        sub-layer (encoder self-attention, decoder self-attention,
        and decoder cross-attention all use the same ``nhead``).
        Default: ``8``.
    num_encoder_layers : int, optional
        Number of layers in the encoder stack.  Default: ``6``.
    num_decoder_layers : int, optional
        Number of layers in the decoder stack.  Default: ``6``.
    dim_feedforward : int, optional
        Inner dimension of the position-wise FFN in each layer.
        Default: ``2048``.
    dropout : float, optional
        Dropout probability applied throughout the model.
        Default: ``0.1``.
    activation : str, optional
        Activation function for all FFN layers: ``"relu"`` (default)
        or ``"gelu"``.
    batch_first : bool, optional
        If ``True``, all input and output tensors use
        ``(batch, seq, feature)`` layout.  Default: ``False``
        (sequence-first).
    device : DeviceLike, optional
        Device for all parameters.
    dtype : DTypeLike, optional
        Data type for all parameters.

    Attributes
    ----------
    encoder : TransformerEncoder
        The encoder module (``num_encoder_layers`` stacked encoder
        layers + final LayerNorm).
    decoder : TransformerDecoder
        The decoder module (``num_decoder_layers`` stacked decoder
        layers + final LayerNorm).
    d_model : int
        Model dimension used at construction.
    nhead : int
        Number of attention heads used at construction.

    Shape
    -----
    When ``batch_first=False`` (default):

    * ``src``: :math:`(S, N, E)`
    * ``tgt``: :math:`(T, N, E)`
    * Output: :math:`(T, N, E)`

    When ``batch_first=True``:

    * ``src``: :math:`(N, S, E)`
    * ``tgt``: :math:`(N, T, E)`
    * Output: :math:`(N, T, E)`

    where :math:`S` is the source length, :math:`T` is the target
    length, :math:`N` is the batch size, and :math:`E` = ``d_model``.

    Notes
    -----
    **Positional encoding** is *not* included in this module.
    The caller is responsible for adding positional information to
    ``src`` and ``tgt`` before passing them in.  The standard approach
    is sinusoidal encodings (Vaswani et al.) or learned absolute /
    rotary position embeddings.

    **Masking conventions**:

    * ``src_mask`` / ``tgt_mask`` / ``memory_mask``: attention bias
      masks of shape ``(S, S)``, ``(T, T)``, or ``(T, S)``
      respectively (or ``(N*H, ...)`` per-head variants).  Boolean
      ``True`` = ignore that position.
    * ``*_key_padding_mask``: boolean masks of shape ``(N, S)`` or
      ``(N, T)`` indicating padding positions (``True`` = pad).

    **Initialisation**: All projection weights use Xavier uniform
    initialisation (via :class:`MultiheadAttention`) and all biases
    are zeroed.  This matches the original paper's setup and provides
    stable gradient flow at the start of training.

    **Inference / generation**: For autoregressive decoding, invoke
    the encoder once to obtain ``memory``, then call the decoder
    iteratively with a growing ``tgt`` tensor and an appropriate
    causal mask.  Each decoder call produces logits for the next
    token at the last position.

    Examples
    --------
    **Standard translation model** (default hyperparameters):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> model = nn.Transformer(d_model=512, nhead=8,
    ...                        num_encoder_layers=6,
    ...                        num_decoder_layers=6)
    >>> src = lucid.randn(10, 2, 512)       # (src_len, batch, d_model)
    >>> tgt = lucid.randn(7, 2, 512)        # (tgt_len, batch, d_model)
    >>> out = model(src, tgt)
    >>> out.shape
    (7, 2, 512)

    **Compact model with batch_first and GELU**:

    >>> model = nn.Transformer(
    ...     d_model=256, nhead=4,
    ...     num_encoder_layers=3, num_decoder_layers=3,
    ...     dim_feedforward=512, dropout=0.0,
    ...     activation="gelu", batch_first=True,
    ... )
    >>> src = lucid.randn(2, 20, 256)
    >>> tgt = lucid.randn(2, 15, 256)
    >>> out = model(src, tgt)
    >>> out.shape
    (2, 15, 256)
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        enc_layer = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            batch_first,
            device=device,
            dtype=dtype,
        )
        dec_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            batch_first,
            device=device,
            dtype=dtype,
        )
        enc_norm = LayerNorm(d_model, device=device, dtype=dtype)
        dec_norm = LayerNorm(d_model, device=device, dtype=dtype)
        self.encoder = TransformerEncoder(enc_layer, num_encoder_layers, norm=enc_norm)
        self.decoder = TransformerDecoder(dec_layer, num_decoder_layers, norm=dec_norm)
        self.d_model = d_model
        self.nhead = nhead

    def forward(  # type: ignore[override]  # narrower signature than Function/Module base by design
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        memory = cast(
            Tensor,
            self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask),
        )
        return cast(
            Tensor,
            self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask),
        )

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, nhead={self.nhead}"
