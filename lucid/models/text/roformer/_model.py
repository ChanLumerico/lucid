"""RoFormer model (Su et al., 2021) — BERT with rotary position embedding.

Architecturally identical to :class:`lucid.models.text.bert.BERTModel` except
that:

    * No additive position embedding (``position_embeddings`` is removed).
    * Inside every self-attention layer, ``q`` and ``k`` are rotated by
      :func:`lucid.models.text.apply_rotary_emb` before the dot product.

State-dict key naming mirrors HF ``RoFormerModel`` so future weight ports
amount to a flat rename.
"""

import math
from typing import ClassVar, cast, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import MaskedLMMixin
from lucid.models._output import (
    BaseModelOutputWithPooling,
    MaskedLMOutput,
)
from lucid.models._utils._text import extended_attention_mask, text_activation
from lucid.nn.functional import apply_rotary_emb
from lucid.nn.module import Module
from lucid.models.text.roformer._config import RoFormerConfig

# ─────────────────────────────────────────────────────────────────────────────
# Rotary tables — *interleaved* (original RoPE) layout, matching the RoFormer
# reference checkpoints.  Distinct from the shared LLaMA-style
# :class:`lucid.nn.RotaryEmbedding` (half-split layout): here each frequency
# is repeated twice so cos / sin line up with the consecutive feature pairs
# ``(x_{2i}, x_{2i+1})`` that ``apply_rotary_emb(..., interleaved=True)`` rotates.
# ─────────────────────────────────────────────────────────────────────────────


@final
class _RoFormerRotaryEmbedding(Module):
    """Precompute interleaved-layout ``(cos, sin)`` tables for RoPE.

    Each table has shape ``(max_position_embeddings, head_dim)`` and repeats
    every frequency twice along the last dimension:

        cos[p] = [cos(p·θ_0), cos(p·θ_0), cos(p·θ_1), cos(p·θ_1), ...]
        sin[p] = [sin(p·θ_0), sin(p·θ_0), sin(p·θ_1), sin(p·θ_1), ...]

    with θ_i = base^(-2 i / head_dim).  This is the layout consumed by
    :func:`lucid.nn.functional.apply_rotary_emb` with ``interleaved=True``.
    """

    cos_cached: Tensor
    sin_cached: Tensor

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(
                f"_RoFormerRotaryEmbedding requires an even head_dim, got {head_dim}"
            )
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        half = head_dim // 2
        cos_rows: list[list[float]] = []
        sin_rows: list[list[float]] = []
        for p in range(max_position_embeddings):
            cos_row: list[float] = [0.0] * head_dim
            sin_row: list[float] = [0.0] * head_dim
            for i in range(half):
                theta = p / (base ** (2.0 * i / head_dim))
                c = math.cos(theta)
                s = math.sin(theta)
                cos_row[2 * i] = c
                cos_row[2 * i + 1] = c
                sin_row[2 * i] = s
                sin_row[2 * i + 1] = s
            cos_rows.append(cos_row)
            sin_rows.append(sin_row)

        self.register_buffer("cos_cached", lucid.tensor(cos_rows), persistent=False)
        self.register_buffer("sin_cached", lucid.tensor(sin_rows), persistent=False)

    @override
    def forward(self) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        return self.cos_cached, self.sin_cached


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings — *no* position embedding (RoPE handles position implicitly)
# ─────────────────────────────────────────────────────────────────────────────


@final
class _RoFormerEmbeddings(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

    @override
    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        token_type_ids: Tensor | None = None,
    ) -> Tensor:
        B, T = int(input_ids.shape[0]), int(input_ids.shape[1])
        dev = input_ids.device.type

        words = cast(Tensor, self.word_embeddings(input_ids))
        if token_type_ids is None:
            token_type_ids = lucid.zeros((B, T), device=dev).long()
        types = cast(Tensor, self.token_type_embeddings(token_type_ids))

        emb = cast(Tensor, self.LayerNorm(words + types))
        return cast(Tensor, self.dropout(emb))


# ─────────────────────────────────────────────────────────────────────────────
# Rotary self-attention
# ─────────────────────────────────────────────────────────────────────────────


@final
class _RoFormerSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE applied to Q and K."""

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = math.sqrt(self.head_dim)

        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.dropout = nn.Dropout(p=config.attention_dropout)

    def _shape(self, x: Tensor, B: int, T: int) -> Tensor:
        return x.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    @override
    def forward(  # type: ignore[override]
        self,
        hidden: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        B, T, _ = hidden.shape
        q = self._shape(cast(Tensor, self.query(hidden)), B, T)
        k = self._shape(cast(Tensor, self.key(hidden)), B, T)
        v = self._shape(cast(Tensor, self.value(hidden)), B, T)

        # Rotate Q / K by the absolute position phase (values stay un-rotated).
        # RoFormer uses the original interleaved RoPE pairing (x_2i, x_2i+1).
        q, k = apply_rotary_emb(q, k, cos, sin, interleaved=True)

        scores: Tensor = q @ k.permute(0, 1, 3, 2) / self.scale
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = cast(Tensor, self.dropout(F.softmax(scores, dim=-1)))

        ctx: Tensor = probs @ v
        return ctx.permute(0, 2, 1, 3).reshape(B, T, self.num_heads * self.head_dim)


@final
class _RoFormerSelfOutput(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

    @override
    def forward(  # type: ignore[override]
        self, hidden: Tensor, input_tensor: Tensor
    ) -> Tensor:
        h = cast(Tensor, self.dropout(cast(Tensor, self.dense(hidden))))
        return cast(Tensor, self.LayerNorm(h + input_tensor))


@final
class _RoFormerAttention(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.self = _RoFormerSelfAttention(config)
        self.output = _RoFormerSelfOutput(config)

    @override
    def forward(  # type: ignore[override]
        self,
        hidden: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        attn_out = cast(
            Tensor, self.self(hidden, cos, sin, attention_mask=attention_mask)
        )
        return cast(Tensor, self.output(attn_out, hidden))


@final
class _RoFormerIntermediate(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self._act_name = config.hidden_act

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return text_activation(self._act_name, cast(Tensor, self.dense(x)))


@final
class _RoFormerOutput(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

    @override
    def forward(  # type: ignore[override]
        self, hidden: Tensor, input_tensor: Tensor
    ) -> Tensor:
        h = cast(Tensor, self.dropout(cast(Tensor, self.dense(hidden))))
        return cast(Tensor, self.LayerNorm(h + input_tensor))


@final
class _RoFormerLayer(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.attention = _RoFormerAttention(config)
        self.intermediate = _RoFormerIntermediate(config)
        self.output = _RoFormerOutput(config)

    @override
    def forward(  # type: ignore[override]
        self,
        hidden: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        attn_out = cast(
            Tensor, self.attention(hidden, cos, sin, attention_mask=attention_mask)
        )
        inter = cast(Tensor, self.intermediate(attn_out))
        return cast(Tensor, self.output(inter, attn_out))


@final
class _RoFormerEncoder(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            [_RoFormerLayer(config) for _ in range(config.num_hidden_layers)]
        )

    @override
    def forward(  # type: ignore[override]
        self,
        hidden: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layer:
            hidden = cast(
                Tensor, layer(hidden, cos, sin, attention_mask=attention_mask)
            )
        return hidden


@final
class _RoFormerPooler(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    @override
    def forward(self, hidden: Tensor) -> Tensor:  # type: ignore[override]
        cls_hidden = hidden[:, 0]
        return F.tanh(cast(Tensor, self.dense(cls_hidden)))


# ─────────────────────────────────────────────────────────────────────────────
# Top-level encoder + heads
# ─────────────────────────────────────────────────────────────────────────────


class RoFormerModel(PretrainedModel):
    r"""Bare RoFormer encoder trunk with rotary position embeddings.

    Architecturally a BERT-base encoder (12 layers, 768 hidden, 12 heads,
    intermediate width 3072) where the absolute learned position embedding
    is removed and rotary position embedding (RoPE) is applied to the query
    and key vectors inside every self-attention layer.  RoPE injects
    relative-position information **multiplicatively** inside attention so
    the dot product :math:`\langle R_m q, R_n k \rangle` depends only on
    the offset :math:`m - n` — granting better length extrapolation and
    composing naturally with linear attention.

    Parameters
    ----------
    config : RoFormerConfig
        Hyperparameters controlling vocabulary, depth, width, head count,
        and the RoPE ``rotary_base``.  The per-head dimension
        ``hidden_size / num_attention_heads`` must be even (enforced by
        :class:`RoFormerConfig`).

    Attributes
    ----------
    embeddings : nn.Module
        Token + token-type embedding block followed by LayerNorm and
        dropout.  Notably *no* learned position embedding.
    rotary : Module
        Pre-computed interleaved-layout ``(cos, sin)`` tables of shape
        ``(max_position_embeddings, head_dim)``, sliced to the actual
        sequence length at each ``forward``.  Uses the original RoPE
        pairing ``(x_2i, x_2i+1)`` (each frequency repeated twice) rather
        than the shared half-split layout, matching the RoFormer reference
        checkpoints.
    encoder : nn.Module
        Stack of ``config.num_hidden_layers`` rotary-attention transformer
        layers.
    pooler : nn.Module
        Dense + tanh projection of the first token's hidden state used by
        sentence-level classification heads.

    Notes
    -----
    Reference: Su, Lu, Pan, Murtadha, Wen, and Liu, *"RoFormer: Enhanced
    Transformer with Rotary Position Embedding"*, Neurocomputing,
    vol. 568, 2024, article 127063 (arXiv:2104.09864).

    Rotary rotation per :math:`(\cos\theta_i, \sin\theta_i)` pair, with
    :math:`\theta_i = \mathrm{base}^{-2i/d_{\text{head}}}`:

    .. math::

        R_\theta = \begin{pmatrix}
            \cos\theta & -\sin\theta \\
            \sin\theta & \cos\theta
        \end{pmatrix}.

    Applied independently to every :math:`(x_{2i}, x_{2i+1})` pair of
    :math:`Q` and :math:`K` so :math:`q_m^\top k_n` becomes a function of
    :math:`m - n` only.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.roformer import RoFormerConfig, RoFormerModel
    >>> cfg = RoFormerConfig(num_hidden_layers=2, hidden_size=128,
    ...                      num_attention_heads=2, intermediate_size=512)
    >>> model = RoFormerModel(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 2088, 102]])
    >>> out = model(input_ids)
    >>> out.last_hidden_state.shape, out.pooler_output.shape
    ((1, 4, 128), (1, 128))
    """

    config_class: ClassVar[type[RoFormerConfig]] = RoFormerConfig
    base_model_prefix: ClassVar[str] = "roformer"

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__(config)
        head_dim = config.hidden_size // config.num_attention_heads
        self._max_pos = config.max_position_embeddings
        self.embeddings = _RoFormerEmbeddings(config)
        self.rotary = _RoFormerRotaryEmbedding(
            head_dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rotary_base,
        )
        self.encoder = _RoFormerEncoder(config)
        self.pooler = _RoFormerPooler(config)

    @override
    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.word_embeddings

    @override
    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, nn.Embedding):
            raise TypeError(
                "RoFormerModel input embeddings must be nn.Embedding, got "
                f"{type(value).__name__}"
            )
        self.embeddings.word_embeddings = value

    @override
    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
    ) -> BaseModelOutputWithPooling:
        B, T = int(input_ids.shape[0]), int(input_ids.shape[1])
        if T > self._max_pos:
            raise ValueError(
                f"Sequence length {T} exceeds max_position_embeddings {self._max_pos}"
            )

        cos, sin = self.rotary()
        # Slice the precomputed tables to the current sequence length so we
        # don't carry around (max_pos, head_dim) when the prompt is short.
        cos = cos[:T]
        sin = sin[:T]

        hidden = cast(Tensor, self.embeddings(input_ids, token_type_ids=token_type_ids))
        ext_mask = extended_attention_mask(attention_mask, (B, T))
        sequence_output = cast(
            Tensor, self.encoder(hidden, cos, sin, attention_mask=ext_mask)
        )
        pooled_output = cast(Tensor, self.pooler(sequence_output))

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output, pooler_output=pooled_output
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task heads — mirror the BERT set since RoFormer is encoder-only
# ─────────────────────────────────────────────────────────────────────────────


@final
class _RoFormerPredictionHeadTransform(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self._act_name = config.hidden_act

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.dense(x))
        x = text_activation(self._act_name, x)
        return cast(Tensor, self.LayerNorm(x))


@final
class _RoFormerLMPredictionHead(nn.Module):
    bias: Tensor

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.transform = _RoFormerPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(lucid.zeros(config.vocab_size))

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.transform(x))
        logits = cast(Tensor, self.decoder(x))
        return logits + self.bias


@final
class _RoFormerOnlyMLMHead(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.predictions = _RoFormerLMPredictionHead(config)

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.predictions(x))


class RoFormerForMaskedLM(PretrainedModel, MaskedLMMixin):
    r"""RoFormer with a tied masked-language-modeling head.

    Wraps :class:`RoFormerModel` with the BERT-style two-layer projection
    (dense + activation + LayerNorm) mapping each hidden state to
    vocabulary logits.  The decoder weight is tied to the input
    ``word_embeddings`` table when ``config.tie_word_embeddings`` is
    ``True``.  Use for MLM pre-training when RoPE-based relative position
    information is preferred to absolute position embeddings.

    Parameters
    ----------
    config : RoFormerConfig
        Hyperparameters.  ``config.tie_word_embeddings`` (default True)
        controls whether the MLM decoder weight is bound to the input
        embedding matrix.

    Attributes
    ----------
    roformer : RoFormerModel
        Underlying RoPE encoder trunk.
    cls : nn.Module
        Masked-LM prediction head — dense + LayerNorm transform plus a
        decoder of shape ``(hidden_size, vocab_size)``.

    Notes
    -----
    Reference: Su, Lu, Pan, Murtadha, Wen, and Liu, *"RoFormer: Enhanced
    Transformer with Rotary Position Embedding"*, Neurocomputing,
    vol. 568, 2024 (arXiv:2104.09864).

    When ``labels`` is provided, the loss is the standard masked
    cross-entropy

    .. math::

        \mathcal{L}_{\mathrm{MLM}} = -\frac{1}{|M|}
            \sum_{i \in M} \log p_\theta(x_i \mid x_{\setminus M}),

    over the set :math:`M` of masked positions (positions whose label
    equals ``-100`` are excluded).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.roformer import RoFormerConfig, RoFormerForMaskedLM
    >>> cfg = RoFormerConfig(num_hidden_layers=2, hidden_size=128,
    ...                      num_attention_heads=2, intermediate_size=512)
    >>> model = RoFormerForMaskedLM(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 103, 102]])   # [CLS] hello [MASK] [SEP]
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, T=4, V=50000)
    (1, 4, 50000)
    """

    config_class: ClassVar[type[RoFormerConfig]] = RoFormerConfig
    base_model_prefix: ClassVar[str] = "roformer"

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__(config)
        self.roformer = RoFormerModel(config)
        self.cls = _RoFormerOnlyMLMHead(config)
        if config.tie_word_embeddings:
            self.cls.predictions.decoder.weight = (
                self.roformer.embeddings.word_embeddings.weight
            )

    @override
    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> MaskedLMOutput:
        outputs = cast(
            BaseModelOutputWithPooling,
            self.roformer(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ),
        )
        prediction_scores = cast(Tensor, self.cls(outputs.last_hidden_state))

        loss: Tensor | None = None
        if labels is not None:
            loss = self.compute_lm_loss(prediction_scores, labels)

        return MaskedLMOutput(logits=prediction_scores, loss=loss)


class RoFormerForSequenceClassification(PretrainedModel):
    r"""RoFormer with a pooled-CLS sequence-classification head.

    Wraps :class:`RoFormerModel` with a dropout-regularised linear
    classifier operating on the pooled first-token embedding.  Pattern
    mirrors :class:`BERTForSequenceClassification`; use for GLUE-style
    fine-tunes when the RoPE-based relative position bias is preferred.

    Parameters
    ----------
    config : RoFormerConfig
        Hyperparameters.  ``config.num_labels`` sets the output dimension;
        ``config.classifier_dropout`` (falling back to ``hidden_dropout``)
        sets the dropout applied before the linear.

    Attributes
    ----------
    roformer : RoFormerModel
        Underlying RoPE encoder trunk.
    dropout : nn.Dropout
        Dropout layer applied to the pooled first-token embedding.
    classifier : nn.Linear
        Final linear of shape ``(hidden_size, num_labels)`` producing
        per-class logits.

    Notes
    -----
    Reference: Su, Lu, Pan, Murtadha, Wen, and Liu, *"RoFormer: Enhanced
    Transformer with Rotary Position Embedding"*, Neurocomputing,
    vol. 568, 2024 (arXiv:2104.09864).

    Loss (when ``labels`` is provided) is the standard softmax
    cross-entropy:

    .. math::

        \mathcal{L} = -\frac{1}{B} \sum_{b=1}^{B}
            \log p_\theta(y_b \mid x_b).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.roformer import (
    ...     RoFormerConfig, RoFormerForSequenceClassification,
    ... )
    >>> cfg = RoFormerConfig(num_labels=3, num_hidden_layers=2,
    ...                      hidden_size=128, num_attention_heads=2,
    ...                      intermediate_size=512)
    >>> model = RoFormerForSequenceClassification(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, num_labels=3)
    (1, 3)
    """

    config_class: ClassVar[type[RoFormerConfig]] = RoFormerConfig
    base_model_prefix: ClassVar[str] = "roformer"

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__(config)
        self.roformer = RoFormerModel(config)
        drop = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout
        )
        self.dropout = nn.Dropout(p=drop)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    @override
    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> MaskedLMOutput:
        outputs = cast(
            BaseModelOutputWithPooling,
            self.roformer(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ),
        )
        pooled = cast(Tensor, self.dropout(outputs.pooler_output))
        logits = cast(Tensor, self.classifier(pooled))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.long())

        return MaskedLMOutput(logits=logits, loss=loss)


class RoFormerForTokenClassification(PretrainedModel, MaskedLMMixin):
    r"""RoFormer with a per-token classification head.

    Wraps :class:`RoFormerModel` with a dropout-regularised linear
    classifier applied independently at every sequence position.  Pattern
    mirrors :class:`BERTForTokenClassification` — used for sequence-
    labelling tasks (NER / POS / chunking) on RoPE-trained encoders.

    Parameters
    ----------
    config : RoFormerConfig
        Hyperparameters.  ``config.num_labels`` sets the per-position output
        dimension; ``config.classifier_dropout`` (falling back to
        ``hidden_dropout``) sets the dropout applied before the linear.

    Attributes
    ----------
    roformer : RoFormerModel
        Underlying RoPE encoder trunk.
    dropout : nn.Dropout
        Dropout applied to the full sequence hidden states.
    classifier : nn.Linear
        Final linear of shape ``(hidden_size, num_labels)`` mapping each
        token's hidden state to per-class logits.

    Notes
    -----
    Reference: Su, Lu, Pan, Murtadha, Wen, and Liu, *"RoFormer: Enhanced
    Transformer with Rotary Position Embedding"*, Neurocomputing,
    vol. 568, 2024 (arXiv:2104.09864).

    Loss when ``labels`` is provided is the masked cross-entropy inherited
    from :class:`MaskedLMMixin`:

    .. math::

        \mathcal{L} = -\frac{1}{|V|} \sum_{(b, t) \in V}
            \log p_\theta(y_{b, t} \mid x_b),

    where :math:`V` is the set of positions with ``label != -100``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.roformer import (
    ...     RoFormerConfig, RoFormerForTokenClassification,
    ... )
    >>> cfg = RoFormerConfig(num_labels=9, num_hidden_layers=2,
    ...                      hidden_size=128, num_attention_heads=2,
    ...                      intermediate_size=512)
    >>> model = RoFormerForTokenClassification(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 2088, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, T=4, num_labels=9)
    (1, 4, 9)
    """

    config_class: ClassVar[type[RoFormerConfig]] = RoFormerConfig
    base_model_prefix: ClassVar[str] = "roformer"

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__(config)
        self.roformer = RoFormerModel(config)
        drop = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout
        )
        self.dropout = nn.Dropout(p=drop)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    @override
    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> MaskedLMOutput:
        outputs = cast(
            BaseModelOutputWithPooling,
            self.roformer(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ),
        )
        seq = cast(Tensor, self.dropout(outputs.last_hidden_state))
        logits = cast(Tensor, self.classifier(seq))

        loss: Tensor | None = None
        if labels is not None:
            loss = self.compute_lm_loss(logits, labels)

        return MaskedLMOutput(logits=logits, loss=loss)


# ─────────────────────────────────────────────────────────────────────────────
# Multiple-choice + Question-Answering heads
# ─────────────────────────────────────────────────────────────────────────────


class RoFormerForMultipleChoice(PretrainedModel):
    r"""RoFormer with a per-choice multiple-choice classification head.

    Wraps :class:`RoFormerModel` with a per-choice CLS classifier.  Input
    ``(N, C, L)`` flattens to ``(N*C, L)`` for the encoder; the pooled CLS
    embedding of each choice is projected to a scalar and the resulting
    ``(N, C)`` matrix is the per-choice logit.  Used for fine-tunes on
    SWAG / RACE-style commonsense and reading-comprehension tasks.

    Parameters
    ----------
    config : RoFormerConfig
        Hyperparameters.  ``config.classifier_dropout`` (falling back to
        ``hidden_dropout``) sets the dropout applied before the linear.

    Attributes
    ----------
    roformer : RoFormerModel
        Underlying RoPE encoder trunk.
    dropout : nn.Dropout
        Dropout layer applied to each choice's pooled embedding.
    classifier : nn.Linear
        Per-choice scalar projection of shape ``(hidden_size, 1)``.

    Notes
    -----
    Reference: Su, Lu, Pan, Murtadha, Wen, and Liu, *"RoFormer: Enhanced
    Transformer with Rotary Position Embedding"*, Neurocomputing,
    vol. 568, 2024 (arXiv:2104.09864).

    When ``labels`` is supplied the loss is the softmax cross-entropy over
    the ``C`` candidates:

    .. math::

        \mathcal{L}_{\mathrm{MC}}
            = -\frac{1}{N} \sum_{n=1}^{N}
              \log \mathrm{softmax}(z_n)_{y_n}.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.roformer import (
    ...     RoFormerConfig, RoFormerForMultipleChoice,
    ... )
    >>> cfg = RoFormerConfig(num_hidden_layers=2, hidden_size=128,
    ...                      num_attention_heads=2, intermediate_size=512)
    >>> model = RoFormerForMultipleChoice(cfg).eval()
    >>> input_ids = lucid.tensor([[[101, 7592, 102], [101, 2088, 102]]])    # (N=1, C=2, L=3)
    >>> out = model(input_ids)
    >>> out.logits.shape   # (N=1, C=2)
    (1, 2)
    """

    config_class: ClassVar[type[RoFormerConfig]] = RoFormerConfig
    base_model_prefix: ClassVar[str] = "roformer"

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__(config)
        self.roformer = RoFormerModel(config)
        drop = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout
        )
        self.dropout = nn.Dropout(p=drop)
        self.classifier = nn.Linear(config.hidden_size, 1)

    @override
    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> MaskedLMOutput:
        if input_ids.ndim != 3:
            raise ValueError(
                f"RoFormerForMultipleChoice expects input_ids of shape (N, C, L), "
                f"got {tuple(input_ids.shape)}"
            )
        N = int(input_ids.shape[0])
        C = int(input_ids.shape[1])
        L = int(input_ids.shape[2])

        flat_ids = input_ids.reshape(N * C, L)
        flat_mask = (
            attention_mask.reshape(N * C, L) if attention_mask is not None else None
        )
        flat_tt = (
            token_type_ids.reshape(N * C, L) if token_type_ids is not None else None
        )
        outputs = cast(
            BaseModelOutputWithPooling,
            self.roformer(
                flat_ids,
                attention_mask=flat_mask,
                token_type_ids=flat_tt,
            ),
        )
        pooled = cast(Tensor, self.dropout(outputs.pooler_output))  # (N*C, H)
        logits_flat = cast(Tensor, self.classifier(pooled))  # (N*C, 1)
        logits = logits_flat.reshape(N, C)  # (N, C)

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.long())

        return MaskedLMOutput(logits=logits, loss=loss)


class RoFormerForQuestionAnswering(PretrainedModel):
    r"""RoFormer with a 2-way span head for extractive question answering.

    Wraps :class:`RoFormerModel` with a single linear of output width 2,
    producing start- and end-position logits over each token in the input.
    Identical contract to :class:`BERTForQuestionAnswering`; the SQuAD
    v1.1 / v2.0 fine-tuning recipe applied to a RoPE-trained encoder.

    Parameters
    ----------
    config : RoFormerConfig
        Hyperparameters.  The QA head is always 2-way; ``num_labels`` is
        ignored.

    Attributes
    ----------
    roformer : RoFormerModel
        Underlying RoPE encoder trunk.
    qa_outputs : nn.Linear
        Final linear of shape ``(hidden_size, 2)`` mapping each token's
        hidden state to ``(start_logit, end_logit)``.

    Notes
    -----
    Reference: Su, Lu, Pan, Murtadha, Wen, and Liu, *"RoFormer: Enhanced
    Transformer with Rotary Position Embedding"*, Neurocomputing,
    vol. 568, 2024 (arXiv:2104.09864).

    When both ``start_positions`` and ``end_positions`` are supplied the
    loss is the symmetric average of two cross-entropies:

    .. math::

        \mathcal{L} = \tfrac{1}{2}\!\left(
            \mathrm{CE}(z^{\mathrm{start}}, y^{\mathrm{start}})
          + \mathrm{CE}(z^{\mathrm{end}},   y^{\mathrm{end}})
        \right).

    The returned ``logits`` tensor has shape ``(B, T, 2)``; index
    ``[..., 0]`` for start scores and ``[..., 1]`` for end scores.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.roformer import (
    ...     RoFormerConfig, RoFormerForQuestionAnswering,
    ... )
    >>> cfg = RoFormerConfig(num_hidden_layers=2, hidden_size=128,
    ...                      num_attention_heads=2, intermediate_size=512)
    >>> model = RoFormerForQuestionAnswering(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 2040, 2003, 102, 1045, 2572, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, T=7, 2)
    (1, 7, 2)
    """

    config_class: ClassVar[type[RoFormerConfig]] = RoFormerConfig
    base_model_prefix: ClassVar[str] = "roformer"

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__(config)
        self.roformer = RoFormerModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    @override
    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        start_positions: Tensor | None = None,
        end_positions: Tensor | None = None,
    ) -> MaskedLMOutput:
        outputs = cast(
            BaseModelOutputWithPooling,
            self.roformer(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ),
        )
        logits = cast(Tensor, self.qa_outputs(outputs.last_hidden_state))  # (B, T, 2)

        loss: Tensor | None = None
        if start_positions is not None and end_positions is not None:
            start_logits = logits[..., 0]
            end_logits = logits[..., 1]
            loss = (
                F.cross_entropy(start_logits, start_positions.long())
                + F.cross_entropy(end_logits, end_positions.long())
            ) / 2.0

        return MaskedLMOutput(logits=logits, loss=loss)
