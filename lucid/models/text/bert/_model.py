"""BERT model (Devlin et al., 2018) — encoder-only Transformer.

Module / parameter naming matches HuggingFace Transformers' ``BERTModel`` so
state dicts can be ported with a flat key rename.  Top-level layout:

    bert.embeddings.{word, position, token_type}_embeddings
    bert.embeddings.LayerNorm
    bert.encoder.layer.{i}.attention.self.{query, key, value}
    bert.encoder.layer.{i}.attention.output.{dense, LayerNorm}
    bert.encoder.layer.{i}.intermediate.dense
    bert.encoder.layer.{i}.output.{dense, LayerNorm}
    bert.pooler.dense
    cls.predictions.{transform.{dense, LayerNorm}, decoder}   (MLM head)
    classifier                                                (cls / token / qa)
"""

import math
from dataclasses import dataclass
from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import MaskedLMMixin
from lucid.models._output import (
    BaseModelOutputWithPooling,
    CausalLMOutput,
    MaskedLMOutput,
    ModelOutput,
)
from lucid.models._utils._text import extended_attention_mask, text_activation
from lucid.models.text.bert._config import BERTConfig

# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────


class _BERTEmbeddings(nn.Module):
    """Word + position + token-type embedding sum, then LN + Dropout."""

    position_ids: Tensor

    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        # HF names: ``LayerNorm`` (capitalised) so checkpoints port directly.
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

        # Position ids are deterministic [0, max_pos) — register as a buffer so
        # they ride along with .to(device=...).
        pos = lucid.arange(config.max_position_embeddings).long().unsqueeze(0)
        self.register_buffer("position_ids", pos, persistent=False)

    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        token_type_ids: Tensor | None = None,
    ) -> Tensor:
        B, T = int(input_ids.shape[0]), int(input_ids.shape[1])
        dev = input_ids.device.type

        words = cast(Tensor, self.word_embeddings(input_ids))
        pos_ids = self.position_ids[:, :T]
        positions = cast(Tensor, self.position_embeddings(pos_ids))

        if token_type_ids is None:
            token_type_ids = lucid.zeros((B, T), device=dev).long()
        types = cast(Tensor, self.token_type_embeddings(token_type_ids))

        emb = words + positions + types
        emb = cast(Tensor, self.LayerNorm(emb))
        return cast(Tensor, self.dropout(emb))


# ─────────────────────────────────────────────────────────────────────────────
# Multi-head self-attention
# ─────────────────────────────────────────────────────────────────────────────


class _BERTSelfAttention(nn.Module):
    """Multi-head self-attention with separate Q / K / V projections.

    HF stores Q/K/V as three independent ``Linear``s (not fused), which is the
    convention we mirror here so weight porting is a direct rename.
    """

    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = math.sqrt(self.head_dim)

        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.dropout = nn.Dropout(p=config.attention_dropout)

    def _shape(self, x: Tensor, B: int, T: int) -> Tensor:
        # (B, T, hidden) → (B, H, T, head_dim)
        return x.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(  # type: ignore[override]
        self,
        hidden: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        B, T, _ = hidden.shape
        q = self._shape(cast(Tensor, self.query(hidden)), B, T)
        k = self._shape(cast(Tensor, self.key(hidden)), B, T)
        v = self._shape(cast(Tensor, self.value(hidden)), B, T)

        # (B, H, T, T)
        scores: Tensor = q @ k.permute(0, 1, 3, 2) / self.scale
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = F.softmax(scores, dim=-1)
        probs = cast(Tensor, self.dropout(probs))

        # (B, H, T, D) → (B, T, H*D)
        ctx: Tensor = probs @ v
        ctx = ctx.permute(0, 2, 1, 3).reshape(B, T, self.num_heads * self.head_dim)
        return ctx


class _BERTSelfOutput(nn.Module):
    """Post-attention dense + LN + residual."""

    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

    def forward(  # type: ignore[override]
        self, hidden: Tensor, input_tensor: Tensor
    ) -> Tensor:
        h = cast(Tensor, self.dropout(cast(Tensor, self.dense(hidden))))
        return cast(Tensor, self.LayerNorm(h + input_tensor))


class _BERTAttention(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        # HF names ``self`` for the projection block — keep the same key.
        self.self = _BERTSelfAttention(config)
        self.output = _BERTSelfOutput(config)

    def forward(  # type: ignore[override]
        self,
        hidden: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        attn_out = cast(Tensor, self.self(hidden, attention_mask=attention_mask))
        return cast(Tensor, self.output(attn_out, hidden))


# ─────────────────────────────────────────────────────────────────────────────
# Feed-forward block
# ─────────────────────────────────────────────────────────────────────────────


class _BERTIntermediate(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self._act_name = config.hidden_act

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return text_activation(self._act_name, cast(Tensor, self.dense(x)))


class _BERTOutput(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

    def forward(  # type: ignore[override]
        self, hidden: Tensor, input_tensor: Tensor
    ) -> Tensor:
        h = cast(Tensor, self.dropout(cast(Tensor, self.dense(hidden))))
        return cast(Tensor, self.LayerNorm(h + input_tensor))


# ─────────────────────────────────────────────────────────────────────────────
# Transformer layer + encoder stack
# ─────────────────────────────────────────────────────────────────────────────


class _BERTLayer(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.attention = _BERTAttention(config)
        self.intermediate = _BERTIntermediate(config)
        self.output = _BERTOutput(config)

    def forward(  # type: ignore[override]
        self,
        hidden: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        attn_out = cast(Tensor, self.attention(hidden, attention_mask=attention_mask))
        inter = cast(Tensor, self.intermediate(attn_out))
        return cast(Tensor, self.output(inter, attn_out))


class _BERTEncoder(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            [_BERTLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(  # type: ignore[override]
        self,
        hidden: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layer:
            hidden = cast(Tensor, layer(hidden, attention_mask=attention_mask))
        return hidden


# ─────────────────────────────────────────────────────────────────────────────
# Pooler — first-token tanh projection feeding sentence-level heads
# ─────────────────────────────────────────────────────────────────────────────


class _BERTPooler(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden: Tensor) -> Tensor:  # type: ignore[override]
        # CLS token is position 0 by tokenization convention.
        cls_hidden = hidden[:, 0]
        return F.tanh(cast(Tensor, self.dense(cls_hidden)))


# ─────────────────────────────────────────────────────────────────────────────
# Top-level encoder model
# ─────────────────────────────────────────────────────────────────────────────


class BERTModel(PretrainedModel):
    r"""Bare BERT encoder returning hidden states and pooled CLS embedding.

    Implements the bidirectional transformer encoder of Devlin et al., 2018.
    Token, position, and segment embeddings are summed, LayerNormed, and
    dropout-regularised, then passed through :math:`L` transformer blocks of
    multi-head self-attention plus position-wise feed-forward.  A single
    tanh-activated linear ("pooler") on the first ``[CLS]`` token produces a
    sentence-level embedding used by classification heads.

    Use this class as the trunk when you want raw hidden states; the
    task-specific subclasses (``BERTFor*``) wrap it with appropriate heads.

    Parameters
    ----------
    config : BERTConfig
        Hyperparameters controlling vocabulary, depth, width, head count, and
        regularisation.  See :class:`BERTConfig` for the full field list.

    Attributes
    ----------
    embeddings : nn.Module
        Token + position + token-type embedding block followed by LayerNorm
        and dropout.
    encoder : nn.Module
        Stack of ``config.num_hidden_layers`` transformer encoder layers.
    pooler : nn.Module
        Dense + tanh projection of the ``[CLS]`` hidden state.
    config_class : type[BERTConfig]
        Class-level pointer used by the registry to instantiate a matching
        config from disk.
    base_model_prefix : str
        Prefix (``"bert"``) under which sub-module checkpoints are nested in
        task-head variants — used during weight loading.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805).

    Self-attention follows the scaled dot-product form

    .. math::

        \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(
            \frac{Q K^{\top}}{\sqrt{d_k}}
        \right) V

    with :math:`d_k = H / A`.  Each layer applies multi-head attention,
    followed by a feed-forward block

    .. math::

        \mathrm{FFN}(x) = \mathrm{GELU}(x W_1 + b_1) W_2 + b_2,

    each wrapped by a residual connection and post-LayerNorm.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import BERTConfig, BERTModel
    >>> cfg = BERTConfig(num_hidden_layers=2, hidden_size=128, num_attention_heads=2,
    ...                  intermediate_size=512)
    >>> model = BERTModel(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 2088, 102]])   # [CLS] hello world [SEP]
    >>> out = model(input_ids)
    >>> out.last_hidden_state.shape   # (B=1, T=4, H=128)
    (1, 4, 128)
    >>> out.pooler_output.shape       # (B=1, H=128)
    (1, 128)
    """

    config_class: ClassVar[type[BERTConfig]] = BERTConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BERTConfig) -> None:
        super().__init__(config)
        self.embeddings = _BERTEmbeddings(config)
        self.encoder = _BERTEncoder(config)
        self.pooler = _BERTPooler(config)

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, nn.Embedding):
            raise TypeError(
                f"BERTModel input embeddings must be nn.Embedding, got {type(value).__name__}"
            )
        self.embeddings.word_embeddings = value

    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
    ) -> BaseModelOutputWithPooling:
        B, T = int(input_ids.shape[0]), int(input_ids.shape[1])

        ext_mask = extended_attention_mask(attention_mask, (B, T))

        hidden = cast(Tensor, self.embeddings(input_ids, token_type_ids=token_type_ids))
        sequence_output = cast(Tensor, self.encoder(hidden, attention_mask=ext_mask))
        pooled_output = cast(Tensor, self.pooler(sequence_output))

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MLM head — used by BERTForMaskedLM
# ─────────────────────────────────────────────────────────────────────────────


class _BERTPredictionHeadTransform(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self._act_name = config.hidden_act

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.dense(x))
        x = text_activation(self._act_name, x)
        return cast(Tensor, self.LayerNorm(x))


class _BERTLMPredictionHead(nn.Module):
    """Decoder linear (weight tied to input embeddings) + standalone bias."""

    bias: Tensor

    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.transform = _BERTPredictionHeadTransform(config)
        # Decoder is created untied; ``BERTForMaskedLM`` re-binds the weight to
        # the input embedding table when ``tie_word_embeddings`` is set.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # HF keeps the output bias as a standalone parameter on the head, not on
        # the Linear — so checkpoint keys are ``cls.predictions.bias`` /
        # ``cls.predictions.decoder.weight``.
        self.bias = nn.Parameter(lucid.zeros(config.vocab_size))

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.transform(x))
        logits = cast(Tensor, self.decoder(x))
        return logits + self.bias


class _BERTOnlyMLMHead(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.predictions = _BERTLMPredictionHead(config)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.predictions(x))


# ─────────────────────────────────────────────────────────────────────────────
# Task heads
# ─────────────────────────────────────────────────────────────────────────────


class BERTForMaskedLM(PretrainedModel, MaskedLMMixin):
    r"""BERT with a tied masked-language-modeling head.

    Implements the masked-LM half of the Devlin et al. (2018) pre-training
    objective.  A two-layer projection (dense + GELU + LayerNorm) maps each
    hidden state to vocabulary logits via a decoder whose weight matrix is
    tied to the input ``word_embeddings`` table when
    ``config.tie_word_embeddings`` is True.  Use for pre-training from
    scratch, continued pre-training on domain corpora, or fill-in-the-blank
    inference.

    Parameters
    ----------
    config : BERTConfig
        BERT hyperparameters.  ``config.tie_word_embeddings`` (default True)
        controls whether the decoder weight is bound to the input embedding
        matrix to halve the parameter count.

    Attributes
    ----------
    bert : BERTModel
        Underlying bidirectional encoder trunk.
    cls : nn.Module
        Masked-LM prediction head with its own dense + LayerNorm transform
        and an output decoder of shape ``(hidden_size, vocab_size)``.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805), section 3.1 Task #1.

    When ``labels`` is supplied the head computes

    .. math::

        \mathcal{L}_{\mathrm{MLM}} = -\frac{1}{|M|}
            \sum_{i \in M} \log p_{\theta}(x_i \mid x_{\setminus M})

    over the set :math:`M` of masked positions, with positions where the
    label equals ``-100`` excluded from the sum.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import BERTConfig, BERTForMaskedLM
    >>> cfg = BERTConfig(num_hidden_layers=2, hidden_size=128,
    ...                  num_attention_heads=2, intermediate_size=512)
    >>> model = BERTForMaskedLM(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 103, 102]])   # [CLS] hello [MASK] [SEP]
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, T=4, V=30522)
    (1, 4, 30522)
    """

    config_class: ClassVar[type[BERTConfig]] = BERTConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BERTConfig) -> None:
        super().__init__(config)
        self.bert = BERTModel(config)
        self.cls = _BERTOnlyMLMHead(config)
        if config.tie_word_embeddings:
            self._tie_decoder_to_input_embeddings()

    def _tie_decoder_to_input_embeddings(self) -> None:
        # Bind the decoder weight to the input embedding matrix so the two
        # share storage and gradients.  HF does the same thing.
        self.cls.predictions.decoder.weight = (
            self.bert.embeddings.word_embeddings.weight
        )

    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> MaskedLMOutput:
        outputs = cast(
            BaseModelOutputWithPooling,
            self.bert(
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


class BERTForSequenceClassification(PretrainedModel):
    r"""BERT with a pooled-CLS linear classifier for sequence-level tasks.

    Wraps the bidirectional encoder with a dropout-regularised linear head
    operating on the ``[CLS]`` pooled embedding.  This is the standard
    fine-tuning recipe for GLUE-style sentence/sentence-pair tasks (SST-2,
    MNLI, QQP, RTE, ...) introduced in Devlin et al., 2018 §4.1.

    Parameters
    ----------
    config : BERTConfig
        BERT hyperparameters.  ``config.num_labels`` sets the output
        dimension; ``config.classifier_dropout`` (falling back to
        ``hidden_dropout``) sets the dropout applied before the linear.

    Attributes
    ----------
    bert : BERTModel
        Underlying bidirectional encoder trunk.
    dropout : nn.Dropout
        Dropout layer applied to the pooled ``[CLS]`` embedding.
    classifier : nn.Linear
        Final linear of shape ``(hidden_size, num_labels)`` producing logits.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805).

    The pooled representation is
    :math:`p = \tanh(W_{\mathrm{pool}}\, h_{[\mathrm{CLS}]} + b_{\mathrm{pool}})`,
    and the final logits are :math:`z = W_{\mathrm{cls}}\,\mathrm{Dropout}(p) + b_{\mathrm{cls}}`.
    When ``labels`` is provided, cross-entropy over ``num_labels`` classes is
    computed and exposed as ``output.loss``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import BERTConfig, BERTForSequenceClassification
    >>> cfg = BERTConfig(num_labels=3, num_hidden_layers=2, hidden_size=128,
    ...                  num_attention_heads=2, intermediate_size=512)
    >>> model = BERTForSequenceClassification(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, num_labels=3)
    (1, 3)
    """

    config_class: ClassVar[type[BERTConfig]] = BERTConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BERTConfig) -> None:
        super().__init__(config)
        self.bert = BERTModel(config)
        drop = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout
        )
        self.dropout = nn.Dropout(p=drop)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> MaskedLMOutput:
        outputs = cast(
            BaseModelOutputWithPooling,
            self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ),
        )
        pooled = cast(Tensor, self.dropout(outputs.pooler_output))
        logits = cast(Tensor, self.classifier(pooled))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        # We reuse MaskedLMOutput here since it only carries logits + loss; a
        # dedicated SequenceClassificationOutput can be added later if any
        # caller actually needs the extra metadata.
        return MaskedLMOutput(logits=logits, loss=loss)


class BERTForTokenClassification(PretrainedModel, MaskedLMMixin):
    r"""BERT with a per-token linear classifier for tagging tasks.

    Wraps the bidirectional encoder with a dropout-regularised linear head
    applied independently at every sequence position.  Used for token-level
    fine-tunes such as named-entity recognition (CoNLL-2003), part-of-speech
    tagging, and chunking — see Devlin et al., 2018 §4.3.

    Parameters
    ----------
    config : BERTConfig
        BERT hyperparameters.  ``config.num_labels`` sets the per-position
        output dimension; ``config.classifier_dropout`` (falling back to
        ``hidden_dropout``) sets the dropout applied before the linear.

    Attributes
    ----------
    bert : BERTModel
        Underlying bidirectional encoder trunk.
    dropout : nn.Dropout
        Dropout applied to the full sequence hidden states.
    classifier : nn.Linear
        Final linear of shape ``(hidden_size, num_labels)`` mapping each
        token's hidden state to per-class logits.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805).

    The loss (when ``labels`` is provided) is the masked cross-entropy
    inherited from :class:`MaskedLMMixin`:

    .. math::

        \mathcal{L} = -\frac{1}{|V|}
            \sum_{(b, t) \in V} \log p_{\theta}\!\left(y_{b,t} \mid x_b\right),

    where :math:`V` is the set of positions with ``label != -100``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import BERTConfig, BERTForTokenClassification
    >>> cfg = BERTConfig(num_labels=9, num_hidden_layers=2, hidden_size=128,
    ...                  num_attention_heads=2, intermediate_size=512)
    >>> model = BERTForTokenClassification(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 2088, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, T=4, num_labels=9)
    (1, 4, 9)
    """

    config_class: ClassVar[type[BERTConfig]] = BERTConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BERTConfig) -> None:
        super().__init__(config)
        self.bert = BERTModel(config)
        drop = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout
        )
        self.dropout = nn.Dropout(p=drop)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> MaskedLMOutput:
        outputs = cast(
            BaseModelOutputWithPooling,
            self.bert(
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


class BERTForQuestionAnswering(PretrainedModel):
    r"""BERT with a 2-way span head for extractive question answering.

    Wraps the bidirectional encoder with a single linear of output width 2,
    producing start- and end-position logits over each token in the input.
    This is the SQuAD v1.1 / v2.0 fine-tuning recipe of Devlin et al., 2018
    §4.2 — given a ``(question, context)`` pair concatenated with ``[SEP]``,
    the model predicts the answer span inside the context.

    Parameters
    ----------
    config : BERTConfig
        BERT hyperparameters.  The QA head is always 2-way; ``num_labels``
        is ignored here.

    Attributes
    ----------
    bert : BERTModel
        Underlying bidirectional encoder trunk.
    qa_outputs : nn.Linear
        Final linear of shape ``(hidden_size, 2)`` mapping each token's
        hidden state to ``(start_logit, end_logit)``.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805).

    When both ``start_positions`` and ``end_positions`` are provided, the
    loss is the symmetric average of two cross-entropies:

    .. math::

        \mathcal{L} = \tfrac{1}{2}\!\left(
            \mathrm{CE}(z^{\mathrm{start}}, y^{\mathrm{start}})
          + \mathrm{CE}(z^{\mathrm{end}},   y^{\mathrm{end}})
        \right).

    The returned ``logits`` tensor has shape ``(B, T, 2)``; index ``[..., 0]``
    for start scores and ``[..., 1]`` for end scores.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import BERTConfig, BERTForQuestionAnswering
    >>> cfg = BERTConfig(num_hidden_layers=2, hidden_size=128,
    ...                  num_attention_heads=2, intermediate_size=512)
    >>> model = BERTForQuestionAnswering(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 2040, 2003, 102, 1045, 2572, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, T=7, 2)
    (1, 7, 2)
    """

    config_class: ClassVar[type[BERTConfig]] = BERTConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BERTConfig) -> None:
        super().__init__(config)
        self.bert = BERTModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

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
            self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ),
        )
        logits = cast(Tensor, self.qa_outputs(outputs.last_hidden_state))
        # ``logits`` is (B, T, 2); split along the last dim into start / end.
        # We stack them along a new dim so downstream code can index ``[..., 0]``
        # for start and ``[..., 1]`` for end while keeping a single return.

        loss: Tensor | None = None
        if start_positions is not None and end_positions is not None:
            start_logits = logits[..., 0]  # (B, T)
            end_logits = logits[..., 1]  # (B, T)
            # Callers are responsible for keeping span positions inside [0, T).
            loss = (
                F.cross_entropy(start_logits, start_positions.long())
                + F.cross_entropy(end_logits, end_positions.long())
            ) / 2.0

        return MaskedLMOutput(logits=logits, loss=loss)


# ─────────────────────────────────────────────────────────────────────────────
# Additional task heads — pre-training, NSP, causal-LM
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BERTForPreTrainingOutput(ModelOutput):
    r"""Combined output for :class:`BERTForPreTraining`.

    Aggregates the masked-LM logits, next-sentence-prediction logits, and
    optional per-objective and combined losses produced by the full BERT
    pre-training pipeline of Devlin et al., 2018.

    Parameters
    ----------
    prediction_logits : Tensor
        MLM head logits of shape ``(B, T, vocab_size)`` — one distribution
        over the WordPiece vocabulary per input position.
    seq_relationship_logits : Tensor
        NSP head logits of shape ``(B, 2)`` — binary IsNext / NotNext scores
        derived from the pooled ``[CLS]`` embedding.
    loss : Tensor or None, default=None
        Sum of ``mlm_loss`` and ``nsp_loss`` when both are available;
        otherwise the single available loss, or ``None`` if neither label
        set was supplied.
    mlm_loss : Tensor or None, default=None
        Cross-entropy on masked positions when ``labels`` was supplied.
    nsp_loss : Tensor or None, default=None
        Binary cross-entropy on the NSP head when ``next_sentence_label``
        was supplied.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805) §3.1.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import BERTConfig, BERTForPreTraining
    >>> cfg = BERTConfig(num_hidden_layers=2, hidden_size=128,
    ...                  num_attention_heads=2, intermediate_size=512)
    >>> model = BERTForPreTraining(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 102, 2088, 102]])
    >>> out = model(input_ids)
    >>> out.prediction_logits.shape, out.seq_relationship_logits.shape
    ((1, 5, 30522), (1, 2))
    """

    prediction_logits: Tensor
    seq_relationship_logits: Tensor
    loss: Tensor | None = None
    mlm_loss: Tensor | None = None
    nsp_loss: Tensor | None = None


class _BERTOnlyNSPHead(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.seq_relationship(pooled_output))


class _BERTPreTrainingHeads(nn.Module):
    """MLM prediction head + NSP head — used by :class:`BERTForPreTraining`."""

    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.predictions = _BERTLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(  # type: ignore[override]
        self, sequence_output: Tensor, pooled_output: Tensor
    ) -> tuple[Tensor, Tensor]:
        prediction_scores = cast(Tensor, self.predictions(sequence_output))
        seq_relationship_score = cast(Tensor, self.seq_relationship(pooled_output))
        return prediction_scores, seq_relationship_score


class BERTForPreTraining(PretrainedModel, MaskedLMMixin):
    r"""BERT with the original joint MLM + NSP pre-training objective.

    Combines the masked-language-modeling head (decoder weight tied to input
    embeddings when ``config.tie_word_embeddings`` is True) with the
    next-sentence-prediction head on top of the pooled ``[CLS]`` embedding.
    This is the exact head configuration used in Devlin et al., 2018 to
    train BERT-Base and BERT-Large from scratch.

    Supply ``labels`` (MLM targets) and/or ``next_sentence_label`` (binary
    NSP target) to compute the corresponding losses; their sum is exposed as
    ``output.loss``.  Use this class only when reproducing the original
    pre-training recipe — newer encoder-only LMs typically drop NSP and use
    :class:`BERTForMaskedLM` directly.

    Parameters
    ----------
    config : BERTConfig
        BERT hyperparameters.  ``config.tie_word_embeddings`` (default True)
        controls whether the MLM decoder weight is tied to the input
        embedding matrix.

    Attributes
    ----------
    bert : BERTModel
        Underlying bidirectional encoder trunk.
    cls : nn.Module
        Combined head holding both the MLM prediction projection
        (``cls.predictions``) and the NSP binary linear
        (``cls.seq_relationship``).

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805) §3.1.

    The combined loss when both objectives are supplied is

    .. math::

        \mathcal{L}_{\mathrm{pretrain}}
            = \mathcal{L}_{\mathrm{MLM}} + \mathcal{L}_{\mathrm{NSP}}.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import BERTConfig, BERTForPreTraining
    >>> cfg = BERTConfig(num_hidden_layers=2, hidden_size=128,
    ...                  num_attention_heads=2, intermediate_size=512)
    >>> model = BERTForPreTraining(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 103, 102, 2088, 102]])
    >>> out = model(input_ids)
    >>> out.prediction_logits.shape    # MLM logits  (B=1, T=6, V=30522)
    (1, 6, 30522)
    >>> out.seq_relationship_logits.shape   # NSP logits  (B=1, 2)
    (1, 2)
    """

    config_class: ClassVar[type[BERTConfig]] = BERTConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BERTConfig) -> None:
        super().__init__(config)
        self.bert = BERTModel(config)
        self.cls = _BERTPreTrainingHeads(config)
        if config.tie_word_embeddings:
            self.cls.predictions.decoder.weight = (
                self.bert.embeddings.word_embeddings.weight
            )

    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
        next_sentence_label: Tensor | None = None,
    ) -> BERTForPreTrainingOutput:
        outputs = cast(
            BaseModelOutputWithPooling,
            self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ),
        )
        prediction_scores, seq_relationship_score = self.cls(
            outputs.last_hidden_state, outputs.pooler_output
        )

        mlm_loss: Tensor | None = None
        nsp_loss: Tensor | None = None
        total_loss: Tensor | None = None
        if labels is not None:
            mlm_loss = self.compute_lm_loss(prediction_scores, labels)
        if next_sentence_label is not None:
            nsp_loss = F.cross_entropy(
                seq_relationship_score, next_sentence_label.long()
            )
        if mlm_loss is not None and nsp_loss is not None:
            total_loss = mlm_loss + nsp_loss
        elif mlm_loss is not None:
            total_loss = mlm_loss
        elif nsp_loss is not None:
            total_loss = nsp_loss

        return BERTForPreTrainingOutput(
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            loss=total_loss,
            mlm_loss=mlm_loss,
            nsp_loss=nsp_loss,
        )


class BERTForNextSentencePrediction(PretrainedModel):
    r"""BERT with the standalone next-sentence-prediction head.

    Wraps the bidirectional encoder with a single binary linear classifier
    operating on the pooled ``[CLS]`` embedding.  This is pre-training task 2
    of Devlin et al., 2018 §3.1 in isolation — useful for reproducing
    historical experiments or as a sanity check for sentence-pair coherence.

    NSP was abandoned by RoBERTa, ALBERT, and DeBERTa as offering no
    downstream value, so prefer :class:`BERTForMaskedLM` (MLM-only) or
    :class:`BERTForSequenceClassification` for new work.

    Parameters
    ----------
    config : BERTConfig
        BERT hyperparameters.

    Attributes
    ----------
    bert : BERTModel
        Underlying bidirectional encoder trunk.
    cls : nn.Module
        NSP head — a single ``Linear(hidden_size, 2)`` over the pooled
        embedding.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805) §3.1 (task #2).

    When ``labels`` is provided, the loss is the binary cross-entropy

    .. math::

        \mathcal{L}_{\mathrm{NSP}}
            = -\frac{1}{B}\sum_{b=1}^{B}
              \log p_{\theta}(y_b \mid x_b^{(A)}, x_b^{(B)}),

    where :math:`y_b \in \{0, 1\}` denotes IsNext vs. NotNext.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import BERTConfig, BERTForNextSentencePrediction
    >>> cfg = BERTConfig(num_hidden_layers=2, hidden_size=128,
    ...                  num_attention_heads=2, intermediate_size=512)
    >>> model = BERTForNextSentencePrediction(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 102, 2088, 102]])
    >>> token_type_ids = lucid.tensor([[0, 0, 0, 1, 1]])
    >>> out = model(input_ids, token_type_ids=token_type_ids)
    >>> out.logits.shape   # (B=1, 2)
    (1, 2)
    """

    config_class: ClassVar[type[BERTConfig]] = BERTConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BERTConfig) -> None:
        super().__init__(config)
        self.bert = BERTModel(config)
        self.cls = _BERTOnlyNSPHead(config)

    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> MaskedLMOutput:
        outputs = cast(
            BaseModelOutputWithPooling,
            self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ),
        )
        seq_relationship_score = cast(Tensor, self.cls(outputs.pooler_output))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(seq_relationship_score, labels.long())

        return MaskedLMOutput(logits=seq_relationship_score, loss=loss)


class BERTForCausalLM(PretrainedModel):
    r"""BERT trunk repurposed as a left-to-right (causal) language model.

    Standard BERT attends bidirectionally; this wrapper injects a
    lower-triangular causal mask on top of the existing additive
    attention/padding mask so the same encoder weights behave as a decoder.
    The LM head is the same tied projection used by
    :class:`BERTForMaskedLM`.  Use this class when you want to apply
    pre-trained BERT weights to a generative or sequence-continuation task.

    Parameters
    ----------
    config : BERTConfig
        BERT hyperparameters.  ``config.tie_word_embeddings`` (default True)
        ties the LM decoder weight to the input embedding matrix.

    Attributes
    ----------
    bert : BERTModel
        Underlying transformer trunk; only ``embeddings`` and ``encoder`` are
        invoked in ``forward`` (the pooler is bypassed).
    cls : nn.Module
        Tied LM prediction head — same architecture as
        :class:`BERTForMaskedLM`.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805); causal adaptation follows the standard left-to-right
    LM masking scheme.

    The additive causal mask :math:`M \in \mathbb{R}^{T \times T}` satisfies

    .. math::

        M_{ij} =
        \begin{cases}
            0, & j \le i \\
            -10^{4}, & j > i
        \end{cases}

    and is broadcast against a padding mask when present.  Loss (when
    ``labels`` is supplied) uses the standard next-token shift:

    .. math::

        \mathcal{L}_{\mathrm{CLM}}
            = -\frac{1}{B(T-1)} \sum_{b,t}
              \log p_{\theta}(y_{b, t+1} \mid x_{b,\le t}),

    with positions labelled ``-100`` excluded.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import BERTConfig, BERTForCausalLM
    >>> cfg = BERTConfig(num_hidden_layers=2, hidden_size=128,
    ...                  num_attention_heads=2, intermediate_size=512)
    >>> model = BERTForCausalLM(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 2088, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, T=4, V=30522)
    (1, 4, 30522)
    """

    config_class: ClassVar[type[BERTConfig]] = BERTConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BERTConfig) -> None:
        super().__init__(config)
        self.bert = BERTModel(config)
        self.cls = _BERTOnlyMLMHead(config)
        if config.tie_word_embeddings:
            self.cls.predictions.decoder.weight = (
                self.bert.embeddings.word_embeddings.weight
            )

    def _causal_attention_mask(
        self,
        attention_mask: Tensor | None,
        B: int,
        T: int,
        device: str,
    ) -> Tensor:
        """Combine a ``(B, T)`` padding mask with a lower-triangular causal
        mask, returning the additive ``(B, 1, T, T)`` form.
        """
        causal = lucid.tril(lucid.ones((T, T), device=device))  # (T, T)
        causal_add = (1.0 - causal) * -1e4  # 0 / -1e4
        causal_add = causal_add.reshape(1, 1, T, T)
        if attention_mask is None:
            return causal_add
        pad_add = (1.0 - attention_mask.float()) * -1e4  # (B, T)
        pad_add = pad_add.reshape(B, 1, 1, T)
        return causal_add + pad_add

    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> CausalLMOutput:
        B, T = int(input_ids.shape[0]), int(input_ids.shape[1])
        dev = input_ids.device.type
        ext_mask = self._causal_attention_mask(attention_mask, B, T, dev)

        hidden = cast(
            Tensor, self.bert.embeddings(input_ids, token_type_ids=token_type_ids)
        )
        sequence_output = cast(
            Tensor, self.bert.encoder(hidden, attention_mask=ext_mask)
        )
        prediction_scores = cast(Tensor, self.cls(sequence_output))

        loss: Tensor | None = None
        if labels is not None:
            # Standard causal-LM shift: predict token t+1 from positions [0..t].
            B_, T_, V = prediction_scores.shape
            shift_logits = prediction_scores[:, :-1, :].reshape(B_ * (T_ - 1), V)
            shift_labels = labels[:, 1:].reshape(B_ * (T_ - 1)).long()
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        return CausalLMOutput(logits=prediction_scores, loss=loss)
