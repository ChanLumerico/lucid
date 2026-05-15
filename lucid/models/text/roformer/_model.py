"""RoFormer model (Su et al., 2021) — BERT with rotary position embedding.

Architecturally identical to :class:`lucid.models.text.bert.BertModel` except
that:

    * No additive position embedding (``position_embeddings`` is removed).
    * Inside every self-attention layer, ``q`` and ``k`` are rotated by
      :func:`lucid.models.text.apply_rotary_emb` before the dot product.

State-dict key naming mirrors HF ``RoFormerModel`` so future weight ports
amount to a flat rename.
"""

import math
from typing import ClassVar, cast

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
from lucid.nn import RotaryEmbedding
from lucid.nn.functional import apply_rotary_emb
from lucid.models.text.roformer._config import RoFormerConfig

# ─────────────────────────────────────────────────────────────────────────────
# Embeddings — *no* position embedding (RoPE handles position implicitly)
# ─────────────────────────────────────────────────────────────────────────────


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
        q, k = apply_rotary_emb(q, k, cos, sin)

        scores: Tensor = q @ k.permute(0, 1, 3, 2) / self.scale
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = cast(Tensor, self.dropout(F.softmax(scores, dim=-1)))

        ctx: Tensor = probs @ v
        return ctx.permute(0, 2, 1, 3).reshape(B, T, self.num_heads * self.head_dim)


class _RoFormerSelfOutput(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

    def forward(  # type: ignore[override]
        self, hidden: Tensor, input_tensor: Tensor
    ) -> Tensor:
        h = cast(Tensor, self.dropout(cast(Tensor, self.dense(hidden))))
        return cast(Tensor, self.LayerNorm(h + input_tensor))


class _RoFormerAttention(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.self = _RoFormerSelfAttention(config)
        self.output = _RoFormerSelfOutput(config)

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


class _RoFormerIntermediate(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self._act_name = config.hidden_act

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return text_activation(self._act_name, cast(Tensor, self.dense(x)))


class _RoFormerOutput(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

    def forward(  # type: ignore[override]
        self, hidden: Tensor, input_tensor: Tensor
    ) -> Tensor:
        h = cast(Tensor, self.dropout(cast(Tensor, self.dense(hidden))))
        return cast(Tensor, self.LayerNorm(h + input_tensor))


class _RoFormerLayer(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.attention = _RoFormerAttention(config)
        self.intermediate = _RoFormerIntermediate(config)
        self.output = _RoFormerOutput(config)

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


class _RoFormerEncoder(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            [_RoFormerLayer(config) for _ in range(config.num_hidden_layers)]
        )

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


class _RoFormerPooler(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden: Tensor) -> Tensor:  # type: ignore[override]
        cls_hidden = hidden[:, 0]
        return F.tanh(cast(Tensor, self.dense(cls_hidden)))


# ─────────────────────────────────────────────────────────────────────────────
# Top-level encoder + heads
# ─────────────────────────────────────────────────────────────────────────────


class RoFormerModel(PretrainedModel):
    """RoFormer encoder trunk."""

    config_class: ClassVar[type[RoFormerConfig]] = RoFormerConfig
    base_model_prefix: ClassVar[str] = "roformer"

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__(config)
        head_dim = config.hidden_size // config.num_attention_heads
        self._max_pos = config.max_position_embeddings
        self.embeddings = _RoFormerEmbeddings(config)
        self.rotary = RotaryEmbedding(
            head_dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rotary_base,
        )
        self.encoder = _RoFormerEncoder(config)
        self.pooler = _RoFormerPooler(config)

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, nn.Embedding):
            raise TypeError(
                "RoFormerModel input embeddings must be nn.Embedding, got "
                f"{type(value).__name__}"
            )
        self.embeddings.word_embeddings = value

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


class _RoFormerPredictionHeadTransform(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self._act_name = config.hidden_act

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.dense(x))
        x = text_activation(self._act_name, x)
        return cast(Tensor, self.LayerNorm(x))


class _RoFormerLMPredictionHead(nn.Module):
    bias: Tensor

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.transform = _RoFormerPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(lucid.zeros(config.vocab_size))

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.transform(x))
        logits = cast(Tensor, self.decoder(x))
        return logits + self.bias


class _RoFormerOnlyMLMHead(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.predictions = _RoFormerLMPredictionHead(config)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.predictions(x))


class RoFormerForMaskedLM(PretrainedModel, MaskedLMMixin):
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
    """RoFormer + per-choice CLS classifier — RACE / SWAG-style fine-tunes.

    Input ``(N, C, L)`` flattens to ``(N*C, L)`` for the encoder; the pooled
    CLS embedding of each choice is projected to a scalar and the resulting
    ``(N, C)`` matrix is the per-choice logit.
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
    """RoFormer + SQuAD-style span (start / end) head.

    Identical contract to :class:`BertForQuestionAnswering` — output logits
    are ``(B, T, 2)`` where ``[..., 0]`` are start scores and ``[..., 1]``
    are end scores; supply ``start_positions`` + ``end_positions`` to get
    the averaged span loss.
    """

    config_class: ClassVar[type[RoFormerConfig]] = RoFormerConfig
    base_model_prefix: ClassVar[str] = "roformer"

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__(config)
        self.roformer = RoFormerModel(config)
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
