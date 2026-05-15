"""BERT model (Devlin et al., 2018) — encoder-only Transformer.

Module / parameter naming matches HuggingFace Transformers' ``BertModel`` so
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
from lucid.models.text.bert._config import BertConfig

# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────


class _BertEmbeddings(nn.Module):
    """Word + position + token-type embedding sum, then LN + Dropout."""

    position_ids: Tensor

    def __init__(self, config: BertConfig) -> None:
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


class _BertSelfAttention(nn.Module):
    """Multi-head self-attention with separate Q / K / V projections.

    HF stores Q/K/V as three independent ``Linear``s (not fused), which is the
    convention we mirror here so weight porting is a direct rename.
    """

    def __init__(self, config: BertConfig) -> None:
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


class _BertSelfOutput(nn.Module):
    """Post-attention dense + LN + residual."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

    def forward(  # type: ignore[override]
        self, hidden: Tensor, input_tensor: Tensor
    ) -> Tensor:
        h = cast(Tensor, self.dropout(cast(Tensor, self.dense(hidden))))
        return cast(Tensor, self.LayerNorm(h + input_tensor))


class _BertAttention(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        # HF names ``self`` for the projection block — keep the same key.
        self.self = _BertSelfAttention(config)
        self.output = _BertSelfOutput(config)

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


class _BertIntermediate(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self._act_name = config.hidden_act

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return text_activation(self._act_name, cast(Tensor, self.dense(x)))


class _BertOutput(nn.Module):
    def __init__(self, config: BertConfig) -> None:
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


class _BertLayer(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.attention = _BertAttention(config)
        self.intermediate = _BertIntermediate(config)
        self.output = _BertOutput(config)

    def forward(  # type: ignore[override]
        self,
        hidden: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        attn_out = cast(Tensor, self.attention(hidden, attention_mask=attention_mask))
        inter = cast(Tensor, self.intermediate(attn_out))
        return cast(Tensor, self.output(inter, attn_out))


class _BertEncoder(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            [_BertLayer(config) for _ in range(config.num_hidden_layers)]
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


class _BertPooler(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden: Tensor) -> Tensor:  # type: ignore[override]
        # CLS token is position 0 by tokenization convention.
        cls_hidden = hidden[:, 0]
        return F.tanh(cast(Tensor, self.dense(cls_hidden)))


# ─────────────────────────────────────────────────────────────────────────────
# Top-level encoder model
# ─────────────────────────────────────────────────────────────────────────────


class BertModel(PretrainedModel):
    """Bare BERT encoder returning hidden states + pooled CLS embedding."""

    config_class: ClassVar[type[BertConfig]] = BertConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.embeddings = _BertEmbeddings(config)
        self.encoder = _BertEncoder(config)
        self.pooler = _BertPooler(config)

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, nn.Embedding):
            raise TypeError(
                f"BertModel input embeddings must be nn.Embedding, got {type(value).__name__}"
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
# MLM head — used by BertForMaskedLM
# ─────────────────────────────────────────────────────────────────────────────


class _BertPredictionHeadTransform(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self._act_name = config.hidden_act

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.dense(x))
        x = text_activation(self._act_name, x)
        return cast(Tensor, self.LayerNorm(x))


class _BertLMPredictionHead(nn.Module):
    """Decoder linear (weight tied to input embeddings) + standalone bias."""

    bias: Tensor

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.transform = _BertPredictionHeadTransform(config)
        # Decoder is created untied; ``BertForMaskedLM`` re-binds the weight to
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


class _BertOnlyMLMHead(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.predictions = _BertLMPredictionHead(config)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.predictions(x))


# ─────────────────────────────────────────────────────────────────────────────
# Task heads
# ─────────────────────────────────────────────────────────────────────────────


class BertForMaskedLM(PretrainedModel, MaskedLMMixin):
    """BERT + tied MLM head (the pre-training objective)."""

    config_class: ClassVar[type[BertConfig]] = BertConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = _BertOnlyMLMHead(config)
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


class BertForSequenceClassification(PretrainedModel):
    """BERT + pooled-CLS linear classifier (GLUE-style fine-tunes)."""

    config_class: ClassVar[type[BertConfig]] = BertConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.bert = BertModel(config)
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


class BertForTokenClassification(PretrainedModel, MaskedLMMixin):
    """BERT + per-token linear classifier (NER, POS tagging)."""

    config_class: ClassVar[type[BertConfig]] = BertConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.bert = BertModel(config)
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


class BertForQuestionAnswering(PretrainedModel):
    """BERT + 2-way linear (start / end span logits) — SQuAD-style fine-tunes."""

    config_class: ClassVar[type[BertConfig]] = BertConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.bert = BertModel(config)
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
class BertForPreTrainingOutput(ModelOutput):
    """Joint MLM + NSP output for :class:`BertForPreTraining`."""

    prediction_logits: Tensor
    seq_relationship_logits: Tensor
    loss: Tensor | None = None
    mlm_loss: Tensor | None = None
    nsp_loss: Tensor | None = None


class _BertOnlyNSPHead(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.seq_relationship(pooled_output))


class _BertPreTrainingHeads(nn.Module):
    """MLM prediction head + NSP head — used by :class:`BertForPreTraining`."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.predictions = _BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(  # type: ignore[override]
        self, sequence_output: Tensor, pooled_output: Tensor
    ) -> tuple[Tensor, Tensor]:
        prediction_scores = cast(Tensor, self.predictions(sequence_output))
        seq_relationship_score = cast(Tensor, self.seq_relationship(pooled_output))
        return prediction_scores, seq_relationship_score


class BertForPreTraining(PretrainedModel, MaskedLMMixin):
    """BERT + the original Devlin et al. pre-training objective.

    Combines the masked-LM head (tied to input embeddings when
    ``config.tie_word_embeddings`` is True) with the next-sentence-prediction
    head.  Supply ``labels`` (MLM targets) and / or ``next_sentence_label``
    (binary NSP target) to compute the corresponding losses; their sum is
    exposed as ``output.loss``.
    """

    config_class: ClassVar[type[BertConfig]] = BertConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = _BertPreTrainingHeads(config)
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
    ) -> BertForPreTrainingOutput:
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

        return BertForPreTrainingOutput(
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            loss=total_loss,
            mlm_loss=mlm_loss,
            nsp_loss=nsp_loss,
        )


class BertForNextSentencePrediction(PretrainedModel):
    """BERT + standalone NSP head (Devlin et al. §3.1 pretraining task 2).

    Note: NSP was abandoned by RoBERTa / ALBERT / DeBERTa as adding no
    downstream value, so this class is kept mostly for parity with the
    original BERT release and historical experiments.
    """

    config_class: ClassVar[type[BertConfig]] = BertConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = _BertOnlyNSPHead(config)

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


class BertForCausalLM(PretrainedModel):
    """BERT trunk used as a left-to-right LM.

    Standard BERT attends bidirectionally; this wrapper injects a causal mask
    on top of the existing additive attention mask so the encoder behaves as
    a decoder.  The LM head is the same tied projection used by
    :class:`BertForMaskedLM`.
    """

    config_class: ClassVar[type[BertConfig]] = BertConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = _BertOnlyMLMHead(config)
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
