import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor

from ._model import BERT, BERTConfig


__all__ = [
    "BERTForPreTraining",
    "BERTForMaskedLM",
    "BERTForCausalLM",
    "BERTForNextSentencePrediction",
    "BERTForSequenceClassification",
    "BERTForTokenClassification",
    "BERTForQuestionAnswering",
    "bert_for_pre_training_base",
    "bert_for_pre_training_large",
    "bert_for_masked_lm_base",
    "bert_for_masked_lm_large",
    "bert_for_causal_lm_base",
    "bert_for_causal_lm_large",
    "bert_for_next_sentence_prediction_base",
    "bert_for_next_sentence_prediction_large",
    "bert_for_sequence_classification_base",
    "bert_for_sequence_classification_large",
    "bert_for_token_classification_base",
    "bert_for_token_classification_large",
    "bert_for_question_answering_base",
    "bert_for_question_answering_large",
]


class _BERTTaskWrapperMixin:
    config: BERTConfig
    bert: BERT

    def get_input_embeddings(self) -> nn.Embedding:
        return self.bert.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.bert.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Linear | None:
        return self.bert.get_output_embeddings()

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.bert.set_output_embeddings(value)

    def tie_weights(self) -> None:
        self.bert.tie_weights()


class _BERTPredictionHeadTransform(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = config.hidden_act
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layernorm(hidden_states)

        return hidden_states


class _BERTLMPredictionHead(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.transform = _BERTPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)

        return hidden_states


class _BERTOnlyMLMHead(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.predictions = _BERTLMPredictionHead(config)

    def forward(self, sequence_output: Tensor) -> Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class _BERTOnlyNSPHead(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output: Tensor) -> Tensor:
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class _BERTPreTrainingHeads(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.predictions = _BERTLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(
        self, sequence_output: Tensor, pooled_output: Tensor
    ) -> tuple[Tensor, Tensor]:
        prediction_score = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)

        return prediction_score, seq_relationship_score


def _classification_accuracy(predictions: Tensor, labels: Tensor) -> Tensor:
    pred_flat = predictions.reshape(-1).astype(lucid.Int)
    label_flat = labels.reshape(-1).astype(lucid.Int)
    return (pred_flat == label_flat).astype(lucid.Float32).mean()


def _masked_token_accuracy(
    predictions: Tensor, labels: Tensor, ignore_index: int = -100
) -> Tensor:
    pred_flat = predictions.reshape(-1).astype(lucid.Int)
    label_flat = labels.reshape(-1).astype(lucid.Int)
    valid_mask = (label_flat != ignore_index).astype(lucid.Int)

    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return Tensor(0.0, device=labels.device)

    correct = ((pred_flat == label_flat).astype(lucid.Int) * valid_mask).sum()
    return correct.astype(lucid.Float32) / float(valid_count)


def _classification_ce_loss(
    logits: Tensor,
    labels: Tensor,
    reduction: str | None = "mean",
) -> Tensor:
    return F.cross_entropy(
        logits,
        labels.reshape(-1).astype(lucid.Int),
        reduction=reduction,
    )


def _token_ce_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = -100,
    reduction: str | None = "mean",
) -> Tensor:
    num_labels = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, num_labels),
        labels.reshape(-1).astype(lucid.Int),
        ignore_index=ignore_index,
        reduction=reduction,
    )


def _predict_labels_from_logits(logits: Tensor) -> Tensor:
    return lucid.argmax(logits, axis=-1)


def _predict_proba_from_logits(logits: Tensor) -> Tensor:
    return F.softmax(logits, axis=-1)


def _shift_causal_lm_labels(
    logits: Tensor, labels: Tensor, shift_labels: bool
) -> tuple[Tensor, Tensor]:
    if not shift_labels:
        return logits, labels

    if logits.shape[1] < 2:
        raise ValueError(
            "shift_labels=True requires sequence length >= 2 "
            f"(got {logits.shape[1]})."
        )
    return logits[:, :-1, :], labels[:, 1:]


def _qa_ce_loss(
    start_logits: Tensor,
    end_logits: Tensor,
    start_positions: Tensor,
    end_positions: Tensor,
    reduction: str | None = "mean",
) -> Tensor:
    start_targets = start_positions.reshape(-1).astype(lucid.Int)
    end_targets = end_positions.reshape(-1).astype(lucid.Int)

    start_loss = F.cross_entropy(start_logits, start_targets, reduction=reduction)
    end_loss = F.cross_entropy(end_logits, end_targets, reduction=reduction)
    return (start_loss + end_loss) / 2.0


class BERTForPreTraining(_BERTTaskWrapperMixin, nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.config = config
        self.bert = BERT(config)
        self.cls = _BERTPreTrainingHeads(config)
        self.cls.apply(self.bert._init_weights)
        self.bert.set_output_embeddings(self.cls.predictions.decoder)

    def forward(
        self,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        sequence_output, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        if pooled_output is None:
            raise RuntimeError("BERTForPreTraining requires pooled output.")
        return self.cls(sequence_output, pooled_output)

    def get_mlm_loss(
        self,
        mlm_labels: Tensor,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
        *,
        ignore_index: int = -100,
        reduction: str | None = "mean",
    ) -> Tensor:
        prediction_scores, _ = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return _token_ce_loss(
            prediction_scores,
            mlm_labels,
            ignore_index=ignore_index,
            reduction=reduction,
        )

    def get_nsp_loss(
        self,
        nsp_labels: Tensor,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
        *,
        reduction: str | None = "mean",
    ) -> Tensor:
        _, seq_relationship_scores = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return _classification_ce_loss(
            seq_relationship_scores,
            nsp_labels,
            reduction=reduction,
        )

    def get_loss(
        self,
        mlm_labels: Tensor,
        nsp_labels: Tensor,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
        *,
        mlm_weight: float = 1.0,
        nsp_weight: float = 1.0,
        ignore_index: int = -100,
        reduction: str | None = "mean",
    ) -> Tensor:
        prediction_scores, seq_relationship_scores = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        mlm_loss = _token_ce_loss(
            prediction_scores,
            mlm_labels,
            ignore_index=ignore_index,
            reduction=reduction,
        )
        nsp_loss = _classification_ce_loss(
            seq_relationship_scores,
            nsp_labels,
            reduction=reduction,
        )
        return (mlm_weight * mlm_loss) + (nsp_weight * nsp_loss)


class BERTForMaskedLM(_BERTTaskWrapperMixin, nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.config = config
        self.bert = BERT(config)
        self.cls = _BERTOnlyMLMHead(config)
        self.cls.apply(self.bert._init_weights)
        self.bert.set_output_embeddings(self.cls.predictions.decoder)

    def forward(
        self,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
    ) -> Tensor:
        sequence_output, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return self.cls(sequence_output)

    def get_loss(
        self,
        labels: Tensor,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
        *,
        ignore_index: int = -100,
        reduction: str | None = "mean",
    ) -> Tensor:
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return _token_ce_loss(
            logits,
            labels,
            ignore_index=ignore_index,
            reduction=reduction,
        )

    def create_masked_lm_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        special_tokens_mask: Tensor | None = None,
        *,
        mask_token_id: int = 103,
        mlm_probability: float = 0.15,
        mask_replace_prob: float = 0.8,
        random_replace_prob: float = 0.1,
        ignore_index: int = -100,
    ) -> tuple[Tensor, Tensor]:
        if not 0.0 <= mlm_probability <= 1.0:
            raise ValueError("mlm_probability must be in [0, 1].")
        if not 0.0 <= mask_replace_prob <= 1.0:
            raise ValueError("mask_replace_prob must be in [0, 1].")
        if not 0.0 <= random_replace_prob <= 1.0:
            raise ValueError("random_replace_prob must be in [0, 1].")
        if mask_replace_prob + random_replace_prob > 1.0:
            raise ValueError("mask_replace_prob + random_replace_prob must be <= 1.")

        device = input_ids.device
        input_ids = input_ids.astype(lucid.Long)
        labels = lucid.full(
            input_ids.shape, ignore_index, dtype=lucid.Long, device=device
        )

        if attention_mask is None:
            can_mask = lucid.ones(input_ids.shape, dtype=lucid.Int, device=device)
        else:
            can_mask = (attention_mask > 0).astype(lucid.Int)

        if special_tokens_mask is not None:
            can_mask = can_mask * (special_tokens_mask == 0).astype(lucid.Int)

        select_rand = lucid.random.rand(input_ids.shape, device=device)
        masked_positions = (
            (select_rand < mlm_probability).astype(lucid.Int) * can_mask
        ).astype(lucid.Int)

        labels = lucid.where(masked_positions > 0, input_ids, labels)
        masked_input_ids = input_ids.detach()

        replace_rand = lucid.random.rand(input_ids.shape, device=device)
        mask_token_positions = (
            (replace_rand < mask_replace_prob).astype(lucid.Int) * masked_positions
        ).astype(lucid.Int)
        random_token_positions = (
            (replace_rand >= mask_replace_prob).astype(lucid.Int)
            * (replace_rand < (mask_replace_prob + random_replace_prob)).astype(
                lucid.Int
            )
            * masked_positions
        ).astype(lucid.Int)

        mask_token_tensor = lucid.full(
            input_ids.shape, mask_token_id, dtype=lucid.Long, device=device
        )
        random_tokens = lucid.random.randint(
            0, self.config.vocab_size, input_ids.shape, device=device
        ).astype(lucid.Long)

        masked_input_ids = lucid.where(
            mask_token_positions > 0, mask_token_tensor, masked_input_ids
        )
        masked_input_ids = lucid.where(
            random_token_positions > 0, random_tokens, masked_input_ids
        )
        return masked_input_ids, labels

    def predict_token_ids(
        self,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
    ) -> Tensor:
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return _predict_labels_from_logits(logits)

    def get_accuracy(
        self,
        labels: Tensor,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
        *,
        ignore_index: int = -100,
    ) -> Tensor:
        pred_ids = self.predict_token_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return _masked_token_accuracy(pred_ids, labels, ignore_index=ignore_index)


class BERTForCausalLM(_BERTTaskWrapperMixin, nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.config = config
        self.bert = BERT(config)
        self.cls = _BERTOnlyMLMHead(config)
        self.cls.apply(self.bert._init_weights)
        self.bert.set_output_embeddings(self.cls.predictions.decoder)

    def forward(
        self,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
        encoder_hidden_states: lucid.FloatTensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        past_key_values: nn.KVCache | nn.EncoderDecoderCache | None = None,
        use_cache: bool | None = None,
        cache_position: Tensor | None = None,
    ) -> Tensor:
        sequence_output, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        return self.cls(sequence_output)

    def get_loss(
        self,
        labels: Tensor,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
        encoder_hidden_states: lucid.FloatTensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        past_key_values: nn.KVCache | nn.EncoderDecoderCache | None = None,
        use_cache: bool | None = None,
        cache_position: Tensor | None = None,
        *,
        shift_labels: bool = True,
        ignore_index: int = -100,
        reduction: str | None = "mean",
    ) -> Tensor:
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        logits, labels = _shift_causal_lm_labels(logits, labels, shift_labels)
        return _token_ce_loss(
            logits,
            labels,
            ignore_index=ignore_index,
            reduction=reduction,
        )

    def predict_token_ids(
        self,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
        encoder_hidden_states: lucid.FloatTensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        past_key_values: nn.KVCache | nn.EncoderDecoderCache | None = None,
        use_cache: bool | None = None,
        cache_position: Tensor | None = None,
    ) -> Tensor:
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        return _predict_labels_from_logits(logits)

    def get_accuracy(
        self,
        labels: Tensor,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
        encoder_hidden_states: lucid.FloatTensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        past_key_values: nn.KVCache | nn.EncoderDecoderCache | None = None,
        use_cache: bool | None = None,
        cache_position: Tensor | None = None,
        *,
        shift_labels: bool = True,
        ignore_index: int = -100,
    ) -> Tensor:
        pred_ids = self.predict_token_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        target = labels
        if shift_labels:
            if pred_ids.shape[1] < 2:
                raise ValueError(
                    "shift_labels=True requires sequence length >= 2 "
                    f"(got {pred_ids.shape[1]})."
                )
            pred_ids = pred_ids[:, :-1]
            target = labels[:, 1:]
        return _masked_token_accuracy(pred_ids, target, ignore_index=ignore_index)

    def get_perplexity(
        self,
        labels: Tensor,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
        encoder_hidden_states: lucid.FloatTensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        past_key_values: nn.KVCache | nn.EncoderDecoderCache | None = None,
        use_cache: bool | None = None,
        cache_position: Tensor | None = None,
        *,
        shift_labels: bool = True,
        ignore_index: int = -100,
    ) -> Tensor:
        loss = self.get_loss(
            labels=labels,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            shift_labels=shift_labels,
            ignore_index=ignore_index,
            reduction="mean",
        )
        return lucid.exp(loss)

    def get_next_token_logits(
        self,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
        encoder_hidden_states: lucid.FloatTensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        past_key_values: nn.KVCache | nn.EncoderDecoderCache | None = None,
        use_cache: bool | None = None,
        cache_position: Tensor | None = None,
    ) -> Tensor:
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        return logits[:, -1, :]

    def predict_next_token_id(
        self,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
        encoder_hidden_states: lucid.FloatTensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        past_key_values: nn.KVCache | nn.EncoderDecoderCache | None = None,
        use_cache: bool | None = None,
        cache_position: Tensor | None = None,
    ) -> Tensor:
        next_logits = self.get_next_token_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        return lucid.argmax(next_logits, axis=-1)


class BERTForNextSentencePrediction(_BERTTaskWrapperMixin, nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.config = config
        self.bert = BERT(config)
        self.cls = _BERTOnlyNSPHead(config)
        self.cls.apply(self.bert._init_weights)

    def forward(
        self,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
    ) -> Tensor:
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        if pooled_output is None:
            raise RuntimeError("BERTForNextSentencePrediction requires pooled output.")
        return self.cls(pooled_output)

    def get_loss(
        self,
        labels: Tensor,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
        *,
        reduction: str | None = "mean",
    ) -> Tensor:
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return _classification_ce_loss(logits, labels, reduction=reduction)

    def predict_labels(
        self,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
    ) -> Tensor:
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return _predict_labels_from_logits(logits)

    def predict_proba(
        self,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
    ) -> Tensor:
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return _predict_proba_from_logits(logits)

    def get_accuracy(
        self,
        labels: Tensor,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
    ) -> Tensor:
        preds = self.predict_labels(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return _classification_accuracy(preds, labels)


class BERTForSequenceClassification(_BERTTaskWrapperMixin, nn.Module):
    def __init__(self, config: BERTConfig, num_labels: int = 2) -> None:
        super().__init__()
        self.config = config
        self.bert = BERT(config)
        self.num_labels = num_labels

        classifier_dropout = config.classifier_dropout
        if classifier_dropout is None:
            classifier_dropout = config.hidden_dropout_prob

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.classifier.apply(self.bert._init_weights)

    def forward(
        self,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
    ) -> Tensor:
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        if pooled_output is None:
            raise RuntimeError("BERTForSequenceClassification requires pooled output.")
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def get_loss(
        self,
        labels: Tensor,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
        *,
        reduction: str | None = "mean",
    ) -> Tensor:
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return _classification_ce_loss(logits, labels, reduction=reduction)

    def predict_labels(
        self,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
    ) -> Tensor:
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return _predict_labels_from_logits(logits)

    def predict_proba(
        self,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
    ) -> Tensor:
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return _predict_proba_from_logits(logits)

    def get_accuracy(
        self,
        labels: Tensor,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
    ) -> Tensor:
        preds = self.predict_labels(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return _classification_accuracy(preds, labels)


class BERTForTokenClassification(_BERTTaskWrapperMixin, nn.Module):
    def __init__(self, config: BERTConfig, num_labels: int = 2) -> None:
        super().__init__()
        self.config = config
        self.bert = BERT(config)
        self.num_labels = num_labels

        classifier_dropout = config.classifier_dropout
        if classifier_dropout is None:
            classifier_dropout = config.hidden_dropout_prob

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.classifier.apply(self.bert._init_weights)

    def forward(
        self,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
    ) -> Tensor:
        sequence_output, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

    def get_loss(
        self,
        labels: Tensor,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
        *,
        ignore_index: int = -100,
        reduction: str | None = "mean",
    ) -> Tensor:
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return _token_ce_loss(
            logits,
            labels,
            ignore_index=ignore_index,
            reduction=reduction,
        )

    def predict_token_labels(
        self,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
    ) -> Tensor:
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return _predict_labels_from_logits(logits)

    def get_accuracy(
        self,
        labels: Tensor,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
        *,
        ignore_index: int = -100,
    ) -> Tensor:
        preds = self.predict_token_labels(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return _masked_token_accuracy(preds, labels, ignore_index=ignore_index)


class BERTForQuestionAnswering(_BERTTaskWrapperMixin, nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.config = config
        self.bert = BERT(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.qa_outputs.apply(self.bert._init_weights)

    def forward(
        self,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        sequence_output, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        logits = self.qa_outputs(sequence_output)
        start_logits = logits[:, :, 0]
        end_logits = logits[:, :, 1]
        return start_logits, end_logits

    def get_loss(
        self,
        start_positions: Tensor,
        end_positions: Tensor,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
        *,
        reduction: str | None = "mean",
    ) -> Tensor:
        start_logits, end_logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        return _qa_ce_loss(
            start_logits,
            end_logits,
            start_positions,
            end_positions,
            reduction=reduction,
        )

    def predict_spans(
        self,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        start_logits, end_logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        start_pred = _predict_labels_from_logits(start_logits)
        end_pred = _predict_labels_from_logits(end_logits)
        return start_pred, end_pred

    def get_best_spans(
        self,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
        *,
        max_answer_length: int = 30,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if max_answer_length < 1:
            raise ValueError(
                f"max_answer_length must be >= 1 (got {max_answer_length})."
            )

        start_logits, end_logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        start_probs = _predict_proba_from_logits(start_logits)
        end_probs = _predict_proba_from_logits(end_logits)

        batch_size, seq_len = start_probs.shape
        best_starts: list[int] = []
        best_ends: list[int] = []
        best_scores: list[float] = []

        for batch_idx in range(batch_size):
            s_probs = start_probs[batch_idx].tolist()
            e_probs = end_probs[batch_idx].tolist()

            best_score = -1.0
            best_s = 0
            best_e = 0

            for s_idx in range(seq_len):
                max_e = min(seq_len, s_idx + max_answer_length)
                for e_idx in range(s_idx, max_e):
                    score = float(s_probs[s_idx]) * float(e_probs[e_idx])
                    if score > best_score:
                        best_score = score
                        best_s = s_idx
                        best_e = e_idx

            best_starts.append(best_s)
            best_ends.append(best_e)
            best_scores.append(best_score)

        device = start_probs.device
        return (
            lucid.tensor(best_starts, dtype=lucid.Long, device=device),
            lucid.tensor(best_ends, dtype=lucid.Long, device=device),
            lucid.tensor(best_scores, dtype=lucid.Float32, device=device),
        )

    def get_accuracy(
        self,
        start_positions: Tensor,
        end_positions: Tensor,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
    ) -> Tensor:
        start_pred, end_pred = self.predict_spans(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        start_targets = start_positions.reshape(-1).astype(lucid.Int)
        end_targets = end_positions.reshape(-1).astype(lucid.Int)
        start_correct = (
            start_pred.reshape(-1).astype(lucid.Int) == start_targets
        ).astype(lucid.Int)
        end_correct = (end_pred.reshape(-1).astype(lucid.Int) == end_targets).astype(
            lucid.Int
        )
        exact_match = (start_correct * end_correct).astype(lucid.Float32)
        return exact_match.mean()


def _build_bert_config(
    *,
    vocab_size: int,
    hidden_size: int,
    num_attention_heads: int,
    num_hidden_layers: int,
    intermediate_size: int,
    is_decoder: bool,
    use_cache: bool,
    add_pooling_layer: bool,
    **kwargs,
) -> BERTConfig:
    defaults = dict(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        hidden_act=F.gelu,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        tie_word_embedding=True,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_cache=use_cache,
        is_decoder=is_decoder,
        add_cross_attention=False,
        chunk_size_feed_forward=0,
        pad_token_id=0,
        classifier_dropout=None,
        add_pooling_layer=add_pooling_layer,
    )
    defaults.update(kwargs)
    return BERTConfig(**defaults)


def _bert_base_config(
    *,
    is_decoder: bool = False,
    use_cache: bool = False,
    add_pooling_layer: bool = True,
    vocab_size: int = 30522,
    **kwargs,
) -> BERTConfig:
    return _build_bert_config(
        vocab_size=vocab_size,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        intermediate_size=3072,
        is_decoder=is_decoder,
        use_cache=use_cache,
        add_pooling_layer=add_pooling_layer,
        **kwargs,
    )


def _bert_large_config(
    *,
    is_decoder: bool = False,
    use_cache: bool = False,
    add_pooling_layer: bool = True,
    vocab_size: int = 30522,
    **kwargs,
) -> BERTConfig:
    return _build_bert_config(
        vocab_size=vocab_size,
        hidden_size=1024,
        num_attention_heads=16,
        num_hidden_layers=24,
        intermediate_size=4096,
        is_decoder=is_decoder,
        use_cache=use_cache,
        add_pooling_layer=add_pooling_layer,
        **kwargs,
    )


@register_model
def bert_for_pre_training_base(vocab_size: int = 30522, **kwargs) -> BERTForPreTraining:
    config = _bert_base_config(vocab_size=vocab_size, add_pooling_layer=True, **kwargs)
    return BERTForPreTraining(config)


@register_model
def bert_for_pre_training_large(
    vocab_size: int = 30522, **kwargs
) -> BERTForPreTraining:
    config = _bert_large_config(vocab_size=vocab_size, add_pooling_layer=True, **kwargs)
    return BERTForPreTraining(config)


@register_model
def bert_for_masked_lm_base(vocab_size: int = 30522, **kwargs) -> BERTForMaskedLM:
    config = _bert_base_config(vocab_size=vocab_size, add_pooling_layer=False, **kwargs)
    return BERTForMaskedLM(config)


@register_model
def bert_for_masked_lm_large(vocab_size: int = 30522, **kwargs) -> BERTForMaskedLM:
    config = _bert_large_config(
        vocab_size=vocab_size, add_pooling_layer=False, **kwargs
    )
    return BERTForMaskedLM(config)


@register_model
def bert_for_causal_lm_base(vocab_size: int = 30522, **kwargs) -> BERTForCausalLM:
    config = _bert_base_config(
        vocab_size=vocab_size,
        is_decoder=True,
        use_cache=True,
        add_pooling_layer=False,
        **kwargs,
    )
    return BERTForCausalLM(config)


@register_model
def bert_for_causal_lm_large(vocab_size: int = 30522, **kwargs) -> BERTForCausalLM:
    config = _bert_large_config(
        vocab_size=vocab_size,
        is_decoder=True,
        use_cache=True,
        add_pooling_layer=False,
        **kwargs,
    )
    return BERTForCausalLM(config)


@register_model
def bert_for_next_sentence_prediction_base(
    vocab_size: int = 30522, **kwargs
) -> BERTForNextSentencePrediction:
    config = _bert_base_config(vocab_size=vocab_size, add_pooling_layer=True, **kwargs)
    return BERTForNextSentencePrediction(config)


@register_model
def bert_for_next_sentence_prediction_large(
    vocab_size: int = 30522, **kwargs
) -> BERTForNextSentencePrediction:
    config = _bert_large_config(vocab_size=vocab_size, add_pooling_layer=True, **kwargs)
    return BERTForNextSentencePrediction(config)


@register_model
def bert_for_sequence_classification_base(
    num_labels: int = 2, vocab_size: int = 30522, **kwargs
) -> BERTForSequenceClassification:
    config = _bert_base_config(vocab_size=vocab_size, add_pooling_layer=True, **kwargs)
    return BERTForSequenceClassification(config, num_labels=num_labels)


@register_model
def bert_for_sequence_classification_large(
    num_labels: int = 2, vocab_size: int = 30522, **kwargs
) -> BERTForSequenceClassification:
    config = _bert_large_config(vocab_size=vocab_size, add_pooling_layer=True, **kwargs)
    return BERTForSequenceClassification(config, num_labels=num_labels)


@register_model
def bert_for_token_classification_base(
    num_labels: int = 2, vocab_size: int = 30522, **kwargs
) -> BERTForTokenClassification:
    config = _bert_base_config(vocab_size=vocab_size, add_pooling_layer=False, **kwargs)
    return BERTForTokenClassification(config, num_labels=num_labels)


@register_model
def bert_for_token_classification_large(
    num_labels: int = 2, vocab_size: int = 30522, **kwargs
) -> BERTForTokenClassification:
    config = _bert_large_config(
        vocab_size=vocab_size, add_pooling_layer=False, **kwargs
    )
    return BERTForTokenClassification(config, num_labels=num_labels)


@register_model
def bert_for_question_answering_base(
    vocab_size: int = 30522, **kwargs
) -> BERTForQuestionAnswering:
    config = _bert_base_config(vocab_size=vocab_size, add_pooling_layer=False, **kwargs)
    return BERTForQuestionAnswering(config)


@register_model
def bert_for_question_answering_large(
    vocab_size: int = 30522, **kwargs
) -> BERTForQuestionAnswering:
    config = _bert_large_config(
        vocab_size=vocab_size, add_pooling_layer=False, **kwargs
    )
    return BERTForQuestionAnswering(config)
