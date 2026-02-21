import lucid
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor

from lucid.models.text.bert import _wrappers as bert

from ._model import RoFormer, RoFormerConfig


__all__ = [
    "RoFormerForMaskedLM",
    "RoFormerForSequenceClassification",
    "RoFormerForTokenClassification",
    "RoFormerForMultipleChoice",
    "RoFormerForQuestionAnswering",
]


@register_model
class RoFormerForMaskedLM(bert.BERTForMaskedLM):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__(config)
        self.bert = RoFormer(config)
        self.cls.apply(self.bert._init_weights)
        self.bert.set_output_embeddings(self.cls.predictions.decoder)


@register_model
class RoFormerForSequenceClassification(bert.BERTForSequenceClassification):
    def __init__(self, config: RoFormerConfig, num_labels: int = 2) -> None:
        super().__init__(config, num_labels=num_labels)
        self.bert = RoFormer(config)
        self.classifier.apply(self.bert._init_weights)


@register_model
class RoFormerForTokenClassification(bert.BERTForTokenClassification):
    def __init__(self, config: RoFormerConfig, num_labels: int = 2) -> None:
        super().__init__(config, num_labels=num_labels)
        self.bert = RoFormer(config)
        self.classifier.apply(self.bert._init_weights)


@register_model
class RoFormerForMultipleChoice(bert._BERTTaskWrapperMixin, nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.config = config
        self.bert = RoFormer(config)

        classifier_dropout = config.classifier_dropout
        if classifier_dropout is None:
            classifier_dropout = config.hidden_dropout_prob

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier.apply(self.bert._init_weights)

    def forward(
        self,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
    ) -> Tensor:
        if input_ids is not None:
            batch_size, num_choices = input_ids.shape[:2]
            flat_input_ids = input_ids.reshape(-1, input_ids.shape[-1])
        elif inputs_embeds is not None:
            batch_size, num_choices = inputs_embeds.shape[:2]
            flat_input_ids = None
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        flat_attention_mask = (
            attention_mask.reshape(-1, attention_mask.shape[-1])
            if attention_mask is not None
            else None
        )
        flat_token_type_ids = (
            token_type_ids.reshape(-1, token_type_ids.shape[-1])
            if token_type_ids is not None
            else None
        )
        flat_position_ids = (
            position_ids.reshape(-1, position_ids.shape[-1])
            if position_ids is not None
            else None
        )
        flat_inputs_embeds = (
            inputs_embeds.reshape(-1, *inputs_embeds.shape[-2:])
            if inputs_embeds is not None
            else None
        )

        _, pooled_output = self.bert(
            **self._encoder_kwargs(
                input_ids=flat_input_ids,
                attention_mask=flat_attention_mask,
                token_type_ids=flat_token_type_ids,
                position_ids=flat_position_ids,
                inputs_embeds=flat_inputs_embeds,
            )
        )
        if pooled_output is None:
            raise RuntimeError("RoFormerForMultipleChoice requires pooled output.")

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits.reshape(batch_size, num_choices)

    def _forward_logits(
        self,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
    ) -> Tensor:
        return self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

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
        logits = self._forward_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return bert._classification_ce_loss(logits, labels, reduction=reduction)

    def predict_labels(
        self,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
    ) -> Tensor:
        logits = self._forward_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return bert._predict_labels_from_logits(logits)

    def predict_proba(
        self,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
    ) -> Tensor:
        logits = self._forward_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return bert._predict_proba_from_logits(logits)

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
        return bert._classification_accuracy(preds, labels)


@register_model
class RoFormerForQuestionAnswering(bert.BERTForQuestionAnswering):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__(config)
        self.bert = RoFormer(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.qa_outputs.apply(self.bert._init_weights)
