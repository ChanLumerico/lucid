import pytest

import lucid

from lucid.models import (
    BERTConfig,
    BERTForSequenceClassification,
    BERTForSequenceClassificationConfig,
    BERTForTokenClassification,
    BERTForTokenClassificationConfig,
    RoFormerConfig,
    RoFormerForSequenceClassification,
    RoFormerForSequenceClassificationConfig,
    RoFormerForTokenClassification,
    RoFormerForTokenClassificationConfig,
)


def _small_bert_config() -> BERTConfig:
    return BERTConfig.base(
        vocab_size=32,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=16,
        max_position_embeddings=16,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        add_pooling_layer=True,
    )


def _small_roformer_config() -> RoFormerConfig:
    return RoFormerConfig.base(
        vocab_size=32,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=16,
        max_position_embeddings=16,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        add_pooling_layer=True,
    )


def test_text_wrapper_public_imports() -> None:
    assert BERTForSequenceClassification is not None
    assert BERTForSequenceClassificationConfig is not None
    assert BERTForTokenClassification is not None
    assert BERTForTokenClassificationConfig is not None
    assert RoFormerForSequenceClassification is not None
    assert RoFormerForSequenceClassificationConfig is not None
    assert RoFormerForTokenClassification is not None
    assert RoFormerForTokenClassificationConfig is not None


def test_bert_sequence_classification_accepts_config_and_runs() -> None:
    config = BERTForSequenceClassificationConfig(
        bert_config=_small_bert_config(),
        num_labels=3,
    )
    model = BERTForSequenceClassification(config)
    input_ids = lucid.Tensor([[1, 2, 3, 4]], dtype=lucid.Int32)
    attention_mask = lucid.ones((1, 4), dtype=lucid.Int32)
    labels = lucid.Tensor([1], dtype=lucid.Int32)

    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = model.get_loss(
        labels=labels,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    assert model.config is config
    assert logits.shape == (1, 3)
    assert loss.size == 1


def test_bert_token_classification_accepts_config_and_runs() -> None:
    config = BERTForTokenClassificationConfig(
        bert_config=_small_bert_config(),
        num_labels=4,
    )
    model = BERTForTokenClassification(config)
    input_ids = lucid.Tensor([[1, 2, 3, 4]], dtype=lucid.Int32)
    attention_mask = lucid.ones((1, 4), dtype=lucid.Int32)
    labels = lucid.Tensor([[0, 1, 2, 3]], dtype=lucid.Int32)

    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = model.get_loss(
        labels=labels,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    assert model.config is config
    assert logits.shape == (1, 4, 4)
    assert loss.size == 1


def test_roformer_sequence_classification_accepts_config_and_runs() -> None:
    config = RoFormerForSequenceClassificationConfig(
        roformer_config=_small_roformer_config(),
        num_labels=3,
    )
    model = RoFormerForSequenceClassification(config)
    input_ids = lucid.Tensor([[1, 2, 3, 4]], dtype=lucid.Int32)
    attention_mask = lucid.ones((1, 4), dtype=lucid.Int32)
    labels = lucid.Tensor([1], dtype=lucid.Int32)

    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = model.get_loss(
        labels=labels,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    assert model.config is config
    assert logits.shape == (1, 3)
    assert loss.size == 1


def test_roformer_token_classification_accepts_config_and_runs() -> None:
    config = RoFormerForTokenClassificationConfig(
        roformer_config=_small_roformer_config(),
        num_labels=4,
    )
    model = RoFormerForTokenClassification(config)
    input_ids = lucid.Tensor([[1, 2, 3, 4]], dtype=lucid.Int32)
    attention_mask = lucid.ones((1, 4), dtype=lucid.Int32)
    labels = lucid.Tensor([[0, 1, 2, 3]], dtype=lucid.Int32)

    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = model.get_loss(
        labels=labels,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    assert model.config is config
    assert logits.shape == (1, 4, 4)
    assert loss.size == 1


@pytest.mark.parametrize(
    "config_cls,key",
    (
        (BERTForSequenceClassificationConfig, "bert_config"),
        (BERTForTokenClassificationConfig, "bert_config"),
        (RoFormerForSequenceClassificationConfig, "roformer_config"),
        (RoFormerForTokenClassificationConfig, "roformer_config"),
    ),
)
def test_text_wrapper_configs_reject_invalid_values(
    config_cls: type,
    key: str,
) -> None:
    base = _small_bert_config() if key == "bert_config" else _small_roformer_config()

    with pytest.raises((TypeError, ValueError)):
        config_cls(**{key: object(), "num_labels": 2})

    with pytest.raises((TypeError, ValueError)):
        config_cls(**{key: base, "num_labels": 0})
