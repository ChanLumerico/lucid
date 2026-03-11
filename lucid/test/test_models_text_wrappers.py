import lucid

from lucid.models import (
    BERTConfig,
    BERTForSequenceClassification,
    BERTForTokenClassification,
    RoFormerConfig,
    RoFormerForSequenceClassification,
    RoFormerForTokenClassification,
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
    assert BERTForTokenClassification is not None
    assert RoFormerForSequenceClassification is not None
    assert RoFormerForTokenClassification is not None


def test_bert_sequence_classification_accepts_config_and_runs() -> None:
    config = _small_bert_config()
    model = BERTForSequenceClassification(config, num_labels=3)
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
    config = _small_bert_config()
    model = BERTForTokenClassification(config, num_labels=4)
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
    config = _small_roformer_config()
    model = RoFormerForSequenceClassification(config, num_labels=3)
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
    config = _small_roformer_config()
    model = RoFormerForTokenClassification(config, num_labels=4)
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
