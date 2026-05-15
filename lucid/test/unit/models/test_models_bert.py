"""Unit tests for BERT (text family, Phase 4 first concrete model).

Uses tiny config (vocab=64, hidden=32, layers=2, heads=4) so the full suite
runs in well under a second even on Metal.
"""

import pytest

import lucid
from lucid.models import (
    BertConfig,
    BertForMaskedLM,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertModel,
    create_model,
    is_model,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_VOCAB = 64
_HIDDEN = 32
_LAYERS = 2
_HEADS = 4
_INTER = 64
_MAX_POS = 32


def _tiny_config(**overrides: object) -> BertConfig:
    base = {
        "vocab_size": _VOCAB,
        "hidden_size": _HIDDEN,
        "num_hidden_layers": _LAYERS,
        "num_attention_heads": _HEADS,
        "intermediate_size": _INTER,
        "max_position_embeddings": _MAX_POS,
        "type_vocab_size": 2,
        "num_labels": 3,
    }
    base.update(overrides)
    return BertConfig(**base)  # type: ignore[arg-type]


def _ids(B: int = 2, T: int = 8) -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor]:
    ids = lucid.tensor([[i % _VOCAB for i in range(T)] for _ in range(B)]).long()
    attn = lucid.tensor([[1] * T for _ in range(B)]).long()
    tt = lucid.tensor([[0] * T for _ in range(B)]).long()
    return ids, attn, tt


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class TestBertConfig:
    def test_defaults_match_bert_base(self) -> None:
        cfg = BertConfig()
        assert cfg.hidden_size == 768
        assert cfg.num_hidden_layers == 12
        assert cfg.num_attention_heads == 12
        assert cfg.intermediate_size == 3072
        assert cfg.type_vocab_size == 2
        assert cfg.model_type == "bert"

    def test_type_vocab_size_invariant(self) -> None:
        with pytest.raises(ValueError, match="type_vocab_size"):
            BertConfig(type_vocab_size=0)

    def test_num_labels_invariant(self) -> None:
        with pytest.raises(ValueError, match="num_labels"):
            BertConfig(num_labels=0)

    def test_classifier_dropout_range(self) -> None:
        with pytest.raises(ValueError, match="classifier_dropout"):
            BertConfig(classifier_dropout=1.5)

    def test_head_divisibility_inherited(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            BertConfig(hidden_size=10, num_attention_heads=3)


# ─────────────────────────────────────────────────────────────────────────────
# Forward shapes
# ─────────────────────────────────────────────────────────────────────────────


class TestBertModelForward:
    def test_bare_encoder(self) -> None:
        cfg = _tiny_config()
        m = BertModel(cfg).eval()
        ids, attn, tt = _ids(B=2, T=8)
        out = m(ids, attn, tt)
        assert tuple(out.last_hidden_state.shape) == (2, 8, _HIDDEN)
        assert tuple(out.pooler_output.shape) == (2, _HIDDEN)

    def test_attention_mask_optional(self) -> None:
        cfg = _tiny_config()
        m = BertModel(cfg).eval()
        ids, _, _ = _ids(B=1, T=6)
        out = m(ids)  # no mask, no token_type_ids
        assert tuple(out.last_hidden_state.shape) == (1, 6, _HIDDEN)

    def test_input_embeddings_accessor(self) -> None:
        m = BertModel(_tiny_config()).eval()
        emb = m.get_input_embeddings()
        assert emb is m.embeddings.word_embeddings


class TestBertForMaskedLM:
    def test_logits_shape_and_loss(self) -> None:
        cfg = _tiny_config()
        m = BertForMaskedLM(cfg).eval()
        ids, attn, tt = _ids(B=2, T=8)
        labels = lucid.tensor([[3, -100, 5, -100, -100, -100, -100, -100]] * 2).long()
        out = m(ids, attn, tt, labels=labels)
        assert tuple(out.logits.shape) == (2, 8, _VOCAB)
        assert out.loss is not None
        assert float(out.loss.item()) > 0.0

    def test_decoder_weight_is_tied(self) -> None:
        m = BertForMaskedLM(_tiny_config()).eval()
        assert (
            m.cls.predictions.decoder.weight is m.bert.embeddings.word_embeddings.weight
        )

    def test_untied_when_disabled(self) -> None:
        cfg = _tiny_config(tie_word_embeddings=False)
        m = BertForMaskedLM(cfg).eval()
        assert (
            m.cls.predictions.decoder.weight
            is not m.bert.embeddings.word_embeddings.weight
        )


class TestBertForSequenceClassification:
    def test_logits_and_loss(self) -> None:
        cfg = _tiny_config()
        m = BertForSequenceClassification(cfg).eval()
        ids, attn, tt = _ids(B=2, T=8)
        labels = lucid.tensor([1, 2]).long()
        out = m(ids, attn, tt, labels=labels)
        assert tuple(out.logits.shape) == (2, cfg.num_labels)
        assert out.loss is not None


class TestBertForTokenClassification:
    def test_logits_shape(self) -> None:
        cfg = _tiny_config()
        m = BertForTokenClassification(cfg).eval()
        ids, attn, tt = _ids(B=1, T=6)
        out = m(ids, attn, tt)
        assert tuple(out.logits.shape) == (1, 6, cfg.num_labels)


class TestBertForQuestionAnswering:
    def test_logits_and_span_loss(self) -> None:
        cfg = _tiny_config()
        m = BertForQuestionAnswering(cfg).eval()
        ids, attn, tt = _ids(B=2, T=8)
        starts = lucid.tensor([1, 3]).long()
        ends = lucid.tensor([4, 5]).long()
        out = m(ids, attn, tt, start_positions=starts, end_positions=ends)
        assert tuple(out.logits.shape) == (2, 8, 2)
        assert out.loss is not None


# ─────────────────────────────────────────────────────────────────────────────
# Registry — factory dispatch
# ─────────────────────────────────────────────────────────────────────────────


class TestBertRegistry:
    @pytest.mark.parametrize(
        "name",
        [
            "bert_tiny",
            "bert_mini",
            "bert_small",
            "bert_medium",
            "bert_base",
            "bert_large",
        ],
    )
    def test_base_variants_registered(self, name: str) -> None:
        assert is_model(name)

    @pytest.mark.parametrize(
        "name",
        [
            "bert_base_mlm",
            "bert_large_mlm",
            "bert_base_cls",
            "bert_large_cls",
            "bert_base_token_cls",
            "bert_base_qa",
        ],
    )
    def test_task_heads_registered(self, name: str) -> None:
        assert is_model(name)

    def test_tiny_factory_instantiates_with_override(self) -> None:
        # Override to tiny dims so we don't allocate a real BERT-Tiny (~4M params)
        # in CI just to verify the factory plumbing.
        m = create_model(
            "bert_tiny",
            vocab_size=_VOCAB,
            hidden_size=_HIDDEN,
            num_hidden_layers=_LAYERS,
            num_attention_heads=_HEADS,
            intermediate_size=_INTER,
            max_position_embeddings=_MAX_POS,
        )
        assert isinstance(m, BertModel)
        ids, _, _ = _ids(B=1, T=4)
        out = m.eval()(ids)
        assert tuple(out.last_hidden_state.shape) == (1, 4, _HIDDEN)


# ─────────────────────────────────────────────────────────────────────────────
# Extra task wrappers — PreTraining / NSP / CausalLM
# ─────────────────────────────────────────────────────────────────────────────


from lucid.models import (
    BertForCausalLM,
    BertForNextSentencePrediction,
    BertForPreTraining,
)


class TestBertForPreTraining:
    def test_combined_loss(self) -> None:
        cfg = _tiny_config()
        m = BertForPreTraining(cfg).eval()
        ids, attn, tt = _ids(B=2, T=8)
        mlm_labels = lucid.tensor(
            [[3, -100, 5, -100, -100, -100, -100, -100]] * 2
        ).long()
        nsp_labels = lucid.tensor([1, 0]).long()
        out = m(ids, attn, tt, labels=mlm_labels, next_sentence_label=nsp_labels)
        assert tuple(out.prediction_logits.shape) == (2, 8, _VOCAB)
        assert tuple(out.seq_relationship_logits.shape) == (2, 2)
        assert (
            out.loss is not None
            and out.mlm_loss is not None
            and out.nsp_loss is not None
        )

    def test_mlm_only(self) -> None:
        m = BertForPreTraining(_tiny_config()).eval()
        ids, attn, tt = _ids(B=1, T=6)
        out = m(ids, attn, tt, labels=ids)
        assert out.mlm_loss is not None
        assert out.nsp_loss is None
        assert out.loss is not None  # = mlm_loss

    def test_tied_decoder(self) -> None:
        m = BertForPreTraining(_tiny_config()).eval()
        assert (
            m.cls.predictions.decoder.weight is m.bert.embeddings.word_embeddings.weight
        )


class TestBertForNextSentencePrediction:
    def test_logits_and_loss(self) -> None:
        m = BertForNextSentencePrediction(_tiny_config()).eval()
        ids, attn, tt = _ids(B=2, T=8)
        out = m(ids, attn, tt, labels=lucid.tensor([1, 0]).long())
        assert tuple(out.logits.shape) == (2, 2)
        assert out.loss is not None


class TestBertForCausalLM:
    def test_logits_and_shift_loss(self) -> None:
        m = BertForCausalLM(_tiny_config()).eval()
        ids, attn, tt = _ids(B=1, T=8)
        out = m(ids, attn, tt, labels=ids)
        assert tuple(out.logits.shape) == (1, 8, _VOCAB)
        assert out.loss is not None

    def test_causal_mask_prevents_leak(self) -> None:
        m = BertForCausalLM(_tiny_config()).eval()
        ids_a = lucid.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]).long()
        ids_b = lucid.tensor([[1, 2, 3, 40, 50, 60, 70, 80]]).long()
        h_a = m(ids_a).logits
        h_b = m(ids_b).logits
        diff = float(((h_a[:, :3, :] - h_b[:, :3, :]) ** 2).sum().item())
        assert diff < 1e-6, f"Causal mask leaks in BertForCausalLM: diff = {diff}"
