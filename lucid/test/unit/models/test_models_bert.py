"""Unit tests for BERT (text family, Phase 4 first concrete model).

Uses tiny config (vocab=64, hidden=32, layers=2, heads=4) so the full suite
runs in well under a second even on Metal.
"""

import pytest

import lucid
from lucid.models import (
    BERTConfig,
    BERTForMaskedLM,
    BERTForQuestionAnswering,
    BERTForSequenceClassification,
    BERTForTokenClassification,
    BERTModel,
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


def _tiny_config(**overrides: object) -> BERTConfig:
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
    return BERTConfig(**base)  # type: ignore[arg-type]


def _ids(B: int = 2, T: int = 8) -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor]:
    ids = lucid.tensor([[i % _VOCAB for i in range(T)] for _ in range(B)]).long()
    attn = lucid.tensor([[1] * T for _ in range(B)]).long()
    tt = lucid.tensor([[0] * T for _ in range(B)]).long()
    return ids, attn, tt


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class TestBERTConfig:
    def test_defaults_match_bert_base(self) -> None:
        cfg = BERTConfig()
        assert cfg.hidden_size == 768
        assert cfg.num_hidden_layers == 12
        assert cfg.num_attention_heads == 12
        assert cfg.intermediate_size == 3072
        assert cfg.type_vocab_size == 2
        assert cfg.model_type == "bert"

    def test_type_vocab_size_invariant(self) -> None:
        with pytest.raises(ValueError, match="type_vocab_size"):
            BERTConfig(type_vocab_size=0)

    def test_num_labels_invariant(self) -> None:
        with pytest.raises(ValueError, match="num_labels"):
            BERTConfig(num_labels=0)

    def test_classifier_dropout_range(self) -> None:
        with pytest.raises(ValueError, match="classifier_dropout"):
            BERTConfig(classifier_dropout=1.5)

    def test_head_divisibility_inherited(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            BERTConfig(hidden_size=10, num_attention_heads=3)


# ─────────────────────────────────────────────────────────────────────────────
# Forward shapes
# ─────────────────────────────────────────────────────────────────────────────


class TestBERTModelForward:
    def test_bare_encoder(self) -> None:
        cfg = _tiny_config()
        m = BERTModel(cfg).eval()
        ids, attn, tt = _ids(B=2, T=8)
        out = m(ids, attn, tt)
        assert tuple(out.last_hidden_state.shape) == (2, 8, _HIDDEN)
        assert tuple(out.pooler_output.shape) == (2, _HIDDEN)

    def test_attention_mask_optional(self) -> None:
        cfg = _tiny_config()
        m = BERTModel(cfg).eval()
        ids, _, _ = _ids(B=1, T=6)
        out = m(ids)  # no mask, no token_type_ids
        assert tuple(out.last_hidden_state.shape) == (1, 6, _HIDDEN)

    def test_input_embeddings_accessor(self) -> None:
        m = BERTModel(_tiny_config()).eval()
        emb = m.get_input_embeddings()
        assert emb is m.embeddings.word_embeddings


class TestBERTForMaskedLM:
    def test_logits_shape_and_loss(self) -> None:
        cfg = _tiny_config()
        m = BERTForMaskedLM(cfg).eval()
        ids, attn, tt = _ids(B=2, T=8)
        labels = lucid.tensor([[3, -100, 5, -100, -100, -100, -100, -100]] * 2).long()
        out = m(ids, attn, tt, labels=labels)
        assert tuple(out.logits.shape) == (2, 8, _VOCAB)
        assert out.loss is not None
        assert float(out.loss.item()) > 0.0

    def test_decoder_weight_is_tied(self) -> None:
        m = BERTForMaskedLM(_tiny_config()).eval()
        assert (
            m.cls.predictions.decoder.weight is m.bert.embeddings.word_embeddings.weight
        )

    def test_untied_when_disabled(self) -> None:
        cfg = _tiny_config(tie_word_embeddings=False)
        m = BERTForMaskedLM(cfg).eval()
        assert (
            m.cls.predictions.decoder.weight
            is not m.bert.embeddings.word_embeddings.weight
        )


class TestBERTForSequenceClassification:
    def test_logits_and_loss(self) -> None:
        cfg = _tiny_config()
        m = BERTForSequenceClassification(cfg).eval()
        ids, attn, tt = _ids(B=2, T=8)
        labels = lucid.tensor([1, 2]).long()
        out = m(ids, attn, tt, labels=labels)
        assert tuple(out.logits.shape) == (2, cfg.num_labels)
        assert out.loss is not None


class TestBERTForTokenClassification:
    def test_logits_shape(self) -> None:
        cfg = _tiny_config()
        m = BERTForTokenClassification(cfg).eval()
        ids, attn, tt = _ids(B=1, T=6)
        out = m(ids, attn, tt)
        assert tuple(out.logits.shape) == (1, 6, cfg.num_labels)


class TestBERTForQuestionAnswering:
    def test_logits_and_span_loss(self) -> None:
        cfg = _tiny_config()
        m = BERTForQuestionAnswering(cfg).eval()
        ids, attn, tt = _ids(B=2, T=8)
        starts = lucid.tensor([1, 3]).long()
        ends = lucid.tensor([4, 5]).long()
        out = m(ids, attn, tt, start_positions=starts, end_positions=ends)
        assert tuple(out.logits.shape) == (2, 8, 2)
        assert out.loss is not None


# ─────────────────────────────────────────────────────────────────────────────
# Registry — factory dispatch
# ─────────────────────────────────────────────────────────────────────────────


class TestBERTRegistry:
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
        assert isinstance(m, BERTModel)
        ids, _, _ = _ids(B=1, T=4)
        out = m.eval()(ids)
        assert tuple(out.last_hidden_state.shape) == (1, 4, _HIDDEN)


# ─────────────────────────────────────────────────────────────────────────────
# Extra task wrappers — PreTraining / NSP / CausalLM
# ─────────────────────────────────────────────────────────────────────────────


from lucid.models import (
    BERTForCausalLM,
    BERTForNextSentencePrediction,
    BERTForPreTraining,
)


class TestBERTForPreTraining:
    def test_combined_loss(self) -> None:
        cfg = _tiny_config()
        m = BERTForPreTraining(cfg).eval()
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
        m = BERTForPreTraining(_tiny_config()).eval()
        ids, attn, tt = _ids(B=1, T=6)
        out = m(ids, attn, tt, labels=ids)
        assert out.mlm_loss is not None
        assert out.nsp_loss is None
        assert out.loss is not None  # = mlm_loss

    def test_tied_decoder(self) -> None:
        m = BERTForPreTraining(_tiny_config()).eval()
        assert (
            m.cls.predictions.decoder.weight is m.bert.embeddings.word_embeddings.weight
        )


class TestBERTForNextSentencePrediction:
    def test_logits_and_loss(self) -> None:
        m = BERTForNextSentencePrediction(_tiny_config()).eval()
        ids, attn, tt = _ids(B=2, T=8)
        out = m(ids, attn, tt, labels=lucid.tensor([1, 0]).long())
        assert tuple(out.logits.shape) == (2, 2)
        assert out.loss is not None


class TestBERTForCausalLM:
    def test_logits_and_shift_loss(self) -> None:
        m = BERTForCausalLM(_tiny_config()).eval()
        ids, attn, tt = _ids(B=1, T=8)
        out = m(ids, attn, tt, labels=ids)
        assert tuple(out.logits.shape) == (1, 8, _VOCAB)
        assert out.loss is not None

    def test_causal_mask_prevents_leak(self) -> None:
        m = BERTForCausalLM(_tiny_config()).eval()
        ids_a = lucid.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]).long()
        ids_b = lucid.tensor([[1, 2, 3, 40, 50, 60, 70, 80]]).long()
        h_a = m(ids_a).logits
        h_b = m(ids_b).logits
        diff = float(((h_a[:, :3, :] - h_b[:, :3, :]) ** 2).sum().item())
        assert diff < 1e-6, f"Causal mask leaks in BERTForCausalLM: diff = {diff}"


# (factory, slug, hidden, num_params, task) — the eight shipped BERT
# checkpoints: six base encoders (Turc miniatures + Devlin base/large) and two
# masked-LM heads.  All Wikipedia + BookCorpus, uncased, 30 522-token vocab.
_SHIPPED = (
    ("BERTTinyWeights", "bert-tiny", 128, 4_385_920, "base"),
    ("BERTMiniWeights", "bert-mini", 256, 11_171_328, "base"),
    ("BERTSmallWeights", "bert-small", 512, 28_763_648, "base"),
    ("BERTMediumWeights", "bert-medium", 512, 41_373_184, "base"),
    ("BERTBaseWeights", "bert-base", 768, 109_482_240, "base"),
    ("BERTLargeWeights", "bert-large", 1024, 335_141_888, "base"),
    ("BERTBaseMLMWeights", "bert-base-mlm", 30_522, 109_514_298, "mlm"),
    ("BERTLargeMLMWeights", "bert-large-mlm", 30_522, 335_174_586, "mlm"),
)
_TAG = "WIKIPEDIA_BOOKSCORPUS"


class TestBERTWeightsEnums:
    """Static contract of the per-variant Weights enums — no network."""

    @staticmethod
    def _enum(name: str) -> type:
        import lucid.models.weights as weights_ns

        return getattr(weights_ns, name)

    def test_default_aliases(self) -> None:
        # Each enum's DEFAULT points at its single shipped tag member.
        for enum_name, *_rest in _SHIPPED:
            cls = self._enum(enum_name)
            assert cls.DEFAULT is cls[_TAG]

    def test_entry_fields(self) -> None:
        for enum_name, slug, num_classes, _nparams, _task in _SHIPPED:
            member = self._enum(enum_name)[_TAG]
            e = member.entry
            assert e.num_classes == num_classes
            assert len(e.sha256) == 64
            assert f"lucid-dl/{slug}" in e.url
            assert f"/{_TAG}/" in e.url
            assert member.meta["tag"] == _TAG
            assert member.meta["license"] == "apache-2.0"

    def test_registered_for_factories(self) -> None:
        # The enum is discoverable from its factory name via lucid.weights.
        from lucid.weights import weights_for

        for enum_name, slug, *_rest in _SHIPPED:
            factory = slug.replace("-", "_")
            resolved = weights_for(factory)
            assert resolved is not None
            assert resolved.__name__ == enum_name
