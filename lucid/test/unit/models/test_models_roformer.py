"""Unit tests for RoFormer (Phase 4 fourth concrete model).

Validates the RoPE-based encoder.  Two RoFormer-specific checks beyond the
BERT test set:

  1. Permuting two tokens changes their hidden states (position info must
     come from somewhere — here, RoPE).
  2. There is no ``position_embeddings`` module on the embedding layer
     (RoFormer's design contract).
"""

import pytest

import lucid
from lucid.models import (
    RoFormerConfig,
    RoFormerForMaskedLM,
    RoFormerForSequenceClassification,
    RoFormerForTokenClassification,
    RoFormerModel,
    create_model,
    is_model,
)

_VOCAB = 64
_HIDDEN = 32
_LAYERS = 2
_HEADS = 4
_INTER = 64
_MAX_POS = 16


def _tiny_config(**overrides: object) -> RoFormerConfig:
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
    return RoFormerConfig(**base)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class TestRoFormerConfig:
    def test_paper_defaults(self) -> None:
        cfg = RoFormerConfig()
        assert cfg.vocab_size == 50_000
        assert cfg.hidden_size == 768
        assert cfg.num_hidden_layers == 12
        assert cfg.position_embedding_type == "rotary"
        assert cfg.rotary_base == 10_000.0
        assert cfg.model_type == "roformer"

    def test_odd_head_dim_rejected(self) -> None:
        # head_dim = 30 / 6 = 5 → odd
        with pytest.raises(ValueError, match="even head_dim"):
            RoFormerConfig(hidden_size=30, num_attention_heads=6)

    def test_rotary_base_invariant(self) -> None:
        with pytest.raises(ValueError, match="rotary_base"):
            RoFormerConfig(rotary_base=1.0)

    def test_classifier_dropout_range(self) -> None:
        with pytest.raises(ValueError, match="classifier_dropout"):
            RoFormerConfig(classifier_dropout=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture sanity
# ─────────────────────────────────────────────────────────────────────────────


class TestRoFormerArchitecture:
    def test_no_absolute_position_embedding(self) -> None:
        m = RoFormerModel(_tiny_config()).eval()
        # Embedding layer must NOT carry a position table — RoPE handles it.
        assert not hasattr(m.embeddings, "position_embeddings")

    def test_rotary_buffers_match_config(self) -> None:
        cfg = _tiny_config()
        m = RoFormerModel(cfg).eval()
        cos, sin = m.rotary()
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        assert tuple(cos.shape) == (cfg.max_position_embeddings, head_dim)
        assert tuple(sin.shape) == (cfg.max_position_embeddings, head_dim)

    def test_position_aware_via_rope(self) -> None:
        """If RoPE is wired correctly, swapping two tokens must change the
        hidden state at both positions.  Position-blind models would give
        the same hidden up to a permutation."""
        m = RoFormerModel(_tiny_config()).eval()
        ids_a = lucid.tensor([[1, 2, 3, 4, 5, 6]]).long()
        ids_b = lucid.tensor([[2, 1, 3, 4, 5, 6]]).long()  # swap positions 0 / 1
        h_a = m(ids_a).last_hidden_state
        h_b = m(ids_b).last_hidden_state
        diff = float(((h_a[:, 0, :] - h_b[:, 0, :]) ** 2).sum().item())
        assert diff > 1e-4, (
            "Position 0 hidden state did not change when token order was "
            "swapped — RoPE may not be applied."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Forward / heads
# ─────────────────────────────────────────────────────────────────────────────


class TestRoFormerForward:
    def test_bare_encoder(self) -> None:
        m = RoFormerModel(_tiny_config()).eval()
        ids = lucid.tensor([[1, 2, 3, 4, 5, 6]]).long()
        out = m(ids)
        assert tuple(out.last_hidden_state.shape) == (1, 6, _HIDDEN)
        assert tuple(out.pooler_output.shape) == (1, _HIDDEN)

    def test_seq_too_long_raises(self) -> None:
        m = RoFormerModel(_tiny_config(max_position_embeddings=4)).eval()
        with pytest.raises(ValueError, match="max_position_embeddings"):
            m(lucid.tensor([[1, 2, 3, 4, 5]]).long())


class TestRoFormerMaskedLM:
    def test_logits_and_loss(self) -> None:
        cfg = _tiny_config()
        m = RoFormerForMaskedLM(cfg).eval()
        ids = lucid.tensor([[1, 2, 3, 4, 5]]).long()
        out = m(ids, labels=ids)
        assert tuple(out.logits.shape) == (1, 5, _VOCAB)
        assert out.loss is not None

    def test_decoder_tied(self) -> None:
        m = RoFormerForMaskedLM(_tiny_config()).eval()
        assert (
            m.cls.predictions.decoder.weight
            is m.roformer.embeddings.word_embeddings.weight
        )


class TestRoFormerClassifiers:
    def test_sequence_cls(self) -> None:
        cfg = _tiny_config()
        m = RoFormerForSequenceClassification(cfg).eval()
        ids = lucid.tensor([[1, 2, 3, 4]]).long()
        out = m(ids, labels=lucid.tensor([2]).long())
        assert tuple(out.logits.shape) == (1, cfg.num_labels)

    def test_token_cls(self) -> None:
        cfg = _tiny_config()
        m = RoFormerForTokenClassification(cfg).eval()
        ids = lucid.tensor([[1, 2, 3, 4]]).long()
        out = m(ids)
        assert tuple(out.logits.shape) == (1, 4, cfg.num_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


class TestRoFormerRegistry:
    @pytest.mark.parametrize(
        "name",
        [
            "roformer",
            "roformer_mlm",
            "roformer_cls",
            "roformer_token_cls",
        ],
    )
    def test_registered(self, name: str) -> None:
        assert is_model(name)

    def test_factory_with_override(self) -> None:
        m = create_model(
            "roformer",
            vocab_size=_VOCAB,
            hidden_size=_HIDDEN,
            num_hidden_layers=_LAYERS,
            num_attention_heads=_HEADS,
            intermediate_size=_INTER,
            max_position_embeddings=_MAX_POS,
        )
        assert isinstance(m, RoFormerModel)
        out = m.eval()(lucid.tensor([[1, 2, 3]]).long())
        assert tuple(out.last_hidden_state.shape) == (1, 3, _HIDDEN)


from lucid.models import RoFormerForMultipleChoice, RoFormerForQuestionAnswering


class TestRoFormerExtras:
    def test_multiple_choice(self) -> None:
        cfg = _tiny_config()
        m = RoFormerForMultipleChoice(cfg).eval()
        ids = lucid.tensor(
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
        ).long()  # (N=1, C=3, L=3)
        out = m(ids, labels=lucid.tensor([1]).long())
        assert tuple(out.logits.shape) == (1, 3)
        assert out.loss is not None

    def test_question_answering(self) -> None:
        cfg = _tiny_config()
        m = RoFormerForQuestionAnswering(cfg).eval()
        ids = lucid.tensor([[1, 2, 3, 4, 5, 6]]).long()
        out = m(
            ids,
            start_positions=lucid.tensor([2]).long(),
            end_positions=lucid.tensor([4]).long(),
        )
        assert tuple(out.logits.shape) == (1, 6, 2)
        assert out.loss is not None
