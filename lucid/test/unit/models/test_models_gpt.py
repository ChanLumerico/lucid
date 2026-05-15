"""Unit tests for GPT-1 (text family, Phase 4 second concrete model).

Validates the decoder-only stack + the first real consumer of
``GenerationMixin.generate`` (greedy + sampling end-to-end through a tiny
randomly-initialised model).
"""

import pytest

import lucid
from lucid.models import (
    GPTConfig,
    GPTForSequenceClassification,
    GPTLMHeadModel,
    GPTModel,
    create_model,
    is_model,
)

_VOCAB = 64
_HIDDEN = 32
_LAYERS = 2
_HEADS = 4
_INTER = 64
_MAX_POS = 16


def _tiny_config(**overrides: object) -> GPTConfig:
    base = {
        "vocab_size": _VOCAB,
        "hidden_size": _HIDDEN,
        "num_hidden_layers": _LAYERS,
        "num_attention_heads": _HEADS,
        "intermediate_size": _INTER,
        "max_position_embeddings": _MAX_POS,
        "num_labels": 3,
    }
    base.update(overrides)
    return GPTConfig(**base)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class TestGPTConfig:
    def test_paper_defaults(self) -> None:
        cfg = GPTConfig()
        assert cfg.hidden_size == 768
        assert cfg.num_hidden_layers == 12
        assert cfg.num_attention_heads == 12
        assert cfg.max_position_embeddings == 512
        assert cfg.pad_token_id is None
        assert cfg.hidden_act == "gelu_new"
        assert cfg.model_type == "gpt"

    def test_num_labels_invariant(self) -> None:
        with pytest.raises(ValueError, match="num_labels"):
            GPTConfig(num_labels=-1)

    def test_classifier_dropout_range(self) -> None:
        with pytest.raises(ValueError, match="classifier_dropout"):
            GPTConfig(classifier_dropout=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Forward shapes
# ─────────────────────────────────────────────────────────────────────────────


class TestGPTModelForward:
    def test_trunk_shape(self) -> None:
        m = GPTModel(_tiny_config()).eval()
        ids = lucid.tensor([[1, 2, 3, 4, 5, 6]]).long()
        out = m(ids)
        assert tuple(out.last_hidden_state.shape) == (1, 6, _HIDDEN)

    def test_seq_too_long_raises(self) -> None:
        m = GPTModel(_tiny_config(max_position_embeddings=4)).eval()
        ids = lucid.tensor([[1, 2, 3, 4, 5]]).long()
        with pytest.raises(ValueError, match="max_position_embeddings"):
            m(ids)

    def test_input_embeddings_accessor(self) -> None:
        m = GPTModel(_tiny_config()).eval()
        assert m.get_input_embeddings() is m.tokens_embed


class TestGPTLMHeadModel:
    def test_logits_and_shift_loss(self) -> None:
        cfg = _tiny_config()
        m = GPTLMHeadModel(cfg).eval()
        ids = lucid.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]).long()
        out = m(ids, labels=ids)
        assert tuple(out.logits.shape) == (1, 8, _VOCAB)
        assert out.loss is not None
        assert float(out.loss.item()) > 0.0

    def test_lm_head_is_tied(self) -> None:
        m = GPTLMHeadModel(_tiny_config()).eval()
        assert m.lm_head.weight is m.transformer.tokens_embed.weight

    def test_lm_head_untied_when_disabled(self) -> None:
        m = GPTLMHeadModel(_tiny_config(tie_word_embeddings=False)).eval()
        assert m.lm_head.weight is not m.transformer.tokens_embed.weight


class TestGPTGenerationMixin:
    def test_greedy_generate(self) -> None:
        m = GPTLMHeadModel(_tiny_config()).eval()
        prompt = lucid.tensor([[1, 2, 3]]).long()
        out = m.generate(prompt, max_length=8, do_sample=False)
        assert tuple(out.shape) == (1, 8)
        # First three tokens must match the prompt verbatim.
        assert [int(out[0, t].item()) for t in range(3)] == [1, 2, 3]

    def test_sampling_generate(self) -> None:
        m = GPTLMHeadModel(_tiny_config()).eval()
        prompt = lucid.tensor([[1, 2, 3]]).long()
        out = m.generate(
            prompt,
            max_length=8,
            do_sample=True,
            temperature=0.8,
            top_k=10,
            top_p=0.95,
        )
        assert tuple(out.shape) == (1, 8)
        # All sampled tokens must be inside the vocabulary.
        tokens = [int(out[0, t].item()) for t in range(8)]
        assert all(0 <= t < _VOCAB for t in tokens)

    def test_max_new_tokens(self) -> None:
        m = GPTLMHeadModel(_tiny_config()).eval()
        prompt = lucid.tensor([[1, 2, 3]]).long()
        out = m.generate(prompt, max_new_tokens=2, max_length=1000)
        assert tuple(out.shape) == (1, 5)


class TestGPTForSequenceClassification:
    def test_logits_with_mask(self) -> None:
        cfg = _tiny_config()
        m = GPTForSequenceClassification(cfg).eval()
        ids = lucid.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]).long()
        mask = lucid.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]).long()
        out = m(ids, attention_mask=mask, labels=lucid.tensor([2]).long())
        assert tuple(out.logits.shape) == (1, cfg.num_labels)
        assert out.loss is not None

    def test_logits_without_mask(self) -> None:
        cfg = _tiny_config()
        m = GPTForSequenceClassification(cfg).eval()
        ids = lucid.tensor([[1, 2, 3, 4]]).long()
        out = m(ids)
        assert tuple(out.logits.shape) == (1, cfg.num_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Causal masking — sanity check
# ─────────────────────────────────────────────────────────────────────────────


class TestCausalMask:
    def test_future_tokens_do_not_leak(self) -> None:
        """Token t's hidden state must be invariant to tokens at positions > t.

        Replacing the suffix of an input must leave the prefix's hidden states
        bit-identical (in inference mode, no dropout).
        """
        m = GPTModel(_tiny_config()).eval()
        ids_a = lucid.tensor([[1, 2, 3, 4, 5, 6]]).long()
        ids_b = lucid.tensor([[1, 2, 3, 40, 50, 60]]).long()  # suffix mutated
        h_a = m(ids_a).last_hidden_state
        h_b = m(ids_b).last_hidden_state
        # Compare positions 0..2 (which only attend to prefix [0..t]).
        diff = float(((h_a[:, :3, :] - h_b[:, :3, :]) ** 2).sum().item())
        assert diff < 1e-8, f"Causal mask leaks: prefix diff = {diff}"


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


class TestGPTRegistry:
    @pytest.mark.parametrize(
        "name",
        ["gpt", "gpt_lm", "gpt_cls"],
    )
    def test_registered(self, name: str) -> None:
        assert is_model(name)

    def test_factory_with_override(self) -> None:
        m = create_model(
            "gpt",
            vocab_size=_VOCAB,
            hidden_size=_HIDDEN,
            num_hidden_layers=_LAYERS,
            num_attention_heads=_HEADS,
            intermediate_size=_INTER,
            max_position_embeddings=_MAX_POS,
        )
        assert isinstance(m, GPTModel)
        ids = lucid.tensor([[1, 2, 3]]).long()
        out = m.eval()(ids)
        assert tuple(out.last_hidden_state.shape) == (1, 3, _HIDDEN)


# ─────────────────────────────────────────────────────────────────────────────
# DoubleHeads — LM + multiple-choice
# ─────────────────────────────────────────────────────────────────────────────


from lucid.models import GPTDoubleHeadsModel


class TestGPTDoubleHeadsModel:
    def test_shapes_and_losses(self) -> None:
        m = GPTDoubleHeadsModel(_tiny_config()).eval()
        # (N=1, C=3, L=4)
        ids = lucid.tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [1, 3, 5, 7]]]).long()
        mc_token_ids = lucid.tensor([[3, 3, 3]]).long()
        out = m(ids, mc_token_ids, labels=ids, mc_labels=lucid.tensor([1]).long())
        assert tuple(out.lm_logits.shape) == (1, 3, 4, _VOCAB)
        assert tuple(out.mc_logits.shape) == (1, 3)
        assert out.lm_loss is not None and out.mc_loss is not None
        assert out.loss is not None

    def test_input_rank_validation(self) -> None:
        m = GPTDoubleHeadsModel(_tiny_config()).eval()
        with pytest.raises(ValueError, match=r"\(N, C, L\)"):
            m(lucid.tensor([[1, 2, 3]]).long(), lucid.tensor([[0]]).long())
