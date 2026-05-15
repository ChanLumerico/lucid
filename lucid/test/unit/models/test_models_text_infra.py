"""Unit tests for the text-domain Phase 4 base layer.

Covers:
- LanguageModelConfig validation
- GenerationMixin greedy + sampling

(Positional-encoding tests — RoPE / sinusoidal PE — live in
``test/unit/nn/test_nn_positional.py`` since those primitives moved to
:mod:`lucid.nn` in 2026-05.)
"""

import pytest

import lucid
import lucid.nn as nn
from lucid.models import (
    CausalLMOutput,
    GenerationMixin,
    LanguageModelConfig,
)

# ─────────────────────────────────────────────────────────────────────────────
# LanguageModelConfig
# ─────────────────────────────────────────────────────────────────────────────


class TestLanguageModelConfig:
    def test_defaults(self) -> None:
        cfg = LanguageModelConfig()
        assert cfg.vocab_size == 30_522
        assert cfg.hidden_size == 768
        assert cfg.num_attention_heads == 12
        assert cfg.hidden_size % cfg.num_attention_heads == 0

    def test_head_divisibility_violation(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            LanguageModelConfig(hidden_size=10, num_attention_heads=3)

    def test_negative_vocab(self) -> None:
        with pytest.raises(ValueError, match="vocab_size"):
            LanguageModelConfig(vocab_size=0)

    def test_dropout_bounds(self) -> None:
        with pytest.raises(ValueError, match="hidden_dropout"):
            LanguageModelConfig(hidden_dropout=1.5)
        with pytest.raises(ValueError, match="attention_dropout"):
            LanguageModelConfig(attention_dropout=-0.1)

    def test_layer_norm_eps_positive(self) -> None:
        with pytest.raises(ValueError, match="layer_norm_eps"):
            LanguageModelConfig(layer_norm_eps=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# GenerationMixin
# ─────────────────────────────────────────────────────────────────────────────


class _DeterministicLM(GenerationMixin, nn.Module):
    """Test fixture: always predicts token (last_token + 1) % vocab."""

    def __init__(self, vocab: int = 7, eos: int | None = None) -> None:
        super().__init__()
        self.vocab = vocab
        self.config = type(
            "C", (), {"eos_token_id": eos, "pad_token_id": 0, "vocab_size": vocab}
        )()

    def forward(self, input_ids: lucid.Tensor) -> CausalLMOutput:
        B = int(input_ids.shape[0])
        T = int(input_ids.shape[1])
        rows: list[list[list[float]]] = []
        for b in range(B):
            seq_logits: list[list[float]] = []
            for t in range(T):
                tok = int(input_ids[b, t].item())
                row = [0.0] * self.vocab
                row[(tok + 1) % self.vocab] = 10.0
                seq_logits.append(row)
            rows.append(seq_logits)
        logits = lucid.tensor(rows)
        return CausalLMOutput(logits=logits, loss=None)


class TestGenerationMixin:
    def test_greedy_extends_correctly(self) -> None:
        m = _DeterministicLM(vocab=7)
        prompt = lucid.tensor([[1, 2]]).long()
        out = m.generate(prompt, max_length=5)
        assert tuple(out.shape) == (1, 5)
        # Each step picks (last + 1) % 7 → 1, 2, 3, 4, 5
        assert [int(out[0, t].item()) for t in range(5)] == [1, 2, 3, 4, 5]

    def test_max_new_tokens_takes_precedence(self) -> None:
        m = _DeterministicLM(vocab=7)
        prompt = lucid.tensor([[1, 2]]).long()
        out = m.generate(prompt, max_length=100, max_new_tokens=2)
        assert tuple(out.shape) == (1, 4)

    def test_eos_stops_and_pads(self) -> None:
        # eos_token_id = 4 → after producing 4, should pad with 0.
        m = _DeterministicLM(vocab=7, eos=4)
        prompt = lucid.tensor([[1, 2]]).long()
        out = m.generate(prompt, max_length=7, eos_token_id=4)
        toks = [int(out[0, t].item()) for t in range(int(out.shape[1]))]
        # 1, 2, 3, 4 then padded to length 7 with pad_token_id=0
        assert toks[:4] == [1, 2, 3, 4]
        assert all(t == 0 for t in toks[4:])

    def test_sampling_runs(self) -> None:
        m = _DeterministicLM(vocab=7)
        prompt = lucid.tensor([[1, 2]]).long()
        out = m.generate(
            prompt, max_length=5, do_sample=True, temperature=0.7, top_k=3, top_p=0.9
        )
        assert tuple(out.shape) == (1, 5)

    def test_input_must_be_2d(self) -> None:
        m = _DeterministicLM(vocab=7)
        with pytest.raises(ValueError, match="2-D"):
            m.generate(lucid.tensor([1, 2, 3]).long())
