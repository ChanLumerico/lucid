"""Unit tests for the text-domain Phase 4 base layer.

Covers:
- LanguageModelConfig validation
- CausalLMMixin greedy + sampling

(Positional-encoding tests — RoPE / sinusoidal PE — live in
``test/unit/nn/test_nn_positional.py`` since those primitives moved to
:mod:`lucid.nn` in 2026-05.)
"""

import pytest

import lucid
import lucid.nn as nn
from lucid.models import (
    CausalLMOutput,
    CausalLMMixin,
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
# CausalLMMixin
# ─────────────────────────────────────────────────────────────────────────────


class _DeterministicLM(CausalLMMixin, nn.Module):
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


class TestCausalLMMixin:
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


class TestSamplingFilters:
    """The vectorized on-device sampling primitives (no per-element CPU loops)."""

    def test_top_k_keeps_exactly_k(self) -> None:
        from lucid.models._mixins import _top_k_filter

        logits = lucid.tensor([[1.0, 5.0, 2.0, 4.0, 3.0]])
        out = _top_k_filter(logits, 2)  # keep 5.0, 4.0
        kept = [v > -1e8 for v in out[0].numpy().tolist()]
        assert kept == [False, True, False, True, False]

    def test_top_k_ge_vocab_is_identity(self) -> None:
        from lucid.models._mixins import _top_k_filter

        logits = lucid.tensor([[1.0, 2.0, 3.0]])
        out = _top_k_filter(logits, 5)
        assert out[0].numpy().tolist() == [1.0, 2.0, 3.0]

    def test_top_p_keeps_dominant_token(self) -> None:
        from lucid.models._mixins import _top_p_filter

        # softmax mass concentrates on index 1; a tight p keeps only it.
        logits = lucid.tensor([[0.0, 10.0, 0.1, 0.2]])
        out = _top_p_filter(logits, 0.5)
        kept = [v > -1e8 for v in out[0].numpy().tolist()]
        assert kept[1] is True  # argmax always kept
        assert sum(kept) == 1  # nucleus is just the dominant token

    def test_repetition_penalty_lowers_seen(self) -> None:
        from lucid.models._mixins import _apply_repetition_penalty

        logits = lucid.tensor([[2.0, 2.0, 2.0, 2.0]])
        prefix = lucid.tensor([[1, 3]]).long()  # tokens 1, 3 already generated
        out = _apply_repetition_penalty(logits, prefix, 2.0)
        row = out[0].numpy().tolist()
        assert row[0] == 2.0 and row[2] == 2.0  # unseen unchanged
        assert row[1] == 1.0 and row[3] == 1.0  # seen positive logit halved

    def test_multinomial_in_range_and_deterministic(self) -> None:
        from lucid.models._mixins import _multinomial_one

        probs = lucid.tensor([[0.1, 0.2, 0.3, 0.4]])
        lucid.manual_seed(0)
        a = _multinomial_one(probs, device="cpu")
        lucid.manual_seed(0)
        b = _multinomial_one(probs, device="cpu")
        assert int(a[0].item()) == int(b[0].item())  # same RNG → same draw
        assert 0 <= int(a[0].item()) < 4  # valid index
