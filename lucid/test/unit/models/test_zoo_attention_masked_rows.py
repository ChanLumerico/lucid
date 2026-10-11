"""Zoo attention takes one masked-softmax core on every path (LCD-387).

BERT's ``output_attentions`` path and the attention-dropout paths of GPT,
GPT-2 and RoFormer each spelled ``softmax(q kᵀ + mask) v`` by hand.  A query
row whose every key is ``-inf`` came out NaN there but ``0`` from the fused
kernel, so the answer depended on whether the caller asked for weights or
was training.  Every path now runs through
``lucid.nn.functional.attention._masked_attention``, which owns the rule.
"""

import ast
from pathlib import Path

import pytest

import lucid
from lucid.models.text.bert._config import BERTConfig
from lucid.models.text.bert._model import _BERTSelfAttention
from lucid.models.text.gpt._config import GPTConfig
from lucid.models.text.gpt._model import _GPTSelfAttention
from lucid.models.text.gpt2._config import GPT2Config
from lucid.models.text.gpt2._model import _GPT2SelfAttention
from lucid.models.text.roformer._config import RoFormerConfig
from lucid.models.text.roformer._model import _RoFormerSelfAttention

_MODELS = Path(__file__).resolve().parents[3] / "models"
B, T, HIDDEN, HEADS = 2, 4, 16, 2
_SMALL = {
    "hidden_size": HIDDEN,
    "num_attention_heads": HEADS,
    "num_hidden_layers": 1,
    "intermediate_size": 32,
    "vocab_size": 32,
    "hidden_dropout": 0.0,
}


def _mask_with_dead_row() -> lucid.Tensor:
    """``(B, 1, T, T)`` additive mask; batch 0, query 1 sees no key."""
    keep = lucid.ones((B, 1, T, T), dtype=lucid.bool_)
    keep[0, 0, 1, :] = False
    zeros = lucid.zeros((B, 1, T, T))
    return lucid.masked_fill(zeros, lucid.logical_not(keep), float("-inf"))


def _finite(t: lucid.Tensor) -> bool:
    return bool(lucid.isfinite(t).all().item())


def test_bert_weights_path_matches_the_fused_path() -> None:
    lucid.manual_seed(0)
    attn = _BERTSelfAttention(BERTConfig(**_SMALL)).eval()
    x = lucid.randn(B, T, HIDDEN)
    mask = _mask_with_dead_row()

    fused, _ = attn(x, attention_mask=mask)
    explicit, weights = attn(x, attention_mask=mask, output_attentions=True)

    assert weights is not None and _finite(weights) and _finite(explicit)
    assert float((fused - explicit).abs().max().item()) < 1e-5
    assert float(explicit[0, 1].abs().max().item()) == 0.0  # attends to nothing
    assert float(weights[0, :, 1].abs().max().item()) == 0.0


def test_bert_head_mask_removes_a_head() -> None:
    lucid.manual_seed(1)
    attn = _BERTSelfAttention(BERTConfig(**_SMALL)).eval()
    x = lucid.randn(B, T, HIDDEN)
    full, _ = attn(x, attention_mask=_mask_with_dead_row())
    half, w = attn(
        x, attention_mask=_mask_with_dead_row(), head_mask=lucid.tensor([1.0, 0.0])
    )
    d = HIDDEN // HEADS
    assert _finite(half)
    assert float((half[..., :d] - full[..., :d]).abs().max().item()) < 1e-5
    assert float(half[..., d:].abs().max().item()) == 0.0
    assert w is None  # weights only when asked for


def _gpt(cls: type, cfg: type) -> lucid.nn.Module:
    return cls(cfg(**_SMALL, attention_dropout=0.5, max_position_embeddings=8))


@pytest.mark.parametrize(
    "build",
    [
        lambda: _gpt(_GPTSelfAttention, GPTConfig),
        lambda: _gpt(_GPT2SelfAttention, GPT2Config),
    ],
    ids=["gpt", "gpt2"],
)
def test_causal_lm_dropout_path_keeps_a_dead_row_finite(build) -> None:
    lucid.manual_seed(2)
    attn = build()
    x = lucid.randn(B, T, HIDDEN)
    # Padding that bans every key for batch 0 leaves its rows no key at all.
    pad = lucid.zeros((B, 1, 1, T))
    pad[0] = float("-inf")
    eval_out = attn.eval()(x, attention_mask=pad)
    train_out = attn.train()(x, attention_mask=pad)
    assert _finite(train_out) and _finite(eval_out)
    # Both attend to nothing, so both rows are the projection bias.
    assert float((train_out[0] - eval_out[0]).abs().max().item()) < 1e-5


def test_roformer_dropout_path_keeps_a_dead_row_finite() -> None:
    lucid.manual_seed(3)
    attn = _RoFormerSelfAttention(
        RoFormerConfig(**_SMALL, attention_dropout=0.5, max_position_embeddings=8)
    )
    x = lucid.randn(B, T, HIDDEN)
    cos = lucid.ones((T, HIDDEN // HEADS))
    sin = lucid.zeros((T, HIDDEN // HEADS))
    mask = _mask_with_dead_row()
    eval_out = attn.eval()(x, cos, sin, attention_mask=mask)
    train_out = attn.train()(x, cos, sin, attention_mask=mask)
    assert _finite(train_out) and _finite(eval_out)
    assert float(train_out[0, 1].abs().max().item()) == 0.0
    assert float(eval_out[0, 1].abs().max().item()) == 0.0


def _hand_rolled_dual_paths(source: str) -> list[str]:
    """Functions that call the fused attention *and* a softmax themselves —
    a second, hand-written path that can drift from the masked core."""
    hits: list[str] = []
    for fn in ast.walk(ast.parse(source)):
        if not isinstance(fn, ast.FunctionDef):
            continue
        called = {
            (
                n.func.attr
                if isinstance(n.func, ast.Attribute)
                else getattr(n.func, "id", "")
            )
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
        }
        if {"scaled_dot_product_attention", "softmax"} <= called:
            hits.append(fn.name)
    return hits


def test_zoo_attention_has_no_hand_rolled_second_path() -> None:
    offenders = {
        str(p.relative_to(_MODELS)): hits
        for p in sorted(_MODELS.rglob("*.py"))
        if (hits := _hand_rolled_dual_paths(p.read_text()))
    }
    assert not offenders, (
        f"{offenders} pair the fused attention with a hand-written softmax; "
        "pass dropout_p to F.scaled_dot_product_attention, or call "
        "nn.functional.attention._masked_attention(need_weights=True)"
    )


def test_the_dual_path_check_can_fail() -> None:
    src = (
        "def forward(self, q, k, v):\n"
        "    if self.training:\n"
        "        return F.softmax(q @ k, dim=-1) @ v\n"
        "    return F.scaled_dot_product_attention(q, k, v)\n"
    )
    assert _hand_rolled_dual_paths(src) == ["forward"]
