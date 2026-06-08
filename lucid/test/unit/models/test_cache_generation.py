"""Cached-decoding correctness across every generation path lucid caches.

The core invariant: incremental decoding with a KV cache must produce
**token-identical** output (and bit-close logits) to re-encoding the full
prefix every step.  Verified for GPT-1, GPT-2 (decoder-only) and the
encoder-decoder Transformer.
"""

import lucid
from lucid.models.text.gpt._config import GPTConfig
from lucid.models.text.gpt._model import GPTLMHeadModel
from lucid.models.text.gpt2._config import GPT2Config
from lucid.models.text.gpt2._model import GPT2LMHeadModel
from lucid.models.text.transformer._config import TransformerConfig
from lucid.models.text.transformer._model import TransformerForSeq2SeqLM
from lucid.test._helpers.compare import assert_close
from lucid.utils.cache import DynamicCache


def _gpt2() -> GPT2LMHeadModel:
    cfg = GPT2Config(
        vocab_size=50,
        hidden_size=32,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    return GPT2LMHeadModel(cfg).eval()


def _gpt() -> GPTLMHeadModel:
    cfg = GPTConfig(
        vocab_size=50,
        hidden_size=32,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    return GPTLMHeadModel(cfg).eval()


def _seq2seq() -> TransformerForSeq2SeqLM:
    cfg = TransformerConfig(
        vocab_size=50,
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_decoder_layers=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    return TransformerForSeq2SeqLM(cfg).eval()


def _decoder_only_incremental_equals_full(model: object) -> None:
    lucid.manual_seed(0)
    ids = lucid.tensor([[3, 7, 1, 9, 2, 8]]).long()
    T = int(ids.shape[1])

    full = model(ids).logits  # type: ignore[operator]

    cache = DynamicCache()
    steps = []
    last = None
    for t in range(T):
        out = model(ids[:, t : t + 1], use_cache=True, past_key_values=cache)  # type: ignore[operator]
        steps.append(out.logits[:, -1, :])
        last = out
    inc = lucid.stack(steps, dim=1)

    assert_close(inc, full, atol=1e-4)
    assert cache.get_seq_length() == T
    # past_key_values populated, one (key, value) pair per layer, (B, H, T, D)
    assert last is not None and last.past_key_values is not None
    assert len(last.past_key_values) == 2
    assert tuple(last.past_key_values.key_cache[0].shape) == (1, 2, T, 16)


class TestGPT2Cache:
    def test_incremental_equals_full(self) -> None:
        _decoder_only_incremental_equals_full(_gpt2())

    def test_generate_cached_equals_uncached(self) -> None:
        lucid.manual_seed(0)
        m = _gpt2()
        prompt = lucid.tensor([[3, 7, 1]]).long()
        g_cache = m.generate(prompt, max_new_tokens=6, do_sample=False, use_cache=True)
        g_nocache = m.generate(
            prompt, max_new_tokens=6, do_sample=False, use_cache=False
        )
        assert_close(g_cache, g_nocache, atol=0.0)


class TestGPTCache:
    def test_incremental_equals_full(self) -> None:
        _decoder_only_incremental_equals_full(_gpt())

    def test_generate_cached_equals_uncached(self) -> None:
        lucid.manual_seed(0)
        m = _gpt()
        prompt = lucid.tensor([[3, 7, 1]]).long()
        g_cache = m.generate(prompt, max_new_tokens=6, do_sample=False, use_cache=True)
        g_nocache = m.generate(
            prompt, max_new_tokens=6, do_sample=False, use_cache=False
        )
        assert_close(g_cache, g_nocache, atol=0.0)


class TestTransformerSeq2SeqCache:
    def test_decode_incremental_equals_full(self) -> None:
        lucid.manual_seed(0)
        m = _seq2seq()
        src = lucid.tensor([[3, 7, 1, 9, 2]]).long()
        tgt = lucid.tensor([[1, 4, 2, 5, 6]]).long()
        T = int(tgt.shape[1])

        memory = m.transformer.encode(src)
        full = m.transformer.decode(tgt, memory)  # (B, T, d)

        from lucid.utils.cache import EncoderDecoderCache

        cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
        steps = []
        for t in range(T):
            dec = m.transformer.decode(
                tgt[:, t : t + 1],
                memory,
                past_key_value=cache,
                use_cache=True,
            )
            steps.append(dec[:, -1, :])
        inc = lucid.stack(steps, dim=1)

        assert_close(inc, full, atol=1e-4)
        assert cache.self_attention_cache.get_seq_length() == T
        # cross-attention computed once, then reused
        assert cache.cross_attention_cache.get_seq_length() == int(memory.shape[1])
        assert all(cache.is_updated.values())

    def test_generate_cached_equals_uncached(self) -> None:
        lucid.manual_seed(0)
        m = _seq2seq()
        src = lucid.tensor([[3, 7, 1, 9, 2]]).long()
        g_cache = m.generate(src, max_length=8, use_cache=True)
        g_nocache = m.generate(src, max_length=8, use_cache=False)
        assert_close(g_cache, g_nocache, atol=0.0)
