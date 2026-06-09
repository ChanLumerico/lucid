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

    def test_static_cache_equals_dynamic(self) -> None:
        lucid.manual_seed(0)
        m = _gpt2()
        prompt = lucid.tensor([[3, 7, 1]]).long()
        g_static = m.generate(
            prompt,
            max_new_tokens=6,
            do_sample=False,
            use_cache=True,
            cache_implementation="static",
        )
        g_dynamic = m.generate(
            prompt,
            max_new_tokens=6,
            do_sample=False,
            use_cache=True,
            cache_implementation="dynamic",
        )
        assert_close(g_static, g_dynamic, atol=0.0)

    def test_static_cache_oversized_equals_dynamic(self) -> None:
        # An OVER-SIZED max_cache_len (buffer far larger than tokens decoded) must
        # be both correct AND free: the read_len narrowing makes eager StaticCache
        # attend over only the filled prefix, so an oversized buffer stays
        # bit-identical to DynamicCache (no O(max_cache_len) attention penalty).
        lucid.manual_seed(0)
        m = _gpt2()
        prompt = lucid.tensor([[3, 7, 1]]).long()
        g_static = m.generate(
            prompt,
            max_new_tokens=6,
            do_sample=False,
            use_cache=True,
            cache_implementation="static",
            max_cache_len=32,  # 3 + 6 = 9 filled, buffer is 32 → mostly empty
        )
        g_dynamic = m.generate(
            prompt, max_new_tokens=6, do_sample=False, use_cache=True
        )
        assert_close(g_static, g_dynamic, atol=0.0)

    def test_static_cache_incremental_equals_full(self) -> None:
        from lucid.utils.cache import StaticCache

        lucid.manual_seed(0)
        m = _gpt2()
        ids = lucid.tensor([[3, 7, 1, 9, 2, 8]]).long()
        T = int(ids.shape[1])
        full = m(ids).logits
        cache = StaticCache(max_cache_len=16)
        steps = []
        for t in range(T):
            out = m(ids[:, t : t + 1], use_cache=True, past_key_values=cache)
            steps.append(out.logits[:, -1, :])
        inc = lucid.stack(steps, dim=1)
        assert_close(inc, full, atol=1e-4)
        assert cache.get_seq_length() == T
        assert tuple(cache.key_cache[0].shape) == (1, 2, 16, 16)  # fixed buffer

    def test_compiled_static_decode_equals_dynamic(self) -> None:
        lucid.manual_seed(0)
        m = _gpt2()
        prompt = lucid.tensor([[3, 7, 1, 9]]).long()
        g_compiled = m.generate(
            prompt,
            max_new_tokens=8,
            do_sample=False,
            cache_implementation="static",
            compile_decode=True,
        )
        g_dynamic = m.generate(prompt, max_new_tokens=8, do_sample=False)
        assert_close(g_compiled, g_dynamic, atol=0.0)

    def test_compiled_static_decode_batch(self) -> None:
        lucid.manual_seed(0)
        m = _gpt2()
        prompt = lucid.tensor([[3, 7, 1, 9], [2, 5, 8, 0]]).long()
        g_compiled = m.generate(
            prompt,
            max_new_tokens=6,
            do_sample=False,
            cache_implementation="static",
            compile_decode=True,
        )
        g_dynamic = m.generate(prompt, max_new_tokens=6, do_sample=False)
        assert_close(g_compiled, g_dynamic, atol=0.0)


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

    def test_static_cache_equals_dynamic(self) -> None:
        # GPT-1's eager StaticCache path uses the past_len-slice branches (its
        # trunk does not default cache_position), distinct from GPT-2's — cover
        # it explicitly so a future edit can't silently regress it.
        lucid.manual_seed(0)
        m = _gpt()
        prompt = lucid.tensor([[3, 7, 1]]).long()
        g_static = m.generate(
            prompt,
            max_new_tokens=6,
            do_sample=False,
            cache_implementation="static",
        )
        g_dynamic = m.generate(prompt, max_new_tokens=6, do_sample=False)
        assert_close(g_static, g_dynamic, atol=0.0)

    def test_compiled_static_decode_equals_dynamic(self) -> None:
        lucid.manual_seed(0)
        m = _gpt()
        prompt = lucid.tensor([[3, 7, 1, 9]]).long()
        g_compiled = m.generate(
            prompt,
            max_new_tokens=8,
            do_sample=False,
            cache_implementation="static",
            compile_decode=True,
        )
        g_dynamic = m.generate(prompt, max_new_tokens=8, do_sample=False)
        assert_close(g_compiled, g_dynamic, atol=0.0)


class TestCompiledStaticDecode:
    """The compiled StaticCache decode driver — capability guard, the
    ``from_buffers`` view constructor, and the single-executable property."""

    def test_capability_flag_defaults_false(self) -> None:
        from lucid.models._mixins import CausalLMMixin

        assert CausalLMMixin.supports_compiled_static_decode is False
        assert GPT2LMHeadModel.supports_compiled_static_decode is True
        assert GPTLMHeadModel.supports_compiled_static_decode is True

    def test_unsupported_host_falls_back_to_eager(self) -> None:
        # A CausalLMMixin host that does NOT declare static-decode support
        # must silently ignore compile_decode and stay token-identical to eager.
        class _Unsupported(GPT2LMHeadModel):
            supports_compiled_static_decode = False

        lucid.manual_seed(0)
        m = _Unsupported(
            GPT2Config(
                vocab_size=50,
                hidden_size=32,
                num_attention_heads=2,
                num_hidden_layers=2,
                intermediate_size=64,
                max_position_embeddings=32,
            )
        ).eval()
        assert m.supports_compiled_static_decode is False
        prompt = lucid.tensor([[3, 7, 1, 9]]).long()
        g_flagged = m.generate(
            prompt,
            max_new_tokens=6,
            do_sample=False,
            cache_implementation="static",
            compile_decode=True,
        )
        g_eager = m.generate(prompt, max_new_tokens=6, do_sample=False)
        assert_close(g_flagged, g_eager, atol=0.0)

    def test_from_buffers_roundtrip(self) -> None:
        from lucid.utils.cache import StaticCache

        lucid.manual_seed(0)
        m = _gpt2()
        ids = lucid.tensor([[3, 7, 1, 9]]).long()
        # Prefill a real StaticCache, then re-wrap its buffers via from_buffers.
        cache = StaticCache(max_cache_len=16)
        m(ids, use_cache=True, past_key_values=cache)
        view = StaticCache.from_buffers(
            cache.key_cache, cache.value_cache, cache.max_cache_len
        )
        assert len(view) == len(cache)
        assert view.max_cache_len == 16
        assert view.get_seq_length() == 0  # counter resets; position is authoritative
        assert view.key_cache[0] is cache.key_cache[0]  # buffers shared by reference

    def test_target_exceeds_max_position_raises(self) -> None:
        # StaticCache writes by absolute position; a target past the model's
        # positional range must raise (not silently corrupt) on every static
        # path — the compiled path's counter reset defeats the trunk's guard.
        import pytest

        lucid.manual_seed(0)
        m = _gpt2()  # max_position_embeddings=32
        prompt = lucid.tensor([[3, 7, 1, 9]]).long()
        for compile_decode in (False, True):
            with pytest.raises(ValueError, match="max_position_embeddings"):
                m.generate(
                    prompt,
                    max_length=40,
                    do_sample=False,
                    cache_implementation="static",
                    compile_decode=compile_decode,
                )

    def test_max_cache_len_too_small_raises(self) -> None:
        import pytest

        lucid.manual_seed(0)
        m = _gpt2()
        prompt = lucid.tensor([[3, 7, 1, 9]]).long()
        for compile_decode in (False, True):
            with pytest.raises(ValueError, match="max_cache_len"):
                m.generate(
                    prompt,
                    max_new_tokens=10,
                    do_sample=False,
                    cache_implementation="static",
                    max_cache_len=8,  # < 4 + 10 target
                    compile_decode=compile_decode,
                )

    def test_single_compile_across_positions(self, device_gpu_only: str) -> None:
        # The payoff property: positions sharing a read-window bucket reuse ONE
        # executable.  Positions 4..7 all map to ceil_pow2(pos+1) == 8.
        from lucid.models._utils._compiled_decode import (
            _CompiledStaticDecoder,
            _DECODE_CACHE_ATTR,
        )
        from lucid.utils.cache import StaticCache

        lucid.manual_seed(0)
        m = _gpt2().to(device_gpu_only)
        ids = lucid.tensor([[3, 7, 1, 9]], device=device_gpu_only).long()
        cache = StaticCache(max_cache_len=16)
        m(ids, use_cache=True, past_key_values=cache)
        decoder = _CompiledStaticDecoder(m, cache)
        tok = lucid.tensor([[5]], device=device_gpu_only).long()
        for pos in (4, 5, 6, 7):
            cp = lucid.tensor([pos], device=device_gpu_only).long()
            decoder.step(tok, cp, pos)
        cm_cache = getattr(m, _DECODE_CACHE_ATTR)
        assert len(cm_cache) == 1  # single bucket (8) → single executable
        compiled = next(iter(cm_cache.values()))
        assert len(compiled._cache) == 1
        assert len(compiled._eager_only.snapshot()) == 0

    def test_bucket_ladder_bounded_and_token_identical(
        self, device_gpu_only: str
    ) -> None:
        # Crossing a bucket boundary (8 → 16) compiles a second executable but the
        # count stays bounded (<= log2(max_cache_len)+1), and the decoded tokens
        # are bit-identical to the eager StaticCache path across the boundary.
        from lucid.models._utils._compiled_decode import _DECODE_CACHE_ATTR

        lucid.manual_seed(0)
        m = _gpt2().to(device_gpu_only)
        prompt = lucid.tensor([[3, 7, 1, 9, 2, 4]], device=device_gpu_only).long()
        # decode 8 tokens: fill goes 6→14, crossing the 8→16 bucket boundary.
        g_compiled = m.generate(
            prompt,
            max_new_tokens=8,
            do_sample=False,
            use_cache=True,
            cache_implementation="static",
            max_cache_len=16,
            compile_decode=True,
        )
        g_eager = m.generate(
            prompt,
            max_new_tokens=8,
            do_sample=False,
            use_cache=True,
            cache_implementation="static",
            max_cache_len=16,
        )
        assert_close(g_compiled, g_eager, atol=0.0)
        cm_cache = getattr(m, _DECODE_CACHE_ATTR)
        # buckets touched: 8 and 16 → 2 executables; bounded by log2(16)+1 = 5.
        assert 1 <= len(cm_cache) <= 5


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

    def test_static_encoder_decoder_cache_decode_equals_dynamic(self) -> None:
        # The eager StaticCache enc-dec decode path (self-attn cache_position
        # write + masked future tail + constant cross) must match DynamicCache.
        from lucid.utils.cache import EncoderDecoderCache, StaticCache

        lucid.manual_seed(0)
        m = _seq2seq()
        src = lucid.tensor([[3, 7, 1, 9, 2]]).long()
        tgt = [1, 4, 2, 5, 6, 3]
        memory = m.transformer.encode(src)
        s_len = int(memory.shape[1])

        dyn = EncoderDecoderCache(DynamicCache(), DynamicCache())
        sta = EncoderDecoderCache(StaticCache(max_cache_len=16), StaticCache(s_len))
        dlog, slog = [], []
        for t, tok in enumerate(tgt):
            tok_t = lucid.tensor([[tok]]).long()
            d = m.transformer.decode(tok_t, memory, past_key_value=dyn, use_cache=True)
            s = m.transformer.decode(
                tok_t,
                memory,
                past_key_value=sta,
                use_cache=True,
                cache_position=lucid.tensor([t]).long(),
            )
            dlog.append(d[:, -1, :])
            slog.append(s[:, -1, :])
        assert_close(lucid.stack(slog, dim=1), lucid.stack(dlog, dim=1), atol=1e-4)
        assert all(sta.is_updated.values())  # cross filled once
        # fixed self-attention buffer; cross capacity == source length
        assert tuple(sta.self_attention_cache.key_cache[0].shape)[2] == 16
        assert tuple(sta.cross_attention_cache.key_cache[0].shape)[2] == s_len

    def test_compiled_decode_equals_eager(self) -> None:
        lucid.manual_seed(0)
        m = _seq2seq()
        src = lucid.tensor([[3, 7, 1, 9, 2]]).long()
        g_eager = m.generate(src, max_length=12, do_sample=False)
        g_compiled = m.generate(
            src, max_length=12, do_sample=False, compile_decode=True
        )
        assert_close(g_compiled, g_eager, atol=0.0)

    def test_compiled_decode_batch(self) -> None:
        lucid.manual_seed(0)
        m = _seq2seq()
        src = lucid.tensor([[3, 7, 1, 9, 2], [5, 5, 8, 0, 4]]).long()
        g_eager = m.generate(src, max_length=10, do_sample=False)
        g_compiled = m.generate(
            src, max_length=10, do_sample=False, compile_decode=True
        )
        assert_close(g_compiled, g_eager, atol=0.0)

    def test_compiled_decode_sampling_matches_eager(self) -> None:
        m = _seq2seq()
        src = lucid.tensor([[3, 7, 1, 9, 2]]).long()
        lucid.manual_seed(7)
        g_eager = m.generate(
            src, max_length=12, do_sample=True, top_k=8, temperature=0.9
        )
        lucid.manual_seed(7)
        g_compiled = m.generate(
            src,
            max_length=12,
            do_sample=True,
            top_k=8,
            temperature=0.9,
            compile_decode=True,
        )
        assert_close(g_compiled, g_eager, atol=0.0)

    def test_compiled_decode_logits_parity(self) -> None:
        # Degeneracy-independent: full per-step logits through the compiled
        # driver must match the eager DynamicCache logits.
        from lucid.models._utils._compiled_decode import _CompiledSeq2SeqDecoder
        from lucid.utils.cache import EncoderDecoderCache, StaticCache

        lucid.manual_seed(0)
        m = _seq2seq()
        src = lucid.tensor([[3, 7, 1, 9, 2]]).long()
        toks = [0, 4, 2, 9, 5, 1]
        memory = m.transformer.encode(src)
        s_len = int(memory.shape[1])

        dyn = EncoderDecoderCache(DynamicCache(), DynamicCache())
        elog = []
        for tok in toks:
            d = m.transformer.decode(
                lucid.tensor([[tok]]).long(), memory, past_key_value=dyn, use_cache=True
            )
            elog.append(m.lm_head(d[:, -1, :]))

        past = EncoderDecoderCache(StaticCache(max_cache_len=16), StaticCache(s_len))
        d0 = m.transformer.decode(
            lucid.tensor([[toks[0]]]).long(),
            memory,
            past_key_value=past,
            use_cache=True,
            cache_position=lucid.tensor([0]).long(),
        )
        clog = [m.lm_head(d0[:, -1, :])]
        mem_mask = lucid.ones((1, int(memory.shape[1])))
        decoder = _CompiledSeq2SeqDecoder(m, past, memory, mem_mask)
        for t in range(1, len(toks)):
            lg = decoder.step(
                lucid.tensor([[toks[t]]]).long(), lucid.tensor([t]).long(), t
            )
            clog.append(lg[:, -1, :])
        assert_close(lucid.stack(clog, dim=1), lucid.stack(elog, dim=1), atol=1e-4)

    def test_compiled_decode_max_length_exceeds_max_pos_raises(self) -> None:
        import pytest

        lucid.manual_seed(0)
        m = _seq2seq()  # max_position_embeddings=32
        src = lucid.tensor([[3, 7, 1, 9, 2]]).long()
        with pytest.raises(ValueError, match="max_position_embeddings"):
            m.generate(src, max_length=40, compile_decode=True)

    def test_bucket_ladder_bounded_across_positions(
        self, device_gpu_only: str
    ) -> None:
        # The self-attention read window grows in power-of-two buckets, so a
        # contiguous decode touches a BOUNDED number of executables
        # (<= log2(self_max_cache_len)+1), each reused within its bucket — never
        # an eager fallback.  Positions 1..4 cross buckets {2, 4, 8}.
        from lucid.models._utils._compiled_decode import (
            _CompiledSeq2SeqDecoder,
            _DECODE_CACHE_ATTR,
        )
        from lucid.utils.cache import EncoderDecoderCache, StaticCache

        lucid.manual_seed(0)
        m = _seq2seq().to(device_gpu_only)
        src = lucid.tensor([[3, 7, 1, 9, 2]], device=device_gpu_only).long()
        memory = m.transformer.encode(src)
        s_len = int(memory.shape[1])
        past = EncoderDecoderCache(StaticCache(max_cache_len=16), StaticCache(s_len))
        m.transformer.decode(
            lucid.tensor([[1]], device=device_gpu_only).long(),
            memory,
            past_key_value=past,
            use_cache=True,
            cache_position=lucid.tensor([0], device=device_gpu_only).long(),
        )
        mem_mask = lucid.ones((1, int(memory.shape[1])), device=device_gpu_only)
        decoder = _CompiledSeq2SeqDecoder(m, past, memory, mem_mask)
        tok = lucid.tensor([[5]], device=device_gpu_only).long()
        for pos in (1, 2, 3, 4):
            decoder.step(tok, lucid.tensor([pos], device=device_gpu_only).long(), pos)
        cm_cache = getattr(m, _DECODE_CACHE_ATTR)
        assert 1 <= len(cm_cache) <= 5  # bounded by log2(16)+1
        for compiled in cm_cache.values():
            assert len(compiled._eager_only.snapshot()) == 0

    def test_decoder_applies_padding_masks(self) -> None:
        # Regression: TransformerDecoder.forward dropped tgt/memory key-padding
        # masks before the layers.  Correctness check: with the mask applied,
        # the decoder output must be INVARIANT to whatever tokens sit in the
        # padded source positions (encoder self-attn + cross-attn both mask).
        lucid.manual_seed(0)
        m = _seq2seq()
        amask = lucid.tensor([[1.0, 1, 1, 1, 1, 0, 0]])
        dec_in = lucid.tensor([[1, 4, 2]]).long()
        src_a = lucid.tensor([[3, 7, 1, 9, 2, 0, 0]]).long()
        src_b = lucid.tensor([[3, 7, 1, 9, 2, 5, 8]]).long()  # differs only in pad cols
        out_a = m.transformer.decode(
            dec_in, m.transformer.encode(src_a, amask), memory_attention_mask=amask
        )
        out_b = m.transformer.decode(
            dec_in, m.transformer.encode(src_b, amask), memory_attention_mask=amask
        )
        assert_close(out_a, out_b, atol=1e-4)  # padded positions do not leak
        # and dropping the mask DOES let them leak (the bug's signature)
        leak_a = m.transformer.decode(dec_in, m.transformer.encode(src_a, amask))
        leak_b = m.transformer.decode(dec_in, m.transformer.encode(src_b, amask))
        assert float((leak_a - leak_b).abs().max().item()) > 1e-3

    def test_compiled_decode_padded_source_equals_eager(self) -> None:
        lucid.manual_seed(0)
        m = _seq2seq()
        src = lucid.tensor([[3, 7, 1, 9, 2, 0, 0]]).long()
        amask = lucid.tensor([[1.0, 1, 1, 1, 1, 0, 0]])
        g_eager = m.generate(src, max_length=12, do_sample=False, attention_mask=amask)
        g_compiled = m.generate(
            src,
            max_length=12,
            do_sample=False,
            attention_mask=amask,
            compile_decode=True,
        )
        assert_close(g_compiled, g_eager, atol=0.0)
