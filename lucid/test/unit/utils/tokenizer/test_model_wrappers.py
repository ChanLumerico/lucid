"""Phase C — per-model tokenizer wrappers.

Each text-model family ships a `_tokenizer/` package with
`{Model}Tokenizer` + `{Model}TokenizerFast` subclassing the matching
base algorithm with model-specific defaults baked in (special-token
registry, normalizer settings).  These tests cover:

* The wrappers import from the family package's top-level `__init__`.
* Default special-tokens registry matches the model convention.
* Train / encode / decode work end-to-end.
* Python ↔ Fast parity on the wrapper level (same vocab → same ids).
"""

import pytest

from lucid.models.text.bert import BERTTokenizer, BERTTokenizerFast
from lucid.models.text.gpt import GPTTokenizer, GPTTokenizerFast
from lucid.models.text.gpt2 import GPT2Tokenizer, GPT2TokenizerFast
from lucid.models.text.roformer import (
    RoFormerTokenizer,
    RoFormerTokenizerFast,
)

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumps high",
    "the dog the dog the dog",
    "a quick fox runs",
] * 8


# ── BERT ────────────────────────────────────────────────────────────


class TestBERTTokenizer:
    def test_default_special_tokens(self) -> None:
        for cls in (BERTTokenizer, BERTTokenizerFast):
            tok = cls(vocab={})
            st = tok.special_tokens
            assert st.unk == "[UNK]"
            assert st.pad == "[PAD]"
            assert st.cls == "[CLS]"
            assert st.sep == "[SEP]"
            assert st.mask == "[MASK]"

    def test_train_encode_decode(self) -> None:
        for cls in (BERTTokenizer, BERTTokenizerFast):
            tok = cls(vocab={})
            tok.train(CORPUS, vocab_size=60)
            ids = tok.encode("the dog", add_special_tokens=False)
            assert isinstance(ids, list)
            assert len(ids) > 0

    def test_python_fast_parity(self) -> None:
        slow = BERTTokenizer(vocab={})
        slow.train(CORPUS, vocab_size=60)
        fast = BERTTokenizerFast(vocab=slow.get_vocab())
        for text in ["the dog", "the quick brown fox"]:
            assert slow.encode(text, add_special_tokens=False) == fast.encode(
                text, add_special_tokens=False
            )

    def test_lowercasing_default(self) -> None:
        """BERT default is uncased — encode("THE") should match
        encode("the") because of the bundled BERTNormalizer."""
        tok = BERTTokenizer(vocab={})
        tok.train(CORPUS, vocab_size=60)
        assert tok.encode("the dog", add_special_tokens=False) == tok.encode(
            "THE DOG", add_special_tokens=False
        )


# ── RoFormer ────────────────────────────────────────────────────────


class TestRoFormerTokenizer:
    def test_default_special_tokens(self) -> None:
        for cls in (RoFormerTokenizer, RoFormerTokenizerFast):
            tok = cls(vocab={})
            st = tok.special_tokens
            assert st.unk == "[UNK]"
            assert st.cls == "[CLS]"
            assert st.sep == "[SEP]"
            assert st.mask == "[MASK]"

    def test_python_fast_parity(self) -> None:
        slow = RoFormerTokenizer(vocab={})
        slow.train(CORPUS, vocab_size=60)
        fast = RoFormerTokenizerFast(vocab=slow.get_vocab())
        for text in ["the dog", "the quick brown fox"]:
            assert slow.encode(text, add_special_tokens=False) == fast.encode(
                text, add_special_tokens=False
            )


# ── GPT-1 ───────────────────────────────────────────────────────────


class TestGPTTokenizer:
    """GPT-1 is word BPE with ``</w>``, not GPT-2's byte-level scheme."""

    @staticmethod
    def _marked(cls: type = GPTTokenizer) -> GPTTokenizer:
        vocab = {"i": 0, "n": 1, "n</w>": 2, "in": 3, "in</w>": 4, "s": 5, "s</w>": 6}
        return cls(vocab=vocab, merges=[("i", "n</w>"), ("i", "n")])

    def test_end_of_word_separates_word_from_subword(self) -> None:
        """The same letters must not collapse onto the same id.

        Without the marker, ``in`` the word and ``in`` inside ``ins`` share
        one entry, and that entry silently means two things.
        """
        tok = self._marked()
        assert [tok.id_to_token(i) for i in tok.encode("in")] == ["in</w>"]
        assert [tok.id_to_token(i) for i in tok.encode("ins")] == ["in", "s</w>"]

    def test_round_trip_restores_word_boundaries(self) -> None:
        tok = self._marked()
        assert tok.decode(tok.encode("in ins")) == "in ins"

    def test_input_is_lowercased(self) -> None:
        """§4.1 lowercases the text, so the default normalizer must too."""
        tok = self._marked()
        assert tok.encode("IN") == tok.encode("in")

    def test_fast_matches_python(self) -> None:
        assert self._marked(GPTTokenizerFast).encode("ins") == self._marked().encode(
            "ins"
        )

    def test_train_encode(self) -> None:
        for cls in (GPTTokenizer, GPTTokenizerFast):
            tok = cls(vocab={}, merges=[])
            tok.train(CORPUS, vocab_size=80)
            ids = tok.encode("the dog", add_special_tokens=False)
            assert isinstance(ids, list)
            assert len(ids) > 0

    def test_python_fast_parity(self) -> None:
        slow = GPTTokenizer(vocab={}, merges=[])
        slow.train(CORPUS, vocab_size=80)
        fast = GPTTokenizerFast(vocab=slow.get_vocab(), merges=slow._merges)
        for text in ["the dog", "the quick brown fox"]:
            assert slow.encode(text, add_special_tokens=False) == fast.encode(
                text, add_special_tokens=False
            )


# ── GPT-2 ───────────────────────────────────────────────────────────


class TestGPT2Tokenizer:
    def test_default_endoftext(self) -> None:
        for cls in (GPT2Tokenizer, GPT2TokenizerFast):
            tok = cls(vocab={}, merges=[])
            st = tok.special_tokens
            assert st.bos == "<|endoftext|>"
            assert st.eos == "<|endoftext|>"
            assert st.unk == "<|endoftext|>"

    def test_train_encode(self) -> None:
        for cls in (GPT2Tokenizer, GPT2TokenizerFast):
            tok = cls(vocab={}, merges=[])
            tok.train(CORPUS, vocab_size=80)
            ids = tok.encode("the dog", add_special_tokens=False)
            assert isinstance(ids, list)
            assert len(ids) > 0

    def test_python_fast_parity(self) -> None:
        slow = GPT2Tokenizer(vocab={}, merges=[])
        slow.train(CORPUS, vocab_size=80)
        fast = GPT2TokenizerFast(vocab=slow.get_vocab(), merges=slow._merges)
        for text in ["the dog", "the quick brown fox"]:
            assert slow.encode(text, add_special_tokens=False) == fast.encode(
                text, add_special_tokens=False
            )

    def test_preserves_whitespace(self) -> None:
        """Byte-level BPE preserves spaces (unlike WordPiece / classical BPE)."""
        tok = GPT2Tokenizer(vocab={}, merges=[])
        tok.train(CORPUS, vocab_size=80)
        ids = tok.encode("the dog", add_special_tokens=False)
        assert " " in tok.decode(ids, skip_special_tokens=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
