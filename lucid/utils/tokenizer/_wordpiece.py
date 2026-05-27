"""WordPieceTokenizer — the algorithm BERT/DistilBERT/RoFormer use.

Encode algorithm (greedy longest-match)
---------------------------------------
For each pre-tokenized word:

1. If the whole word is in vocab, emit its id.
2. Otherwise, walk the word left-to-right finding the longest
   prefix that IS in vocab, emit its id, then recurse on the
   remainder — but the remainder must be looked up with the
   continuation prefix (``"##"`` by default) prepended to mark it
   as a non-initial subword.
3. If no valid prefix exists at any position, emit the UNK token
   (typically ``[UNK]``) for the entire word.

On-disk format
--------------
HF-compatible ``vocab.txt`` — one token per line, id = line index.
Continuation pieces start with ``##``.  Loads/saves any published
BERT-family checkpoint without modification.

Training
--------
Greedy frequency-based BPE-style training (matches HF
``WordPieceTrainer`` semantics; faster than the full log-likelihood
training in the original paper).  Standard pre-tokenizer chain is
:class:`BertNormalizer` + :class:`WhitespacePunctuationSplit` —
configure via the constructor if you need a different one.
"""

import os
from typing import Iterable

from lucid._C import engine as _C_engine

from lucid.utils.tokenizer._base import SpecialTokens, Tokenizer
from lucid.utils.tokenizer._lookup_common import (
    _load_special_tokens_map,
    _load_vocab_txt,
    _save_vocab_txt,
)
from lucid.utils.tokenizer._normalizers import BertNormalizer, Normalizer
from lucid.utils.tokenizer._pre_tokenizers import (
    PreTokenizer,
    WhitespacePunctuationSplit,
)


class _WordPieceCommonMixin:
    """Shared algorithm + post-processing for both WordPiece flavours."""

    _normalizer: Normalizer | None
    _pre_tokenizer: PreTokenizer
    _continuing_prefix: str

    def _prepare_words(self, text: str) -> list[str]:
        """Apply normalizer + pre-tokenizer; return word strings."""
        if self._normalizer is not None:
            text = self._normalizer(text)
        return [chunk for chunk, _ in self._pre_tokenizer(text)]

    def _decode_join(self, ids: list[int], id_to_token: dict[int, str]) -> str:
        """Standard BERT-style decode: join tokens; strip ``##`` from
        continuation pieces; emit a space before each non-continuation."""
        out: list[str] = []
        first = True
        for i in ids:
            tok = id_to_token.get(i)
            if tok is None:
                continue
            if tok.startswith(self._continuing_prefix):
                out.append(tok[len(self._continuing_prefix) :])
            else:
                if not first:
                    out.append(" ")
                out.append(tok)
            first = False
        return "".join(out)


# ── Pure-Python WordPiece ───────────────────────────────────────────


class WordPieceTokenizer(_WordPieceCommonMixin, Tokenizer):
    r"""Reference (pure-Python) WordPiece tokenizer.

    Parameters
    ----------
    vocab : dict[str, int]
        BERT-style vocab.  Continuation tokens prefixed with ``##``.
    unk_token : str, default "[UNK]"
        Token emitted when no valid longest-match prefix can be
        found.  Must be present in ``vocab`` for it to actually
        emit an id (otherwise emits silently nothing).
    continuing_prefix : str, default "##"
        Prefix for non-initial subwords.
    max_chars_per_word : int, default 100
        Words longer than this are immediately emitted as UNK
        (matches BERT's ``BasicTokenizer`` cap).
    normalizer : Normalizer, optional
        Default :class:`BertNormalizer`.
    pre_tokenizer : PreTokenizer, optional
        Default :class:`WhitespacePunctuationSplit`.
    special_tokens : SpecialTokens, optional
        Defaults to a registry with ``unk=unk_token`` so the
        tokenizer behaves correctly out of the box.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        *,
        unk_token: str = "[UNK]",
        continuing_prefix: str = "##",
        max_chars_per_word: int = 100,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        self._vocab: dict[str, int] = dict(vocab)
        self._id_to_token: dict[int, str] = {v: k for k, v in self._vocab.items()}
        self._unk_token = unk_token
        self._continuing_prefix = continuing_prefix
        self._max_chars_per_word = max_chars_per_word
        self._normalizer = normalizer if normalizer is not None else BertNormalizer()
        self._pre_tokenizer = (
            pre_tokenizer if pre_tokenizer is not None else WhitespacePunctuationSplit()
        )
        # Default special-tokens registry mirrors BERT's set when
        # the caller doesn't supply one.
        if special_tokens is None:
            special_tokens = SpecialTokens(unk=unk_token)
        super().__init__(special_tokens=special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def algo(self) -> str:
        return "wordpiece"

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def id_to_token(self, token_id: int) -> str | None:
        return self._id_to_token.get(token_id)

    def _encode_word(self, word: str) -> list[int]:
        """Apply greedy longest-match to one pre-tokenized word."""
        # Whole-word shortcut.
        if word in self._vocab:
            return [self._vocab[word]]
        if len(word) > self._max_chars_per_word:
            unk_id = self._vocab.get(self._unk_token)
            return [unk_id] if unk_id is not None else []
        # Greedy longest-match left-to-right.
        ids: list[int] = []
        start = 0
        while start < len(word):
            match_id = None
            match_end = start
            for end in range(len(word), start, -1):
                sub = word[start:end]
                if start > 0:
                    sub = self._continuing_prefix + sub
                tid = self._vocab.get(sub)
                if tid is not None:
                    match_id = tid
                    match_end = end
                    break
            if match_id is None:
                # No prefix matched at this position → whole word is UNK.
                unk_id = self._vocab.get(self._unk_token)
                return [unk_id] if unk_id is not None else []
            ids.append(match_id)
            start = match_end
        return ids

    def _encode_one(self, text: str) -> list[int]:
        out: list[int] = []
        for word in self._prepare_words(text):
            out.extend(self._encode_word(word))
        return out

    def _decode_one(self, ids: list[int]) -> str:
        return self._decode_join(ids, self._id_to_token)

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int = 30_000,
    ) -> None:
        """Greedy frequency-based WordPiece training (BPE-like).

        For each iteration, count adjacent symbol-pair frequencies
        across all words + merge the most frequent.  Same
        deterministic tie-break (lexicographic key order) as BPE.

        The result is a BERT-compatible vocab with ``##``
        continuation markers and the configured UNK token.
        """
        # Pre-tokenize + count word frequencies.
        word_freq: dict[str, int] = {}
        for doc in corpus:
            for w in self._prepare_words(doc):
                word_freq[w] = word_freq.get(w, 0) + 1
        # Seed vocab with the UNK token (id 0 reserved for it) + every
        # distinct codepoint that appears in any word (continuation
        # forms get ``##`` prefix).
        new_vocab: dict[str, int] = {self._unk_token: 0}
        next_id = 1
        # Per-word symbol sequence (head symbol is plain, tail symbols
        # carry the continuation prefix).
        words: list[tuple[list[str], int]] = []
        for w, freq in word_freq.items():
            syms: list[str] = []
            for i, ch in enumerate(w):
                tok = ch if i == 0 else self._continuing_prefix + ch
                if tok not in new_vocab:
                    new_vocab[tok] = next_id
                    next_id += 1
                syms.append(tok)
            words.append((syms, freq))
        # Greedy pair merging.
        while len(new_vocab) < vocab_size:
            pair_freq: dict[tuple[str, str], int] = {}
            for syms, freq in words:
                for k in range(len(syms) - 1):
                    pair_freq[(syms[k], syms[k + 1])] = (
                        pair_freq.get((syms[k], syms[k + 1]), 0) + freq
                    )
            if not pair_freq:
                break
            best_pair = max(pair_freq.items(), key=lambda kv: (kv[1], kv[0]))
            if best_pair[1] < 2:
                break
            left, right = best_pair[0]
            right_core = (
                right[len(self._continuing_prefix) :]
                if right.startswith(self._continuing_prefix)
                else right
            )
            merged = left + right_core
            new_vocab[merged] = next_id
            next_id += 1
            # Rewrite all word symbol sequences.
            new_words: list[tuple[list[str], int]] = []
            for syms, freq in words:
                out: list[str] = []
                k = 0
                while k < len(syms):
                    if k + 1 < len(syms) and syms[k] == left and syms[k + 1] == right:
                        out.append(merged)
                        k += 2
                    else:
                        out.append(syms[k])
                        k += 1
                new_words.append((out, freq))
            words = new_words
        self._vocab = new_vocab
        self._id_to_token = {v: k for k, v in new_vocab.items()}
        self._refresh_special_ids()

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        _save_vocab_txt(self._vocab, os.path.join(directory, "vocab.txt"))
        super().save(directory)

    def _save_extras(self) -> dict[str, object]:
        return {
            "model": {
                "type": "WordPiece",
                "vocab": self._vocab,
                "unk_token": self._unk_token,
                "continuing_prefix": self._continuing_prefix,
                "max_chars_per_word": self._max_chars_per_word,
            }
        }

    @classmethod
    def from_file(
        cls,
        directory: str,
        *,
        unk_token: str = "[UNK]",
        continuing_prefix: str = "##",
        max_chars_per_word: int = 100,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> "WordPieceTokenizer":
        """Load from a BERT-style directory containing ``vocab.txt``."""
        vocab_path = os.path.join(directory, "vocab.txt")
        if not os.path.isfile(vocab_path):
            raise FileNotFoundError(
                f"WordPieceTokenizer.from_file: vocab.txt not found in " f"{directory}"
            )
        vocab = _load_vocab_txt(vocab_path)
        st = special_tokens or _load_special_tokens_map(directory)
        return cls(
            vocab,
            unk_token=unk_token,
            continuing_prefix=continuing_prefix,
            max_chars_per_word=max_chars_per_word,
            normalizer=normalizer,
            pre_tokenizer=pre_tokenizer,
            special_tokens=st,
        )

    from_pretrained = from_file


# ── Fast (C++-backed) WordPiece ─────────────────────────────────────


class WordPieceTokenizerFast(_WordPieceCommonMixin, Tokenizer):
    """C++-backed WordPiece tokenizer.  See :class:`WordPieceTokenizer`.

    The greedy longest-match encode + training loop run in C++ via
    :class:`lucid._C.engine.utils.tokenizer.WordPiece`.  The Python
    side handles normalisation + pre-tokenization (so the same
    pre-processing chain works for both flavours).
    """

    def __init__(
        self,
        vocab: dict[str, int],
        *,
        unk_token: str = "[UNK]",
        continuing_prefix: str = "##",
        max_chars_per_word: int = 100,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        self._vocab: dict[str, int] = dict(vocab)
        self._id_to_token: dict[int, str] = {v: k for k, v in self._vocab.items()}
        self._unk_token = unk_token
        self._continuing_prefix = continuing_prefix
        self._max_chars_per_word = max_chars_per_word
        self._cpp = _C_engine.utils.tokenizer.WordPiece(
            self._vocab,
            unk_token,
            continuing_prefix,
            max_chars_per_word,
        )
        self._normalizer = normalizer if normalizer is not None else BertNormalizer()
        self._pre_tokenizer = (
            pre_tokenizer if pre_tokenizer is not None else WhitespacePunctuationSplit()
        )
        if special_tokens is None:
            special_tokens = SpecialTokens(unk=unk_token)
        super().__init__(special_tokens=special_tokens)

    @property
    def vocab_size(self) -> int:
        return self._cpp.vocab_size()

    @property
    def algo(self) -> str:
        return "wordpiece"

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def id_to_token(self, token_id: int) -> str | None:
        return self._id_to_token.get(token_id)

    def _encode_one(self, text: str) -> list[int]:
        """Normalize + pre-tokenize in Python; encode each word in C++."""
        out: list[int] = []
        for word in self._prepare_words(text):
            out.extend(self._cpp.encode(word))
        return out

    def _decode_one(self, ids: list[int]) -> str:
        return self._decode_join(ids, self._id_to_token)

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int = 30_000,
    ) -> None:
        """Train in C++ via :meth:`WordPiece.train`.

        Pre-encode the corpus through the configured normalizer +
        pre-tokenizer in Python so the C++ training loop sees
        already-normalized, whitespace-separated words (its internal
        splitter is also whitespace, so this composes correctly).
        """
        encoded_corpus: list[str] = []
        for doc in corpus:
            words = self._prepare_words(doc)
            encoded_corpus.append(" ".join(words))
        self._cpp.train(encoded_corpus, vocab_size)
        self._vocab = dict(self._cpp.get_vocab())
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        self._refresh_special_ids()

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        _save_vocab_txt(self._vocab, os.path.join(directory, "vocab.txt"))
        super().save(directory)

    def _save_extras(self) -> dict[str, object]:
        return {
            "model": {
                "type": "WordPiece",
                "vocab": self._vocab,
                "unk_token": self._unk_token,
                "continuing_prefix": self._continuing_prefix,
                "max_chars_per_word": self._max_chars_per_word,
            }
        }

    @classmethod
    def from_file(
        cls,
        directory: str,
        *,
        unk_token: str = "[UNK]",
        continuing_prefix: str = "##",
        max_chars_per_word: int = 100,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> "WordPieceTokenizerFast":
        vocab_path = os.path.join(directory, "vocab.txt")
        if not os.path.isfile(vocab_path):
            raise FileNotFoundError(
                f"WordPieceTokenizerFast.from_file: vocab.txt not found "
                f"in {directory}"
            )
        vocab = _load_vocab_txt(vocab_path)
        st = special_tokens or _load_special_tokens_map(directory)
        return cls(
            vocab,
            unk_token=unk_token,
            continuing_prefix=continuing_prefix,
            max_chars_per_word=max_chars_per_word,
            normalizer=normalizer,
            pre_tokenizer=pre_tokenizer,
            special_tokens=st,
        )

    from_pretrained = from_file
