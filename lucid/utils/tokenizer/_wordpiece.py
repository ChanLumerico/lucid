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
    """Shared algorithm + post-processing for both WordPiece flavours.

    Holds the normalizer + pre-tokenizer chain and the BERT-style
    decode join logic used by both :class:`WordPieceTokenizer` and
    :class:`WordPieceTokenizerFast`.  The encode hot-path lives in
    each subclass (Python loop vs C++ call), but the surrounding
    text preparation + token-to-string rendering are identical.
    """

    _normalizer: Normalizer | None
    _pre_tokenizer: PreTokenizer
    _continuing_prefix: str

    def _prepare_words(self, text: str) -> list[str]:
        """Apply normalizer + pre-tokenizer; return word strings."""
        if self._normalizer is not None:
            text = self._normalizer(text)
        return [chunk for chunk, _ in self._pre_tokenizer(text)]

    def _decode_join(self, ids: list[int], id_to_token: dict[int, str]) -> str:
        """BERT-style decode join (strip ``##`` continuations, insert spaces)."""
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

    Implements BERT's WordPiece algorithm in pure Python.  Each
    pre-tokenized word is split via **greedy longest-match**: the
    longest vocab prefix is emitted, then the algorithm recurses on
    the remainder with the continuation prefix (``"##"`` by default)
    prepended.  If no prefix matches at any position, the whole word
    becomes a single UNK token.

    Easy to read, easy to step through, no C++ build required.  For
    production / latency-sensitive use, prefer
    :class:`WordPieceTokenizerFast` — same algorithm, same vocab
    format, bit-identical encode outputs, but the hot loop runs in
    C++.

    Parameters
    ----------
    vocab : dict[str, int]
        BERT-style vocab — token-string → id map.  Continuation
        tokens are prefixed with ``##``; ``id = line index`` when
        loaded from ``vocab.txt``.
    unk_token : str, default ``"[UNK]"``
        Token emitted when no valid longest-match prefix can be
        found.  Must be present in ``vocab`` for it to actually
        emit an id (otherwise nothing is emitted for that word).
    continuing_prefix : str, default ``"##"``
        Prefix used to distinguish non-initial subwords from initial
        ones.  Almost never overridden outside of WordPiece variants.
    max_chars_per_word : int, default 100
        Words longer than this are short-circuited to UNK without
        running longest-match (matches BERT's ``BasicTokenizer`` cap
        and bounds worst-case encode cost).
    normalizer : Normalizer, optional
        Pre-encode normalisation chain.  Default
        :class:`~lucid.utils.tokenizer._normalizers.BertNormalizer`.
    pre_tokenizer : PreTokenizer, optional
        Chunk-splitter run before WordPiece.  Default
        :class:`~lucid.utils.tokenizer._pre_tokenizers.WhitespacePunctuationSplit`.
    special_tokens : SpecialTokens, optional
        Special-token registry.  Defaults to ``SpecialTokens(unk=unk_token)``
        so the tokenizer behaves correctly out of the box.

    Notes
    -----
    The greedy longest-match is O(W^2) per word in the worst case
    (where W = word length), but typical English words have W < 20,
    so the Python interpreter overhead dominates.
    :class:`WordPieceTokenizerFast` recovers an order of magnitude
    on long-document encoding.

    Examples
    --------
    >>> tok = WordPieceTokenizer.from_pretrained("bert-base-uncased-dir")
    >>> ids = tok.encode("hello world")
    >>> tok.decode(ids)
    'hello world'

    See Also
    --------
    WordPieceTokenizerFast : C++-backed equivalent with the same API.
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
        """Greedy longest-match split of one pre-tokenized word into ids."""
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
        """Re-train this tokenizer from scratch on ``corpus``.

        Greedy frequency-based WordPiece training (matches Hugging
        Face's ``WordPieceTrainer`` semantics; faster than the full
        log-likelihood training in the original paper).  For each
        iteration, counts adjacent symbol-pair frequencies across
        all words and merges the most frequent pair, growing the
        vocab by 1 each step until ``vocab_size`` is reached or no
        pair appears more than once.

        Replaces :attr:`_vocab` in place; previous contents are
        discarded.  The result is a BERT-compatible vocab with
        ``##`` continuation markers and the configured UNK token
        reserved at id 0.  Tie-break is lexicographic key order for
        determinism (same convention as BPE).

        Parameters
        ----------
        corpus : iterable of str
            Each item is one document (or chunk thereof).  Generators
            are consumed exactly once.
        vocab_size : int, default 30 000
            Target total vocab size (chars + merges).  Stops early
            if no more pair-merges occur.
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
        """Persist as a BERT-style directory (``vocab.txt`` +
        unified ``tokenizer.json`` + ``special_tokens_map.json``).

        Writing both legacy and unified formats means the resulting
        directory loads back via :meth:`from_pretrained` regardless
        of which format the caller looks for first.

        Parameters
        ----------
        directory : str
            Output directory; created if missing.
        """
        os.makedirs(directory, exist_ok=True)
        _save_vocab_txt(self._vocab, os.path.join(directory, "vocab.txt"))
        super().save(directory)

    def _save_extras(self) -> dict[str, object]:
        """Emit the WordPiece ``model`` block for ``tokenizer.json``."""
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
    ) -> WordPieceTokenizer:
        """Load from a BERT-style directory containing ``vocab.txt``.

        Parameters
        ----------
        directory : str
            Directory containing ``vocab.txt`` and optionally
            ``special_tokens_map.json``.
        unk_token, continuing_prefix, max_chars_per_word
            See class docstring.  Defaults match BERT.
        normalizer, pre_tokenizer, special_tokens
            See class docstring.  ``special_tokens`` falls back to
            the on-disk ``special_tokens_map.json`` when omitted.

        Returns
        -------
        WordPieceTokenizer
            A new tokenizer populated from disk.

        Raises
        ------
        FileNotFoundError
            If ``vocab.txt`` is missing.
        """
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
    r"""C++-backed WordPiece tokenizer.

    Identical algorithm + vocab format to :class:`WordPieceTokenizer`;
    the hot loops (greedy longest-match encode + training) run in
    C++ via :class:`lucid._C.engine.utils.tokenizer.WordPiece`.
    Encode outputs are bit-identical for the same vocab + same
    normalizer + same pre-tokenizer.

    The Python side still handles normalisation + pre-tokenization
    (so user-defined Python normalizers compose with the C++
    backend) and the BERT-style decode join.

    Parameters
    ----------
    Same as :class:`WordPieceTokenizer`.  The C++ backend is
    constructed transparently in ``__init__`` and held as
    :attr:`_cpp`.

    See Also
    --------
    WordPieceTokenizer : Pure-Python reference; same vocab format.
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
        """Re-train in C++ via :meth:`WordPiece.train`.

        Pre-encodes the corpus through the configured normalizer +
        pre-tokenizer in Python so the C++ training loop sees
        already-normalized, whitespace-separated words (its internal
        splitter is also whitespace, so this composes correctly).
        After training, the Python-side vocab cache is refreshed
        from the C++ side so subsequent encodes see the new state.

        Parameters
        ----------
        corpus : iterable of str
            Each item is one document.  Materialised to a list
            before the C++ binding call.
        vocab_size : int, default 30 000
            Target total vocab size.
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
        """Same format as :meth:`WordPieceTokenizer.save`."""
        os.makedirs(directory, exist_ok=True)
        _save_vocab_txt(self._vocab, os.path.join(directory, "vocab.txt"))
        super().save(directory)

    def _save_extras(self) -> dict[str, object]:
        """Emit the WordPiece ``model`` block for ``tokenizer.json``."""
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
    ) -> WordPieceTokenizerFast:
        """Identical loader to :meth:`WordPieceTokenizer.from_file`;
        the only difference is the returned class (and hence the
        encode backend).

        Parameters
        ----------
        See :meth:`WordPieceTokenizer.from_file`.

        Returns
        -------
        WordPieceTokenizerFast
            A new C++-backed tokenizer populated from disk.

        Raises
        ------
        FileNotFoundError
            If ``vocab.txt`` is missing.
        """
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
