"""WordTokenizer — whitespace split + explicit UNK on OOV.

This file holds **both flavours** of the word-level tokenizer:

* :class:`WordTokenizer` — pure-Python reference implementation.
  Trivial to read / debug; suitable for any environment without
  the C++ engine available.

* :class:`WordTokenizerFast` — C++-backed via
  :class:`lucid._C.engine.utils.tokenizer.WordTokenizer`.  Same
  whitespace-split + dict-lookup semantics, but the hot loop runs
  in C++.  Use this in production where encode throughput matters.

Both flavours share the chunking rule of
:class:`~lucid.utils.tokenizer.WhitespaceTokenizer` (split on
runs of ASCII whitespace) but differ in OOV handling: instead of
silently dropping out-of-vocab words, they emit the configured
``unk`` special token.  When ``special_tokens.unk`` is ``None``
and an OOV word is hit, encoding raises :class:`ValueError` — use
:class:`~lucid.utils.tokenizer.WhitespaceTokenizer` if you want
silent drop semantics instead.

On-disk format
--------------
Both flavours write a single ``vocab.txt`` (one token per line,
line number = id) plus the shared ``tokenizer.json`` /
``special_tokens_map.json`` files produced by the base class.

Use case
--------
Common for classical NLP / language modelling baselines where an
explicit OOV signal in the id sequence is preferable to silent
information loss.
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
from lucid.utils.tokenizer._whitespace import _whitespace_split


class WordTokenizer(Tokenizer):
    r"""Reference (pure-Python) word-level tokenizer with UNK fallback.

    Splits the input on ASCII whitespace runs, looks each word up
    in the vocab, and either emits the configured ``unk`` token or
    raises :class:`ValueError` on out-of-vocab inputs.

    For production / latency-sensitive use, prefer the matching
    :class:`WordTokenizerFast` — same algorithm, same vocab format,
    same OOV-error semantics, but the hot loop runs in C++.

    Parameters
    ----------
    vocab : dict[str, int], optional
        Pre-built token-string → id map.  If omitted, the
        tokenizer starts empty and :meth:`train` must be called
        before encoding.
    special_tokens : SpecialTokens, optional
        Must include a non-None ``unk`` if the caller plans to
        encode text containing OOV words.  Without it, OOV at
        encode time raises :class:`ValueError`.

    Notes
    -----
    :meth:`train` does **not** pre-seed an UNK token.  The caller
    is responsible for adding it to both the vocab and the
    :class:`SpecialTokens` registry before encoding OOV text.

    Examples
    --------
    >>> tok = WordTokenizer(special_tokens=SpecialTokens(unk="<unk>"))
    >>> tok.train(["the quick brown fox", "<unk>"])
    >>> tok.encode("the slow fox")  # 'slow' → unk
    [0, 4, 3]

    See Also
    --------
    WordTokenizerFast : C++-backed sibling with the same surface API.
    """

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        *,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        self._vocab: dict[str, int] = dict(vocab) if vocab else {}
        self._id_to_token: dict[int, str] = {v: k for k, v in self._vocab.items()}
        super().__init__(special_tokens=special_tokens)

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocab."""
        return len(self._vocab)

    @property
    def algo(self) -> str:
        """Algorithm identifier — always ``"word"``."""
        return "word"

    def get_vocab(self) -> dict[str, int]:
        """Return a copy of the token-string → id map."""
        return dict(self._vocab)

    def id_to_token(self, token_id: int) -> str | None:
        """Reverse lookup; returns ``None`` if ``token_id`` is unknown."""
        return self._id_to_token.get(token_id)

    def _encode_one(self, text: str) -> list[int]:
        """Algorithm-specific encode: whitespace-split + dict lookup."""
        out: list[int] = []
        unk = self.unk_token_id
        for w in _whitespace_split(text):
            tid = self._vocab.get(w)
            if tid is not None:
                out.append(tid)
            elif unk is not None:
                out.append(unk)
            else:
                raise ValueError(
                    f"WordTokenizer.encode: OOV word {w!r} encountered "
                    f"and no UNK token is configured.  Set "
                    f"``special_tokens.unk``."
                )
        return out

    def _decode_one(self, ids: list[int]) -> str:
        """Algorithm-specific decode: join known ids with a single space."""
        return " ".join(self._id_to_token[i] for i in ids if i in self._id_to_token)

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int = 30_000,
    ) -> None:
        """Collect unique words in insertion order until ``vocab_size``.

        Same algorithm as :meth:`WhitespaceTokenizer.train` — pure
        insertion-order vocab build with a hard cap.  No frequency
        cut-off is applied.

        Training does **not** pre-seed an UNK token; the caller is
        responsible for adding it to the vocab + the special-token
        registry before encoding OOV text.

        Parameters
        ----------
        corpus : iterable of str
            Each item is one document.  Generators are consumed
            exactly once.
        vocab_size : int, default 30 000
            Hard cap on the number of distinct tokens kept.
        """
        v: dict[str, int] = {}
        next_id = 0
        for doc in corpus:
            for w in _whitespace_split(doc):
                if w not in v:
                    v[w] = next_id
                    next_id += 1
                    if next_id >= vocab_size:
                        break
            if next_id >= vocab_size:
                break
        self._vocab = v
        self._id_to_token = {i: t for t, i in v.items()}
        self._refresh_special_ids()

    def save(self, directory: str) -> None:
        """Persist vocab + unified ``tokenizer.json``.

        Parameters
        ----------
        directory : str
            Output directory.  Created if it does not exist.  Files
            written: ``vocab.txt``, ``tokenizer.json`` (via the
            base class), and optionally ``special_tokens_map.json``.
        """
        os.makedirs(directory, exist_ok=True)
        _save_vocab_txt(self._vocab, os.path.join(directory, "vocab.txt"))
        super().save(directory)

    def _save_extras(self) -> dict[str, object]:
        """Add ``model`` block (type / vocab) to ``tokenizer.json``."""
        return {"model": {"type": "Word", "vocab": self._vocab}}

    @classmethod
    def from_file(
        cls,
        directory: str,
        *,
        special_tokens: SpecialTokens | None = None,
    ) -> WordTokenizer:
        """Load from a directory previously written by :meth:`save`.

        Parameters
        ----------
        directory : str
            Directory containing ``vocab.txt`` (mandatory) and
            optionally ``special_tokens_map.json``.
        special_tokens : SpecialTokens, optional
            Overrides any on-disk ``special_tokens_map.json``.

        Returns
        -------
        WordTokenizer
            Reconstructed tokenizer.
        """
        vocab_path = os.path.join(directory, "vocab.txt")
        if not os.path.isfile(vocab_path):
            raise FileNotFoundError(
                f"WordTokenizer.from_file: vocab.txt not found in {directory}"
            )
        vocab = _load_vocab_txt(vocab_path)
        st = special_tokens or _load_special_tokens_map(directory)
        return cls(vocab=vocab, special_tokens=st)

    from_pretrained = from_file  # alias matching HF naming


class WordTokenizerFast(Tokenizer):
    r"""C++-backed word tokenizer with UNK fallback.

    Identical surface API and OOV-error semantics to
    :class:`WordTokenizer`; the whitespace scan + dict lookup run
    in C++ via :class:`lucid._C.engine.utils.tokenizer.WordTokenizer`.

    Parameters
    ----------
    vocab : dict[str, int], optional
        Pre-built token-string → id map.  Forwarded to the C++
        backend at construction; if omitted, the C++ tokenizer is
        constructed with an empty vocab and :meth:`train` must be
        called before encoding.
    special_tokens : SpecialTokens, optional
        Must include a non-None ``unk`` to allow OOV input; the id
        is mirrored into the C++ registry so the C++ OOV fall-back
        path agrees with the Python-side OOV check.

    Notes
    -----
    The C++ ``LookupTokenizer.encode`` silently substitutes UNK for
    OOV words when one is configured, or drops them otherwise.  To
    match the Python reference's explicit error for unconfigured
    UNK, :meth:`_encode_one` performs an OOV pre-check before
    delegating.

    See Also
    --------
    WordTokenizer : Pure-Python reference sibling.
    """

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        *,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        self._vocab: dict[str, int] = dict(vocab) if vocab else {}
        if self._vocab:
            self._cpp = _C_engine.utils.tokenizer.WordTokenizer(self._vocab)
        else:
            self._cpp = _C_engine.utils.tokenizer.WordTokenizer()
        self._id_to_token: dict[int, str] = {v: k for k, v in self._vocab.items()}
        super().__init__(special_tokens=special_tokens)
        self._sync_specials_to_cpp()

    def _sync_specials_to_cpp(self) -> None:
        """Mirror :attr:`_special_ids` into the C++ tokenizer registry.

        The C++ ``LookupTokenizer.encode`` consults ``special_.unk``
        for OOV fallback — keeping the two sides in sync is what
        makes the slow/fast OOV behaviour match.
        """
        st = _C_engine.utils.tokenizer.SpecialTokens()
        for name in ("pad", "unk", "bos", "eos", "mask", "sep", "cls"):
            tid = self._special_ids.get(name)
            if tid is not None:
                setattr(st, name, tid)
        self._cpp.special_tokens = st

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocab (queried via C++)."""
        return self._cpp.vocab_size()

    @property
    def algo(self) -> str:
        """Algorithm identifier — always ``"word"``."""
        return "word"

    def get_vocab(self) -> dict[str, int]:
        """Return a copy of the cached token-string → id map."""
        return dict(self._vocab)

    def id_to_token(self, token_id: int) -> str | None:
        """Reverse lookup; returns ``None`` if ``token_id`` is unknown."""
        return self._id_to_token.get(token_id)

    def _encode_one(self, text: str) -> list[int]:
        """C++ encode path with a Python-side OOV pre-check."""
        # The C++ tokenizer falls back to UNK silently or drops the
        # token if UNK is unset.  To match the Python reference's
        # explicit error, check for OOV in Python before delegating.
        if self.unk_token_id is None:
            for w in _whitespace_split(text):
                if w not in self._vocab:
                    raise ValueError(
                        f"WordTokenizerFast.encode: OOV word {w!r} "
                        f"encountered and no UNK token is configured.  "
                        f"Set ``special_tokens.unk``."
                    )
        return list(self._cpp.encode(text))

    def _decode_one(self, ids: list[int]) -> str:
        """C++ decode path — returns the joined surface form."""
        return self._cpp.decode(list(ids))

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int = 30_000,
    ) -> None:
        """Re-train in C++; refresh Python-side caches afterwards.

        Parameters
        ----------
        corpus : iterable of str
            Materialised into a ``list`` before being handed to the
            C++ binding.
        vocab_size : int, default 30 000
            Hard cap on the number of distinct tokens kept.
        """
        self._cpp.train(list(corpus), vocab_size)
        self._vocab = dict(self._cpp.get_vocab())
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        self._refresh_special_ids()
        self._sync_specials_to_cpp()

    def save(self, directory: str) -> None:
        """Same on-disk format as :meth:`WordTokenizer.save`.

        Parameters
        ----------
        directory : str
            Output directory.  Created if it does not exist.
        """
        os.makedirs(directory, exist_ok=True)
        _save_vocab_txt(self._vocab, os.path.join(directory, "vocab.txt"))
        super().save(directory)

    def _save_extras(self) -> dict[str, object]:
        """Add ``model`` block (type / vocab) to ``tokenizer.json``."""
        return {"model": {"type": "Word", "vocab": self._vocab}}

    @classmethod
    def from_file(
        cls,
        directory: str,
        *,
        special_tokens: SpecialTokens | None = None,
    ) -> WordTokenizerFast:
        """Identical loader to :meth:`WordTokenizer.from_file`.

        The only difference is the returned class (and hence the
        encode backend).

        Parameters
        ----------
        directory : str
            Directory containing ``vocab.txt``.
        special_tokens : SpecialTokens, optional
            Overrides any on-disk ``special_tokens_map.json``.

        Returns
        -------
        WordTokenizerFast
            Reconstructed tokenizer.
        """
        vocab_path = os.path.join(directory, "vocab.txt")
        if not os.path.isfile(vocab_path):
            raise FileNotFoundError(
                f"WordTokenizerFast.from_file: vocab.txt not found in " f"{directory}"
            )
        vocab = _load_vocab_txt(vocab_path)
        st = special_tokens or _load_special_tokens_map(directory)
        return cls(vocab=vocab, special_tokens=st)

    from_pretrained = from_file
