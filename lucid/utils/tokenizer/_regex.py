"""RegexTokenizer — split via a user-supplied regular expression.

This file holds **both flavours** of the regex tokenizer:

* :class:`RegexTokenizer` — pure-Python reference implementation
  backed by the standard library :mod:`re` module.  Useful for
  exploratory tokenization, debugging, and any pattern that leans
  on Python-specific regex features.

* :class:`RegexTokenizerFast` — C++-backed via
  :class:`lucid._C.engine.utils.tokenizer.RegexTokenizer`, using
  ``std::regex`` (ECMAScript dialect).  Use this when encode
  throughput on long documents dominates.

Every regex match is emitted as a chunk and looked up in the
vocab; non-matching spans are dropped.  The classic recipe
``r"\\w+"`` splits on alpha-numeric runs and treats every
punctuation character as a separator — a useful default for
quick-and-dirty tokenization tasks.

On-disk format
--------------
Both flavours share the same two-file layout written by
:meth:`save`:

* ``vocab.txt`` — one token per line, line number = id.
* ``regex_pattern.txt`` — the source regex string (raw, no
  trailing newline normalisation).
* ``tokenizer.json`` — unified single-file format that bundles
  ``model.type = "Regex"`` + ``model.pattern`` + ``model.vocab``.
* ``special_tokens_map.json`` — optional; written by the base
  class.

Pattern portability
-------------------
The two engines have minor incompatibilities at the edges
(lookaround quirks, named-group syntax, character-class shortcuts
beyond ASCII).  For portability across the slow/fast split, stick
to plain character classes + quantifiers that both Python ``re``
and ``std::regex`` ECMAScript dialect support.

Both flavours compile the pattern once at construction time and
re-use the compiled matcher across encode calls.
"""

import os
import re
from typing import Iterable

from lucid._C import engine as _C_engine

from lucid.utils.tokenizer._base import SpecialTokens, Tokenizer
from lucid.utils.tokenizer._lookup_common import (
    _load_special_tokens_map,
    _load_vocab_txt,
    _save_vocab_txt,
)


class RegexTokenizer(Tokenizer):
    r"""Reference (pure-Python) regex tokenizer.

    Splits the input by repeatedly applying a compiled regular
    expression; every match is treated as one chunk and looked up
    in the vocab.  Characters between matches are dropped — there
    is no canonical reconstruction during decode.

    For production / latency-sensitive use, prefer the matching
    :class:`RegexTokenizerFast` — same matching semantics on
    portable patterns, but the hot loop runs in C++.

    Parameters
    ----------
    pattern : str
        Regex pattern matched against the input via
        :func:`re.finditer`.  Every match's text becomes a chunk;
        unmatched spans are dropped.
    vocab : dict[str, int], optional
        Pre-built token-string → id map.  If omitted, the
        tokenizer starts empty and :meth:`train` must be called
        before encoding.
    special_tokens : SpecialTokens, optional
        Special-token registry — see
        :class:`lucid.utils.tokenizer.SpecialTokens`.  If a
        non-None ``unk`` is configured, OOV matches map to it;
        otherwise OOV matches are silently dropped.

    Notes
    -----
    The pattern is compiled once at construction time.  Re-assigning
    the pattern after construction is not supported — build a new
    instance instead.

    Examples
    --------
    >>> tok = RegexTokenizer(r"\w+")
    >>> tok.train(["hello world hello"])
    >>> tok.encode("hello world")
    [0, 1]

    See Also
    --------
    RegexTokenizerFast : C++-backed sibling with the same surface API.
    """

    def __init__(
        self,
        pattern: str,
        vocab: dict[str, int] | None = None,
        *,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        self._pattern_str = pattern
        self._pattern = re.compile(pattern)
        self._vocab: dict[str, int] = dict(vocab) if vocab else {}
        self._id_to_token: dict[int, str] = {v: k for k, v in self._vocab.items()}
        super().__init__(special_tokens=special_tokens)

    @property
    def pattern(self) -> str:
        """The original regex source — useful for serialisation."""
        return self._pattern_str

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocab."""
        return len(self._vocab)

    @property
    def algo(self) -> str:
        """Algorithm identifier — always ``"regex"``."""
        return "regex"

    def get_vocab(self) -> dict[str, int]:
        """Return a copy of the token-string → id map."""
        return dict(self._vocab)

    def id_to_token(self, token_id: int) -> str | None:
        """Reverse lookup; returns ``None`` if ``token_id`` is unknown."""
        return self._id_to_token.get(token_id)

    def _encode_one(self, text: str) -> list[int]:
        """Algorithm-specific encode: regex-match chunks → ids."""
        unk = self.unk_token_id
        out: list[int] = []
        for m in self._pattern.finditer(text):
            tok = m.group(0)
            tid = self._vocab.get(tok)
            if tid is not None:
                out.append(tid)
            elif unk is not None:
                out.append(unk)
        return out

    def _decode_one(self, ids: list[int]) -> str:
        """Algorithm-specific decode: join known ids with a single space."""
        # Regex tokenizer has no canonical reconstruction — the
        # unmatched delimiter chars were dropped.  Join with single
        # space (matches the C++ side's decode).
        return " ".join(self._id_to_token[i] for i in ids if i in self._id_to_token)

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int = 30_000,
    ) -> None:
        """Build the vocab in insertion order from corpus matches.

        Iterates over ``corpus``, applies the regex to each document,
        and assigns each *first-seen* match a fresh id starting at 0.
        Stops as soon as ``vocab_size`` distinct tokens have been
        collected (no frequency cut-off — the order is purely
        insertion order).

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
            for m in self._pattern.finditer(doc):
                tok = m.group(0)
                if tok not in v:
                    v[tok] = next_id
                    next_id += 1
                    if next_id >= vocab_size:
                        break
            if next_id >= vocab_size:
                break
        self._vocab = v
        self._id_to_token = {i: t for t, i in v.items()}
        self._refresh_special_ids()

    def save(self, directory: str) -> None:
        """Persist vocab + regex pattern + unified ``tokenizer.json``.

        Parameters
        ----------
        directory : str
            Output directory.  Created if it does not exist.  Files
            written: ``vocab.txt``, ``regex_pattern.txt``,
            ``tokenizer.json`` (via the base class) and optionally
            ``special_tokens_map.json``.
        """
        os.makedirs(directory, exist_ok=True)
        _save_vocab_txt(self._vocab, os.path.join(directory, "vocab.txt"))
        # Also persist the pattern so from_file() can reconstruct.
        with open(
            os.path.join(directory, "regex_pattern.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(self._pattern_str)
        super().save(directory)

    def _save_extras(self) -> dict[str, object]:
        """Add ``model`` block (type / pattern / vocab) to ``tokenizer.json``."""
        return {
            "model": {
                "type": "Regex",
                "pattern": self._pattern_str,
                "vocab": self._vocab,
            }
        }

    @classmethod
    def from_file(
        cls,
        directory: str,
        *,
        special_tokens: SpecialTokens | None = None,
    ) -> "RegexTokenizer":
        """Load from a directory previously written by :meth:`save`.

        Parameters
        ----------
        directory : str
            Directory containing at minimum ``regex_pattern.txt``.
            ``vocab.txt`` is optional (an empty vocab is used if
            absent — call :meth:`train` afterwards).
            ``special_tokens_map.json`` is consulted if
            ``special_tokens`` is not provided.
        special_tokens : SpecialTokens, optional
            Overrides any on-disk ``special_tokens_map.json``.

        Returns
        -------
        RegexTokenizer
            Reconstructed tokenizer.
        """
        pat_path = os.path.join(directory, "regex_pattern.txt")
        if not os.path.isfile(pat_path):
            raise FileNotFoundError(
                f"RegexTokenizer.from_file: regex_pattern.txt not found "
                f"in {directory}"
            )
        with open(pat_path, encoding="utf-8") as f:
            pattern = f.read()
        vocab_path = os.path.join(directory, "vocab.txt")
        vocab = _load_vocab_txt(vocab_path) if os.path.isfile(vocab_path) else {}
        st = special_tokens or _load_special_tokens_map(directory)
        return cls(pattern=pattern, vocab=vocab, special_tokens=st)

    from_pretrained = from_file  # alias matching HF naming


class RegexTokenizerFast(Tokenizer):
    r"""C++-backed regex tokenizer.

    Identical surface API to :class:`RegexTokenizer`; the per-chunk
    regex scan runs in C++ via
    :class:`lucid._C.engine.utils.tokenizer.RegexTokenizer`.  Use
    this in production training / inference where encode throughput
    matters.

    Parameters
    ----------
    pattern : str
        Regex pattern.  Interpreted as ECMAScript (``std::regex``
        default), **not** the Python ``re`` dialect — see Notes.
    vocab : dict[str, int], optional
        Pre-built token-string → id map.
    special_tokens : SpecialTokens, optional
        See :class:`lucid.utils.tokenizer.SpecialTokens`.  The ``unk``
        id is mirrored into the C++ side so the OOV fall-back path
        is consistent across both flavours.

    Notes
    -----
    The Fast flavour uses ``std::regex`` (ECMAScript dialect) while
    the Python flavour uses :mod:`re` — patterns that lean on
    Python-specific features (named-group syntax with ``?P<name>``,
    lookbehind variability, etc.) may diverge between the two.
    Stick to plain character classes + quantifiers for portable
    patterns.

    See Also
    --------
    RegexTokenizer : Pure-Python reference sibling.
    """

    def __init__(
        self,
        pattern: str,
        vocab: dict[str, int] | None = None,
        *,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        self._pattern_str = pattern
        self._vocab: dict[str, int] = dict(vocab) if vocab else {}
        self._cpp = _C_engine.utils.tokenizer.RegexTokenizer(pattern, self._vocab)
        self._id_to_token: dict[int, str] = {v: k for k, v in self._vocab.items()}
        super().__init__(special_tokens=special_tokens)
        self._sync_specials_to_cpp()

    def _sync_specials_to_cpp(self) -> None:
        """Mirror :attr:`_special_ids` into the C++ tokenizer registry."""
        st = _C_engine.utils.tokenizer.SpecialTokens()
        for name in ("pad", "unk", "bos", "eos", "mask", "sep", "cls"):
            tid = self._special_ids.get(name)
            if tid is not None:
                setattr(st, name, tid)
        self._cpp.special_tokens = st

    @property
    def pattern(self) -> str:
        """The original regex source — useful for serialisation."""
        return self._pattern_str

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocab (queried via C++)."""
        return self._cpp.vocab_size()

    @property
    def algo(self) -> str:
        """Algorithm identifier — always ``"regex"``."""
        return "regex"

    def get_vocab(self) -> dict[str, int]:
        """Return a copy of the cached token-string → id map."""
        return dict(self._vocab)

    def id_to_token(self, token_id: int) -> str | None:
        """Reverse lookup; returns ``None`` if ``token_id`` is unknown."""
        return self._id_to_token.get(token_id)

    def _encode_one(self, text: str) -> list[int]:
        """C++ encode path — single binding call per text."""
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
        """Same on-disk format as :meth:`RegexTokenizer.save`.

        Parameters
        ----------
        directory : str
            Output directory.  Created if it does not exist.
        """
        os.makedirs(directory, exist_ok=True)
        _save_vocab_txt(self._vocab, os.path.join(directory, "vocab.txt"))
        with open(
            os.path.join(directory, "regex_pattern.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(self._pattern_str)
        super().save(directory)

    def _save_extras(self) -> dict[str, object]:
        """Add ``model`` block (type / pattern / vocab) to ``tokenizer.json``."""
        return {
            "model": {
                "type": "Regex",
                "pattern": self._pattern_str,
                "vocab": self._vocab,
            }
        }

    @classmethod
    def from_file(
        cls,
        directory: str,
        *,
        special_tokens: SpecialTokens | None = None,
    ) -> "RegexTokenizerFast":
        """Identical loader to :meth:`RegexTokenizer.from_file`.

        The only difference is the returned class (and hence the
        encode backend).

        Parameters
        ----------
        directory : str
            Directory containing at minimum ``regex_pattern.txt``.
        special_tokens : SpecialTokens, optional
            Overrides any on-disk ``special_tokens_map.json``.

        Returns
        -------
        RegexTokenizerFast
            Reconstructed tokenizer.
        """
        pat_path = os.path.join(directory, "regex_pattern.txt")
        if not os.path.isfile(pat_path):
            raise FileNotFoundError(
                f"RegexTokenizerFast.from_file: regex_pattern.txt not "
                f"found in {directory}"
            )
        with open(pat_path, encoding="utf-8") as f:
            pattern = f.read()
        vocab_path = os.path.join(directory, "vocab.txt")
        vocab = _load_vocab_txt(vocab_path) if os.path.isfile(vocab_path) else {}
        st = special_tokens or _load_special_tokens_map(directory)
        return cls(pattern=pattern, vocab=vocab, special_tokens=st)

    from_pretrained = from_file
