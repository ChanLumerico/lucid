"""WhitespaceTokenizer — one token per whitespace-delimited word.

The simplest "word-level" tokenizer.  Vocab is the set of distinct
whitespace-separated tokens encountered during training.  OOV words
are silently dropped (use :class:`WordTokenizer` instead if you
want explicit UNK fallback).

Decode joins ids with a single space — original whitespace
(multiple spaces, tabs, newlines) is not preserved.

Useful as a baseline / debugging tool; production text models almost
always use a subword tokenizer (BPE / WordPiece / Unigram) for OOV
robustness + better compression.
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


def _whitespace_split(text: str) -> list[str]:
    """Built-in Python ``str.split()`` — splits on Unicode whitespace
    and drops empty chunks.  Matches the C++ side's behaviour
    (which uses the same ASCII whitespace set; Unicode whitespace is
    treated as opaque chars by the C++ but in practice ``str.split()``
    is the same for the corpora these tokenizers are used on)."""
    return text.split()


class WhitespaceTokenizer(Tokenizer):
    """Reference (pure-Python) whitespace-split tokenizer.

    Parameters
    ----------
    vocab : dict[str, int], optional
        Pre-built vocab.  Empty = empty tokenizer; call :meth:`train`
        first.
    special_tokens : SpecialTokens, optional
        Special-token registry.
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
        """Number of distinct whitespace-split tokens currently registered."""
        return len(self._vocab)

    @property
    def algo(self) -> str:
        """Algorithm identifier (always ``"whitespace"``)."""
        return "whitespace"

    def get_vocab(self) -> dict[str, int]:
        """Return a copy of the word → id map."""
        return dict(self._vocab)

    def id_to_token(self, token_id: int) -> str | None:
        """Return the word for ``token_id`` or ``None`` if unknown."""
        return self._id_to_token.get(token_id)

    def _encode_one(self, text: str) -> list[int]:
        """Whitespace-split + vocab lookup; OOV words are silently dropped."""
        out: list[int] = []
        for w in _whitespace_split(text):
            tid = self._vocab.get(w)
            if tid is not None:
                out.append(tid)
            # OOV silently dropped — that's the WhitespaceTokenizer
            # contract.  Use WordTokenizer for explicit UNK fallback.
        return out

    def _decode_one(self, ids: list[int]) -> str:
        """Join known token surface forms with single spaces."""
        return " ".join(self._id_to_token[i] for i in ids if i in self._id_to_token)

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int = 30_000,
    ) -> None:
        """Collect unique whitespace-split words in insertion order.

        Parameters
        ----------
        corpus : iterable of str
            Each item is one document.  Generators are consumed once.
        vocab_size : int, default 30 000
            Maximum number of distinct words to retain; training stops
            as soon as the cap is hit.
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
        """Persist as ``vocab.txt`` + unified ``tokenizer.json``.

        Parameters
        ----------
        directory : str
            Output directory (created if missing).
        """
        os.makedirs(directory, exist_ok=True)
        _save_vocab_txt(self._vocab, os.path.join(directory, "vocab.txt"))
        super().save(directory)

    def _save_extras(self) -> dict[str, object]:
        """Emit the unified-format ``model`` block for the whitespace algo."""
        return {"model": {"type": "Whitespace", "vocab": self._vocab}}

    @classmethod
    def from_file(
        cls,
        directory: str,
        *,
        special_tokens: SpecialTokens | None = None,
    ) -> WhitespaceTokenizer:
        """Load from a directory containing ``vocab.txt``.

        Parameters
        ----------
        directory : str
            Directory previously written by :meth:`save`.
        special_tokens : SpecialTokens, optional
            Override for the on-disk special-token map.

        Returns
        -------
        WhitespaceTokenizer
        """
        vocab_path = os.path.join(directory, "vocab.txt")
        if not os.path.isfile(vocab_path):
            raise FileNotFoundError(
                f"WhitespaceTokenizer.from_file: vocab.txt not found in " f"{directory}"
            )
        vocab = _load_vocab_txt(vocab_path)
        st = special_tokens or _load_special_tokens_map(directory)
        return cls(vocab=vocab, special_tokens=st)

    from_pretrained = from_file


class WhitespaceTokenizerFast(Tokenizer):
    """C++-backed whitespace tokenizer; bit-identical to
    :class:`WhitespaceTokenizer`.

    Same vocab format + OOV-dropping contract as
    :class:`WhitespaceTokenizer`; the encode loop runs in C++ via
    :class:`lucid._C.engine.utils.tokenizer.WhitespaceTokenizer`.

    Parameters
    ----------
    vocab : dict[str, int], optional
        Pre-built word → id map.  Empty = call :meth:`train` first.
    special_tokens : SpecialTokens, optional
        Special-token registry.
    """

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        *,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        self._vocab: dict[str, int] = dict(vocab) if vocab else {}
        if self._vocab:
            self._cpp = _C_engine.utils.tokenizer.WhitespaceTokenizer(self._vocab)
        else:
            self._cpp = _C_engine.utils.tokenizer.WhitespaceTokenizer()
        self._id_to_token: dict[int, str] = {v: k for k, v in self._vocab.items()}
        super().__init__(special_tokens=special_tokens)

    @property
    def vocab_size(self) -> int:
        return self._cpp.vocab_size()

    @property
    def algo(self) -> str:
        return "whitespace"

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def id_to_token(self, token_id: int) -> str | None:
        return self._id_to_token.get(token_id)

    def _encode_one(self, text: str) -> list[int]:
        return list(self._cpp.encode(text))

    def _decode_one(self, ids: list[int]) -> str:
        return self._cpp.decode(list(ids))

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int = 30_000,
    ) -> None:
        self._cpp.train(list(corpus), vocab_size)
        self._vocab = dict(self._cpp.get_vocab())
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        self._refresh_special_ids()

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        _save_vocab_txt(self._vocab, os.path.join(directory, "vocab.txt"))
        super().save(directory)

    def _save_extras(self) -> dict[str, object]:
        return {"model": {"type": "Whitespace", "vocab": self._vocab}}

    @classmethod
    def from_file(
        cls,
        directory: str,
        *,
        special_tokens: SpecialTokens | None = None,
    ) -> WhitespaceTokenizerFast:
        vocab_path = os.path.join(directory, "vocab.txt")
        if not os.path.isfile(vocab_path):
            raise FileNotFoundError(
                f"WhitespaceTokenizerFast.from_file: vocab.txt not found "
                f"in {directory}"
            )
        vocab = _load_vocab_txt(vocab_path)
        st = special_tokens or _load_special_tokens_map(directory)
        return cls(vocab=vocab, special_tokens=st)

    from_pretrained = from_file
