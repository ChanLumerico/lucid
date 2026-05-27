"""WordTokenizer — whitespace split + explicit UNK on OOV.

Same chunking rule as :class:`WhitespaceTokenizer`, but emits the
configured ``unk`` special token (instead of silently dropping)
when a word is missing from the vocab.  Requires
``special_tokens.unk`` to be set; an empty UNK config + OOV word
raises :class:`ValueError` at encode time.

Use when you want an explicit OOV signal in the id sequence
(common for classical NLP / language modelling baselines).
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
    """Reference (pure-Python) word-level tokenizer with UNK fallback.

    Parameters
    ----------
    vocab : dict[str, int], optional
        Pre-built vocab.
    special_tokens : SpecialTokens, optional
        Must include a non-None ``unk`` (the tokenizer raises
        :class:`ValueError` on encode if OOV is hit without one).
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
        return len(self._vocab)

    @property
    def algo(self) -> str:
        return "word"

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def id_to_token(self, token_id: int) -> str | None:
        return self._id_to_token.get(token_id)

    def _encode_one(self, text: str) -> list[int]:
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
        return " ".join(self._id_to_token[i] for i in ids if i in self._id_to_token)

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int = 30_000,
    ) -> None:
        """Same as :meth:`WhitespaceTokenizer.train` — collect unique
        words in insertion order until ``vocab_size`` is reached.

        Training does NOT pre-seed an UNK token; the caller is
        responsible for adding it to the vocab + the special-token
        registry before encoding OOV text.
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
        os.makedirs(directory, exist_ok=True)
        _save_vocab_txt(self._vocab, os.path.join(directory, "vocab.txt"))
        super().save(directory)

    def _save_extras(self) -> dict[str, object]:
        return {"model": {"type": "Word", "vocab": self._vocab}}

    @classmethod
    def from_file(
        cls,
        directory: str,
        *,
        special_tokens: SpecialTokens | None = None,
    ) -> "WordTokenizer":
        vocab_path = os.path.join(directory, "vocab.txt")
        if not os.path.isfile(vocab_path):
            raise FileNotFoundError(
                f"WordTokenizer.from_file: vocab.txt not found in {directory}"
            )
        vocab = _load_vocab_txt(vocab_path)
        st = special_tokens or _load_special_tokens_map(directory)
        return cls(vocab=vocab, special_tokens=st)

    from_pretrained = from_file


class WordTokenizerFast(Tokenizer):
    """C++-backed word tokenizer.  See :class:`WordTokenizer`."""

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
        """Mirror the unk id (and others) into the C++ tokenizer's
        ``SpecialTokens`` registry — the C++ ``LookupTokenizer.encode``
        consults ``special_.unk`` for OOV fallback."""
        st = _C_engine.utils.tokenizer.SpecialTokens()
        for name in ("pad", "unk", "bos", "eos", "mask", "sep", "cls"):
            tid = self._special_ids.get(name)
            if tid is not None:
                setattr(st, name, tid)
        self._cpp.special_tokens = st

    @property
    def vocab_size(self) -> int:
        return self._cpp.vocab_size()

    @property
    def algo(self) -> str:
        return "word"

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def id_to_token(self, token_id: int) -> str | None:
        return self._id_to_token.get(token_id)

    def _encode_one(self, text: str) -> list[int]:
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
        self._sync_specials_to_cpp()

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        _save_vocab_txt(self._vocab, os.path.join(directory, "vocab.txt"))
        super().save(directory)

    def _save_extras(self) -> dict[str, object]:
        return {"model": {"type": "Word", "vocab": self._vocab}}

    @classmethod
    def from_file(
        cls,
        directory: str,
        *,
        special_tokens: SpecialTokens | None = None,
    ) -> "WordTokenizerFast":
        vocab_path = os.path.join(directory, "vocab.txt")
        if not os.path.isfile(vocab_path):
            raise FileNotFoundError(
                f"WordTokenizerFast.from_file: vocab.txt not found in " f"{directory}"
            )
        vocab = _load_vocab_txt(vocab_path)
        st = special_tokens or _load_special_tokens_map(directory)
        return cls(vocab=vocab, special_tokens=st)

    from_pretrained = from_file
