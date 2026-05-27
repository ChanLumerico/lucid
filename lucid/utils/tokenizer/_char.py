"""CharTokenizer — one Unicode codepoint = one token.

Vocab is learned from the corpus (every distinct codepoint gets a
fresh id in insertion order).  Sequences are about as long as
:class:`ByteTokenizer`'s for ASCII text but ~4× shorter for
non-ASCII (one id per codepoint rather than per byte).

Trade-off vs :class:`ByteTokenizer`:
* **Char** — needs training; OOV (codepoint not in vocab) → UNK.
* **Byte** — no training; no OOV.

Use Char when you want a closed vocab tied to the training corpus
(language modelling on a fixed alphabet, character RNNs); use Byte
when you need an open-vocab universal fallback.
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


def _split_codepoints(text: str) -> list[str]:
    """Split Python ``str`` into per-codepoint chunks.

    Python's ``str`` is already codepoint-indexed, so this is a
    cheap list comprehension; matches the C++ ``CharTokenizer``'s
    UTF-8 codepoint splitter byte-for-byte.
    """
    return list(text)


class CharTokenizer(Tokenizer):
    """Reference (pure-Python) character-level tokenizer.

    Parameters
    ----------
    vocab : dict[str, int], optional
        Pre-built vocab; if omitted, an empty tokenizer is
        constructed (call :meth:`train` before encoding).
    special_tokens : SpecialTokens, optional
        Special-token registry.  Configuring ``unk`` enables OOV
        fallback (otherwise unknown codepoints are dropped).
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
        return "char"

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def id_to_token(self, token_id: int) -> str | None:
        return self._id_to_token.get(token_id)

    def _encode_one(self, text: str) -> list[int]:
        unk = self.unk_token_id
        out: list[int] = []
        for ch in _split_codepoints(text):
            tid = self._vocab.get(ch)
            if tid is not None:
                out.append(tid)
            elif unk is not None:
                out.append(unk)
        return out

    def _decode_one(self, ids: list[int]) -> str:
        return "".join(self._id_to_token[i] for i in ids if i in self._id_to_token)

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int = 1000,
    ) -> None:
        """Build vocab by collecting unique codepoints in insertion
        order across ``corpus``.

        Stops when ``vocab_size`` distinct codepoints have been
        observed.
        """
        v: dict[str, int] = {}
        next_id = 0
        for doc in corpus:
            for ch in _split_codepoints(doc):
                if ch not in v:
                    v[ch] = next_id
                    next_id += 1
                    if next_id >= vocab_size:
                        break
            if next_id >= vocab_size:
                break
        self._vocab = v
        self._id_to_token = {i: c for c, i in v.items()}
        self._refresh_special_ids()

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        _save_vocab_txt(self._vocab, os.path.join(directory, "vocab.txt"))
        super().save(directory)

    def _save_extras(self) -> dict[str, object]:
        return {"model": {"type": "Char", "vocab": self._vocab}}

    @classmethod
    def from_file(
        cls,
        directory: str,
        *,
        special_tokens: SpecialTokens | None = None,
    ) -> "CharTokenizer":
        vocab_path = os.path.join(directory, "vocab.txt")
        if not os.path.isfile(vocab_path):
            raise FileNotFoundError(
                f"CharTokenizer.from_file: vocab.txt not found in {directory}"
            )
        vocab = _load_vocab_txt(vocab_path)
        st = special_tokens or _load_special_tokens_map(directory)
        return cls(vocab=vocab, special_tokens=st)

    from_pretrained = from_file


class CharTokenizerFast(Tokenizer):
    """C++-backed character tokenizer.  See :class:`CharTokenizer`."""

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        *,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        self._vocab: dict[str, int] = dict(vocab) if vocab else {}
        if self._vocab:
            self._cpp = _C_engine.utils.tokenizer.CharTokenizer(self._vocab)
        else:
            self._cpp = _C_engine.utils.tokenizer.CharTokenizer()
        self._id_to_token: dict[int, str] = {v: k for k, v in self._vocab.items()}
        super().__init__(special_tokens=special_tokens)

    @property
    def vocab_size(self) -> int:
        return self._cpp.vocab_size()

    @property
    def algo(self) -> str:
        return "char"

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
        vocab_size: int = 1000,
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
        return {"model": {"type": "Char", "vocab": self._vocab}}

    @classmethod
    def from_file(
        cls,
        directory: str,
        *,
        special_tokens: SpecialTokens | None = None,
    ) -> "CharTokenizerFast":
        vocab_path = os.path.join(directory, "vocab.txt")
        if not os.path.isfile(vocab_path):
            raise FileNotFoundError(
                f"CharTokenizerFast.from_file: vocab.txt not found in " f"{directory}"
            )
        vocab = _load_vocab_txt(vocab_path)
        st = special_tokens or _load_special_tokens_map(directory)
        return cls(vocab=vocab, special_tokens=st)

    from_pretrained = from_file
