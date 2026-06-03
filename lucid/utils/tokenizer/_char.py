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
        r"""Construct a pure-Python character tokenizer.

        Parameters
        ----------
        vocab : dict[str, int] or None, optional
            Pre-built codepoint → id map; ``None`` or empty means
            empty tokenizer (call :meth:`train` to populate).
        special_tokens : SpecialTokens or None, optional, keyword-only
            Special-token registry; configure ``unk`` for OOV
            fallback during encode.
        """
        self._vocab: dict[str, int] = dict(vocab) if vocab else {}
        self._id_to_token: dict[int, str] = {v: k for k, v in self._vocab.items()}
        super().__init__(special_tokens=special_tokens)

    @property
    def vocab_size(self) -> int:
        """Number of distinct codepoint ids currently registered."""
        return len(self._vocab)

    @property
    def algo(self) -> str:
        """Algorithm identifier (always ``"char"``)."""
        return "char"

    def get_vocab(self) -> dict[str, int]:
        """Return a copy of the codepoint → id map."""
        return dict(self._vocab)

    def id_to_token(self, token_id: int) -> str | None:
        """Return the codepoint string for ``token_id`` or ``None``."""
        return self._id_to_token.get(token_id)

    def _encode_one(self, text: str) -> list[int]:
        """Per-codepoint vocab lookup with optional UNK fallback."""
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
        """Concatenate codepoint surface forms for known ids."""
        return "".join(self._id_to_token[i] for i in ids if i in self._id_to_token)

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int = 1000,
    ) -> None:
        """Build vocab by collecting unique codepoints in insertion order.

        Stops when ``vocab_size`` distinct codepoints have been
        observed.

        Parameters
        ----------
        corpus : iterable of str
            Each item is one document.  Generators are consumed once.
        vocab_size : int, default 1000
            Maximum number of distinct codepoints to retain.
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
        """Persist the vocab as ``vocab.txt`` + unified ``tokenizer.json``.

        Parameters
        ----------
        directory : str
            Output directory (created if missing).
        """
        os.makedirs(directory, exist_ok=True)
        _save_vocab_txt(self._vocab, os.path.join(directory, "vocab.txt"))
        super().save(directory)

    def _save_extras(self) -> dict[str, object]:
        """Emit the unified-format ``model`` block for the char algo."""
        return {"model": {"type": "Char", "vocab": self._vocab}}

    @classmethod
    def from_file(
        cls,
        directory: str,
        *,
        special_tokens: SpecialTokens | None = None,
    ) -> CharTokenizer:
        """Load from a directory containing ``vocab.txt``.

        Parameters
        ----------
        directory : str
            Directory previously written by :meth:`save`.
        special_tokens : SpecialTokens, optional
            Override for the on-disk special tokens.

        Returns
        -------
        CharTokenizer
            New instance configured with the loaded vocab + specials.
        """
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
    """C++-backed character tokenizer; bit-identical to :class:`CharTokenizer`.

    Same vocab format + semantics as :class:`CharTokenizer`; the
    per-codepoint encode loop runs in C++ via
    :class:`lucid._C.engine.utils.tokenizer.CharTokenizer`.

    Parameters
    ----------
    vocab : dict[str, int], optional
        Pre-built codepoint → id map.  If omitted the tokenizer is
        constructed empty (call :meth:`train` before encoding).
    special_tokens : SpecialTokens, optional
        Special-token registry; configure ``unk`` for OOV fallback.
    """

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        *,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        r"""Construct a C++-backed character tokenizer.

        Parameters
        ----------
        vocab : dict[str, int] or None, optional
            Pre-built codepoint → id map.  ``None`` or empty
            constructs the C++ backend with an empty vocab; call
            :meth:`train` to populate.
        special_tokens : SpecialTokens or None, optional, keyword-only
            Special-token registry.

        Notes
        -----
        Caches the Python-side vocab so :meth:`get_vocab` and
        :meth:`id_to_token` avoid a binding round-trip; both halves
        are kept in sync by :meth:`train`.
        """
        self._vocab: dict[str, int] = dict(vocab) if vocab else {}
        if self._vocab:
            self._cpp = _C_engine.utils.tokenizer.CharTokenizer(self._vocab)
        else:
            self._cpp = _C_engine.utils.tokenizer.CharTokenizer()
        self._id_to_token: dict[int, str] = {v: k for k, v in self._vocab.items()}
        super().__init__(special_tokens=special_tokens)

    @property
    def vocab_size(self) -> int:
        """Number of distinct codepoint ids (queried from C++)."""
        return self._cpp.vocab_size()

    @property
    def algo(self) -> str:
        """Algorithm identifier (always ``"char"``)."""
        return "char"

    def get_vocab(self) -> dict[str, int]:
        """Return a copy of the codepoint → id map."""
        return dict(self._vocab)

    def id_to_token(self, token_id: int) -> str | None:
        """Return the codepoint string for ``token_id`` or ``None``."""
        return self._id_to_token.get(token_id)

    def _encode_one(self, text: str) -> list[int]:
        """Delegate to the C++ char tokenizer for the codepoint loop."""
        return list(self._cpp.encode(text))

    def _decode_one(self, ids: list[int]) -> str:
        """Delegate to the C++ char tokenizer for surface concatenation."""
        return self._cpp.decode(list(ids))

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int = 1000,
    ) -> None:
        """Train in C++; mirror the resulting vocab back into Python state.

        Parameters
        ----------
        corpus : iterable of str
            Materialised into a list before hand-off to C++.
        vocab_size : int, default 1000
            Maximum number of distinct codepoints to retain.
        """
        self._cpp.train(list(corpus), vocab_size)
        self._vocab = dict(self._cpp.get_vocab())
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        self._refresh_special_ids()

    def save(self, directory: str) -> None:
        """Persist as ``vocab.txt`` + unified ``tokenizer.json``."""
        os.makedirs(directory, exist_ok=True)
        _save_vocab_txt(self._vocab, os.path.join(directory, "vocab.txt"))
        super().save(directory)

    def _save_extras(self) -> dict[str, object]:
        """Emit the unified-format ``model`` block for the char algo."""
        return {"model": {"type": "Char", "vocab": self._vocab}}

    @classmethod
    def from_file(
        cls,
        directory: str,
        *,
        special_tokens: SpecialTokens | None = None,
    ) -> CharTokenizerFast:
        """Load from a directory containing ``vocab.txt``.

        Parameters
        ----------
        directory : str
            Directory previously written by :meth:`save`.
        special_tokens : SpecialTokens, optional
            Override for the on-disk special tokens.

        Returns
        -------
        CharTokenizerFast
            New instance configured with the loaded vocab + specials.
        """
        vocab_path = os.path.join(directory, "vocab.txt")
        if not os.path.isfile(vocab_path):
            raise FileNotFoundError(
                f"CharTokenizerFast.from_file: vocab.txt not found in " f"{directory}"
            )
        vocab = _load_vocab_txt(vocab_path)
        st = special_tokens or _load_special_tokens_map(directory)
        return cls(vocab=vocab, special_tokens=st)

    from_pretrained = from_file
