"""RegexTokenizer — split via a user-supplied regular expression.

Every regex match is emitted as a chunk (and looked up in vocab);
non-matching spans are dropped.  The classic recipe ``r"\\w+"``
splits on alpha-numeric runs, treating every punctuation as a
separator — useful for quick exploratory tokenization.

Both flavours compile the pattern once at construction time and
re-use the compiled matcher across encode calls.

Pattern format
--------------
Python ``re`` (Python flavour) / ECMAScript ``std::regex`` (Fast
flavour).  The two engines have minor incompatibilities at the
edges (lookaround quirks, named-group syntax); for portability use
plain character classes + quantifiers that both support.
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
    """Reference (pure-Python) regex tokenizer.

    Parameters
    ----------
    pattern : str
        Regex pattern matched against the input.  Every match's
        text is a chunk; unmatched spans are dropped.
    vocab : dict[str, int], optional
        Pre-built vocab; if omitted, call :meth:`train` first.
    special_tokens : SpecialTokens, optional
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
        return len(self._vocab)

    @property
    def algo(self) -> str:
        return "regex"

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def id_to_token(self, token_id: int) -> str | None:
        return self._id_to_token.get(token_id)

    def _encode_one(self, text: str) -> list[int]:
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
        os.makedirs(directory, exist_ok=True)
        _save_vocab_txt(self._vocab, os.path.join(directory, "vocab.txt"))
        # Also persist the pattern so from_file() can reconstruct.
        with open(
            os.path.join(directory, "regex_pattern.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(self._pattern_str)
        super().save(directory)

    def _save_extras(self) -> dict[str, object]:
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

    from_pretrained = from_file


class RegexTokenizerFast(Tokenizer):
    """C++-backed regex tokenizer.  See :class:`RegexTokenizer`.

    Note: the Fast flavour uses ``std::regex`` (ECMAScript dialect)
    while the Python flavour uses ``re`` — patterns that lean on
    Python-specific features (named-group syntax with ``?P<name>``,
    lookbehind variability, etc.) may diverge.  Stick to plain
    character classes + quantifiers for portable patterns.
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
        st = _C_engine.utils.tokenizer.SpecialTokens()
        for name in ("pad", "unk", "bos", "eos", "mask", "sep", "cls"):
            tid = self._special_ids.get(name)
            if tid is not None:
                setattr(st, name, tid)
        self._cpp.special_tokens = st

    @property
    def pattern(self) -> str:
        return self._pattern_str

    @property
    def vocab_size(self) -> int:
        return self._cpp.vocab_size()

    @property
    def algo(self) -> str:
        return "regex"

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
        self._sync_specials_to_cpp()

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        _save_vocab_txt(self._vocab, os.path.join(directory, "vocab.txt"))
        with open(
            os.path.join(directory, "regex_pattern.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(self._pattern_str)
        super().save(directory)

    def _save_extras(self) -> dict[str, object]:
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
