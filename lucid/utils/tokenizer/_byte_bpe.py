"""ByteLevelBPETokenizer — Byte-Pair Encoding over byte-mapped text.

The GPT-2 / RoBERTa / BART tokenizer.  Two key differences from
classical :class:`BPETokenizer`:

1. **Pre-tokenizer** is :class:`ByteLevel` — every UTF-8 byte gets
   re-mapped to a printable Unicode codepoint via the GPT-2
   byte-to-unicode table (bytes 0x00-0x20 + 0x7F-0xA0 get shifted
   into a "safe" Unicode block; everything else maps to itself).
   This means the BPE algorithm never sees raw bytes — it operates
   over a printable Unicode alphabet of size 256.  Bonus: no UNK
   needed, because every byte sequence can be encoded.

2. **Decode** must reverse the byte-mapping.  Joining token surface
   forms gives back the byte-mapped string; passing that through
   :meth:`ByteLevel.decode_bytes` recovers the original UTF-8 bytes,
   which we then UTF-8-decode back to ``str``.  Without this step
   the decoded output looks like ``"Ġworld"`` instead of ``" world"``.

Inherits from :class:`BPETokenizer` / :class:`BPETokenizerFast` —
the merge table + training loop are exactly the same.  Per the user
request, no new C++ class — just a subclass that swaps in the right
pre-tokenizer + post-decode.
"""

import os
from typing import Iterable

from lucid.utils.tokenizer._base import SpecialTokens
from lucid.utils.tokenizer._bpe import (
    BPETokenizer,
    BPETokenizerFast,
    _load_merges_txt,
    _load_unified_tokenizer_json,
    _load_vocab_json,
    _special_tokens_from_map,
)
from lucid.utils.tokenizer._normalizers import NFC, Normalizer
from lucid.utils.tokenizer._pre_tokenizers import ByteLevel


class _ByteLevelDecodeMixin:
    """Shared decode override for both BBPE flavours.

    Joins token surface forms, runs them through
    :meth:`ByteLevel.decode_bytes` to undo the GPT-2 byte mapping,
    then UTF-8 decodes back to ``str``.
    """

    _id_to_token: dict[int, str]

    def _decode_one(self, ids: list[int]) -> str:
        # 1. Join the byte-mapped tokens.
        text = "".join(self._id_to_token[i] for i in ids if i in self._id_to_token)
        # 2. Reverse the byte-to-unicode mapping → raw UTF-8 bytes.
        try:
            raw = ByteLevel.decode_bytes(text)
        except KeyError:
            # Decoded id mapped to a token that isn't in the GPT-2
            # byte alphabet (most likely a user-added special).  Fall
            # back to the raw joined text (the special's surface
            # form is the right rendering anyway).
            return text
        # 3. UTF-8 decode (errors=replace so we never raise).
        return raw.decode("utf-8", errors="replace")


# ── Pure-Python BBPE ────────────────────────────────────────────────


class ByteLevelBPETokenizer(_ByteLevelDecodeMixin, BPETokenizer):
    r"""Pure-Python byte-level BPE (GPT-2 / RoBERTa convention).

    Parameters
    ----------
    vocab : dict[str, int]
        Token-string → id map.  Tokens are the byte-mapped Unicode
        strings (e.g. ``"Ġworld"`` for `` world``), NOT raw bytes.
    merges : list[tuple[str, str]]
        BPE merges, same format as plain BPE.
    add_prefix_space : bool, default False
        Whether to prepend a space to the input before pre-tokenizing
        (so the first word also gets a ``Ġ`` prefix).  GPT-2 default
        is ``False``; RoBERTa default is ``True``.
    normalizer : Normalizer, optional
        Default :class:`NFC` (matches GPT-2).
    special_tokens : SpecialTokens, optional
    """

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        *,
        add_prefix_space: bool = False,
        normalizer: Normalizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        self._add_prefix_space = add_prefix_space
        super().__init__(
            vocab,
            merges,
            normalizer=normalizer if normalizer is not None else NFC(),
            pre_tokenizer=ByteLevel(add_prefix_space=add_prefix_space),
            special_tokens=special_tokens,
        )

    @property
    def algo(self) -> str:
        return "byte_bpe"

    def _save_extras(self) -> dict[str, object]:
        return {
            "model": {
                "type": "ByteLevelBPE",
                "vocab": self._vocab,
                "merges": [list(m) for m in self._merges],
                "add_prefix_space": self._add_prefix_space,
            }
        }

    @classmethod
    def from_file(
        cls,
        directory: str,
        *,
        add_prefix_space: bool = False,
        normalizer: Normalizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> "ByteLevelBPETokenizer":
        """Load from HF-compatible directory.

        Same on-disk format as :class:`BPETokenizer`: either
        ``vocab.json`` + ``merges.txt`` legacy pair OR unified
        ``tokenizer.json``.
        """
        import json

        unified = os.path.join(directory, "tokenizer.json")
        if os.path.isfile(unified):
            vocab, merges = _load_unified_tokenizer_json(unified)
            # Check the unified file for add_prefix_space override.
            with open(unified, encoding="utf-8") as f:
                data = json.load(f)
            model = data.get("model", {})
            add_prefix_space = bool(model.get("add_prefix_space", add_prefix_space))
        else:
            vj = os.path.join(directory, "vocab.json")
            mt = os.path.join(directory, "merges.txt")
            if not (os.path.isfile(vj) and os.path.isfile(mt)):
                raise FileNotFoundError(
                    f"ByteLevelBPETokenizer.from_file: neither "
                    f"tokenizer.json nor (vocab.json + merges.txt) found "
                    f"in {directory}"
                )
            vocab = _load_vocab_json(vj)
            merges = _load_merges_txt(mt)
        st = special_tokens
        if st is None:
            sp_path = os.path.join(directory, "special_tokens_map.json")
            if os.path.isfile(sp_path):
                with open(sp_path, encoding="utf-8") as f:
                    sp = json.load(f)
                st = _special_tokens_from_map(sp)
        return cls(
            vocab,
            merges,
            add_prefix_space=add_prefix_space,
            normalizer=normalizer,
            special_tokens=st,
        )

    from_pretrained = from_file

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int = 50_000,
    ) -> None:
        """Train via the parent :meth:`BPETokenizer.train`.

        The configured :class:`ByteLevel` pre-tokenizer means the
        algorithm sees byte-mapped chunks throughout training, so
        the resulting vocab + merges are in GPT-2 byte-mapped form
        (compatible with HF GPT-2 / RoBERTa export).
        """
        super().train(corpus, vocab_size=vocab_size)


# ── Fast (C++-backed) BBPE ──────────────────────────────────────────


class ByteLevelBPETokenizerFast(_ByteLevelDecodeMixin, BPETokenizerFast):
    """C++-backed BBPE.  Identical configuration semantics to
    :class:`ByteLevelBPETokenizer`; uses
    :class:`lucid._C.engine.utils.tokenizer.BPE` for the merge hot
    loop and the same ByteLevel pre-tokenizer in Python."""

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        *,
        add_prefix_space: bool = False,
        normalizer: Normalizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        self._add_prefix_space = add_prefix_space
        super().__init__(
            vocab,
            merges,
            normalizer=normalizer if normalizer is not None else NFC(),
            pre_tokenizer=ByteLevel(add_prefix_space=add_prefix_space),
            special_tokens=special_tokens,
        )

    @property
    def algo(self) -> str:
        return "byte_bpe"

    def _save_extras(self) -> dict[str, object]:
        return {
            "model": {
                "type": "ByteLevelBPE",
                "vocab": self._vocab,
                "merges": [list(m) for m in self._merges],
                "add_prefix_space": self._add_prefix_space,
            }
        }

    @classmethod
    def from_file(
        cls,
        directory: str,
        *,
        add_prefix_space: bool = False,
        normalizer: Normalizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> "ByteLevelBPETokenizerFast":
        import json

        unified = os.path.join(directory, "tokenizer.json")
        if os.path.isfile(unified):
            vocab, merges = _load_unified_tokenizer_json(unified)
            with open(unified, encoding="utf-8") as f:
                data = json.load(f)
            model = data.get("model", {})
            add_prefix_space = bool(model.get("add_prefix_space", add_prefix_space))
        else:
            vj = os.path.join(directory, "vocab.json")
            mt = os.path.join(directory, "merges.txt")
            if not (os.path.isfile(vj) and os.path.isfile(mt)):
                raise FileNotFoundError(
                    f"ByteLevelBPETokenizerFast.from_file: neither "
                    f"tokenizer.json nor (vocab.json + merges.txt) found "
                    f"in {directory}"
                )
            vocab = _load_vocab_json(vj)
            merges = _load_merges_txt(mt)
        st = special_tokens
        if st is None:
            sp_path = os.path.join(directory, "special_tokens_map.json")
            if os.path.isfile(sp_path):
                with open(sp_path, encoding="utf-8") as f:
                    sp = json.load(f)
                st = _special_tokens_from_map(sp)
        return cls(
            vocab,
            merges,
            add_prefix_space=add_prefix_space,
            normalizer=normalizer,
            special_tokens=st,
        )

    from_pretrained = from_file

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int = 50_000,
    ) -> None:
        """Train via the parent :meth:`BPETokenizerFast.train` (which
        delegates to the C++ BPE algorithm).

        The ByteLevel pre-tokenizer normalizes every byte to its
        printable form before the corpus reaches the C++ training
        loop, so the resulting vocab is in byte-mapped form (HF
        GPT-2 compatible).

        Note: the C++ ``BPE::train`` does its OWN whitespace split
        on the raw input, so we pre-encode the corpus through the
        ByteLevel pre-tokenizer here in Python before handing off
        — otherwise the training would learn merges on raw bytes
        rather than byte-mapped codepoints.
        """
        encoded_corpus: list[str] = []
        for doc in corpus:
            if self._normalizer is not None:
                doc = self._normalizer(doc)
            chunks = [chunk for chunk, _ in self._pre_tokenizer(doc)]
            encoded_corpus.append(" ".join(chunks))
        self._cpp.train(encoded_corpus, vocab_size)
        self._vocab = dict(self._cpp.get_vocab())
        self._merges = [(a, b) for a, b in self._cpp.merges()]
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        self._refresh_special_ids()
        self._sync_special_tokens_to_cpp()
