"""Byte-Pair Encoding (Sennrich et al. 2016).

This file holds **both flavours** of the BPE tokenizer:

* :class:`BPETokenizer` — pure-Python reference implementation.
  Slower (per-token Python loop) but easy to read, easy to debug,
  easy to extend with new normalizers / pre-tokenizers without
  touching C++.  Use this when training a new tokenizer or
  experimenting with custom configs.

* :class:`BPETokenizerFast` — C++-backed via
  ``lucid._C.engine.utils.tokenizer.BPE``.  Same vocab / merges
  format, same encode output bit-for-bit (verified by parity
  tests).  Use this in production training / inference where
  encode throughput matters.

Both share the HF-compatible on-disk format:

* ``vocab.json`` — JSON dict ``{token: id}``.
* ``merges.txt`` — one merge per line ``"left right"`` (rank = line #
  with line 0 = highest priority); optional ``#version: 0.2`` header.
* ``tokenizer.json`` — unified single-file format (HF Fast tokenizers
  / Rust ``tokenizers`` crate).  When present, takes precedence over
  the legacy pair.

Vocab files from any published Hugging Face BPE checkpoint (GPT,
RoBERTa, BART, distilGPT, ...) load without modification.
"""

import json
import os
from typing import TYPE_CHECKING, Iterable

from lucid._C import engine as _C_engine

from lucid.utils.tokenizer._base import SpecialTokens, Tokenizer
from lucid.utils.tokenizer._normalizers import NFC, Normalizer
from lucid.utils.tokenizer._pre_tokenizers import (
    PreTokenizer,
    WhitespaceSplit,
)

if TYPE_CHECKING:
    pass


# ── Vocab loader helpers (shared between BPE + BPEFast) ─────────────


def _load_vocab_json(path: str) -> dict[str, int]:
    """Parse a HF-compatible ``vocab.json``.

    Format: ``{"token_str": int_id, ...}``.  Returns the dict
    verbatim; the algorithm-specific BPE constructor turns it into
    a merge table.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"_load_vocab_json: expected a JSON object at {path}, "
            f"got {type(raw).__name__}"
        )
    return {str(k): int(v) for k, v in raw.items()}


def _load_merges_txt(path: str) -> list[tuple[str, str]]:
    """Parse a HF-compatible ``merges.txt``.

    Format: ``"left right"`` per line, with an optional ``#version: ``
    header that's silently skipped.  Empty lines + ``#``-comments
    are skipped.  Line order = merge rank (lower index = higher
    priority).
    """
    merges: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split(" ")
            if len(parts) != 2:
                # Some merges may legitimately contain a space if the
                # left half ends in / right half starts with whitespace
                # after byte-encoding.  Use rsplit as a defence: the
                # last space is the separator.  This matches HF's
                # tokenizers crate.
                idx = line.rfind(" ")
                if idx <= 0:
                    raise ValueError(
                        f"_load_merges_txt: malformed line {line!r} " f"at {path}"
                    )
                parts = [line[:idx], line[idx + 1 :]]
            merges.append((parts[0], parts[1]))
    return merges


def _load_unified_tokenizer_json(
    path: str,
) -> tuple[dict[str, int], list[tuple[str, str]]]:
    """Parse a Hugging Face Fast-tokenizers ``tokenizer.json``.

    The Rust ``tokenizers`` crate emits this as a single file with
    ``model.vocab`` (dict) + ``model.merges`` (list of "left right"
    strings).  We extract just those two pieces here; normalizer /
    pre-tokenizer config is parsed separately by the
    ``BPETokenizer.from_pretrained`` entry point if present.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    model = data.get("model", {})
    vocab = {str(k): int(v) for k, v in model.get("vocab", {}).items()}
    merges_raw = model.get("merges", [])
    merges: list[tuple[str, str]] = []
    for entry in merges_raw:
        # HF formats: either "left right" string OR ["left", "right"] list
        # (the newer schema as of tokenizers 0.20+).
        if isinstance(entry, list) and len(entry) == 2:
            merges.append((str(entry[0]), str(entry[1])))
        elif isinstance(entry, str):
            idx = entry.rfind(" ")
            if idx <= 0:
                raise ValueError(
                    f"_load_unified_tokenizer_json: malformed merge "
                    f"{entry!r} in {path}"
                )
            merges.append((entry[:idx], entry[idx + 1 :]))
        else:
            raise ValueError(
                f"_load_unified_tokenizer_json: unrecognised merge "
                f"entry type {type(entry).__name__} in {path}"
            )
    return vocab, merges


def _save_vocab_json(vocab: dict[str, int], path: str) -> None:
    """Write a vocab dict back to ``vocab.json`` in HF format."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def _save_merges_txt(
    merges: list[tuple[str, str]], path: str, version: str = "0.2"
) -> None:
    """Write a merge list to ``merges.txt`` with the HF header."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"#version: {version}\n")
        for left, right in merges:
            f.write(f"{left} {right}\n")


# ── Shared mixin: normalizer + pre-tokenizer chain ──────────────────


class _BPECommonMixin:
    """Holds the normalizer + pre-tokenizer chain that both
    :class:`BPETokenizer` and :class:`BPETokenizerFast` use.

    Subclasses still implement their own ``_encode_one`` (because
    the algorithm itself lives in different places — Python loop vs
    C++ call) but the surface preprocessing is shared.
    """

    _normalizer: Normalizer | None
    _pre_tokenizer: PreTokenizer

    def _prepare_chunks(self, text: str) -> list[str]:
        """Apply normalizer + pre-tokenizer, return chunk strings."""
        if self._normalizer is not None:
            text = self._normalizer(text)
        return [chunk for chunk, _ in self._pre_tokenizer(text)]


# ── Pure-Python reference impl ──────────────────────────────────────


class BPETokenizer(_BPECommonMixin, Tokenizer):
    r"""Reference (pure-Python) BPE tokenizer.

    Implements the classical Sennrich-2016 BPE encode loop directly
    in Python.  Easy to read, easy to step through with a debugger,
    works on every platform (no C++ build needed).

    For production / latency-sensitive use, prefer the matching
    :class:`BPETokenizerFast` — same algorithm, same vocab format,
    bit-identical encode outputs, but the hot loop is in C++.

    Parameters
    ----------
    vocab : dict[str, int]
        Token-string → id map.  Must include every single character
        that any merge could produce (so the algorithm has a valid
        starting symbol sequence).
    merges : list[tuple[str, str]]
        Ordered BPE merge pairs — index = rank (lower = higher
        priority, applied first during encoding).
    normalizer : Normalizer, optional
        Pre-encode text normalisation chain.  Default
        :class:`~lucid.utils.tokenizer._normalizers.NFC`.
    pre_tokenizer : PreTokenizer, optional
        Chunk-splitter that turns the normalised text into the
        sequence of strings the BPE algorithm processes one at a
        time.  Default
        :class:`~lucid.utils.tokenizer._pre_tokenizers.WhitespaceSplit`.
    special_tokens : SpecialTokens, optional
        Special-token registry — see
        :class:`lucid.utils.tokenizer.SpecialTokens`.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        *,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        self._vocab: dict[str, int] = dict(vocab)
        self._merges: list[tuple[str, str]] = list(merges)
        self._normalizer = normalizer if normalizer is not None else NFC()
        self._pre_tokenizer = (
            pre_tokenizer if pre_tokenizer is not None else WhitespaceSplit()
        )
        # Compiled merge table: ``(left_id, right_id) -> (merged_id, rank)``.
        self._pair_to_merge: dict[tuple[int, int], tuple[int, int]] = {}
        self._rebuild_tables()
        super().__init__(special_tokens=special_tokens)

    def _rebuild_tables(self) -> None:
        """Recompute :attr:`_pair_to_merge` + the reverse vocab.

        Called after construction and after :meth:`train`; subclasses
        adding mutating operations must call this themselves.
        """
        self._id_to_token: dict[int, str] = {v: k for k, v in self._vocab.items()}
        self._pair_to_merge = {}
        for rank, (left, right) in enumerate(self._merges):
            la = self._vocab.get(left)
            rb = self._vocab.get(right)
            merged = self._vocab.get(left + right)
            if la is None or rb is None or merged is None:
                # Malformed merge — skip silently (matches HF behaviour).
                continue
            self._pair_to_merge[(la, rb)] = (merged, rank)

    # ── Tokenizer interface ────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def algo(self) -> str:
        return "bpe"

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def id_to_token(self, token_id: int) -> str | None:
        return self._id_to_token.get(token_id)

    def _encode_one(self, text: str) -> list[int]:
        """Algorithm-specific encode: chunk → ids via BPE merges."""
        all_ids: list[int] = []
        for chunk in self._prepare_chunks(text):
            all_ids.extend(self._encode_chunk(chunk))
        return all_ids

    def _decode_one(self, ids: list[int]) -> str:
        out: list[str] = []
        for i in ids:
            tok = self._id_to_token.get(i)
            if tok is not None:
                out.append(tok)
        return "".join(out)

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Apply BPE merges to one pre-tokenized chunk.

        1. Split into per-codepoint ids via the vocab (drop unknowns
           or fall back to UNK if defined).
        2. Greedily apply the lowest-rank merge until no more apply.

        O(N · M) where N = chunk length and M = applicable merges.
        N is typically tiny (~5–15 codepoints) so the constant
        factor is dominated by Python's interpreter overhead;
        :class:`BPETokenizerFast` recovers an order of magnitude on
        long-document encoding.
        """
        ids: list[int] = []
        unk_id = self.unk_token_id
        for ch in chunk:
            tid = self._vocab.get(ch)
            if tid is not None:
                ids.append(tid)
            elif unk_id is not None:
                ids.append(unk_id)
        if len(ids) < 2:
            return ids
        while True:
            best_rank = -1
            best_pos = -1
            best_merged = -1
            for k in range(len(ids) - 1):
                entry = self._pair_to_merge.get((ids[k], ids[k + 1]))
                if entry is None:
                    continue
                m_id, rank = entry
                if best_rank == -1 or rank < best_rank:
                    best_rank = rank
                    best_pos = k
                    best_merged = m_id
            if best_pos == -1:
                break
            ids[best_pos] = best_merged
            del ids[best_pos + 1]
        return ids

    # ── Training ────────────────────────────────────────────────────

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int = 30_000,
    ) -> None:
        """Re-train this tokenizer from scratch on ``corpus``.

        Implements the classical Sennrich-2016 training loop:

        1. Pre-tokenize each document (using the configured
           pre-tokenizer chain) into a frequency-counted set of
           words.
        2. Seed the vocab with every distinct codepoint appearing
           in any word.
        3. Iteratively count adjacent symbol-pair frequencies + merge
           the most frequent pair, growing the vocab by 1 each step
           until ``vocab_size`` is reached or no pair appears more
           than once.

        Replaces :attr:`_vocab` + :attr:`_merges` in place; previous
        contents are discarded.

        Parameters
        ----------
        corpus : iterable of str
            Each item is one document (or chunk thereof).  Generators
            are consumed exactly once.
        vocab_size : int, default 30 000
            Target total vocab size (chars + merges).  Stops early if
            no more merges occur.
        """
        # 1. Pre-tokenize + count.
        word_freq: dict[str, int] = {}
        for doc in corpus:
            for chunk in self._prepare_chunks(doc):
                word_freq[chunk] = word_freq.get(chunk, 0) + 1
        # 2. Seed vocab with characters.
        new_vocab: dict[str, int] = {}
        next_id = 0
        # Per-word symbol sequence + frequency (parallel arrays).
        words: list[tuple[list[str], int]] = []
        for w, freq in word_freq.items():
            syms = list(w)  # one entry per codepoint
            for s in syms:
                if s not in new_vocab:
                    new_vocab[s] = next_id
                    next_id += 1
            words.append((syms, freq))
        # 3. Iteratively merge most frequent pair.
        new_merges: list[tuple[str, str]] = []
        while len(new_vocab) < vocab_size:
            pair_freq: dict[tuple[str, str], int] = {}
            for syms, freq in words:
                for k in range(len(syms) - 1):
                    pair = (syms[k], syms[k + 1])
                    pair_freq[pair] = pair_freq.get(pair, 0) + freq
            if not pair_freq:
                break
            # Pick max-count, tie-break on lexicographic order for
            # determinism.
            best_pair = max(pair_freq.items(), key=lambda kv: (kv[1], kv[0]))
            if best_pair[1] < 2:
                break
            left, right = best_pair[0]
            merged = left + right
            new_merges.append((left, right))
            new_vocab[merged] = next_id
            next_id += 1
            # Apply merge across all word symbol sequences.
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
        # Commit + rebuild tables + re-resolve special-token ids.
        self._vocab = new_vocab
        self._merges = new_merges
        self._rebuild_tables()
        self._refresh_special_ids()

    # ── Persistence ─────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Persist as both legacy (``vocab.json`` + ``merges.txt``)
        AND unified (``tokenizer.json``) formats.

        Writing both means the resulting directory loads back via
        :meth:`from_pretrained` regardless of which format the caller
        looks for first — and matches how HF distributes BPE
        checkpoints on the Hub.
        """
        os.makedirs(directory, exist_ok=True)
        _save_vocab_json(self._vocab, os.path.join(directory, "vocab.json"))
        _save_merges_txt(self._merges, os.path.join(directory, "merges.txt"))
        # Delegate to base for tokenizer.json + special_tokens_map.json.
        super().save(directory)

    def _save_extras(self) -> dict[str, object]:
        """Add ``model`` block to the unified ``tokenizer.json``."""
        return {
            "model": {
                "type": "BPE",
                "vocab": self._vocab,
                "merges": [list(m) for m in self._merges],
            }
        }

    @classmethod
    def from_file(
        cls,
        directory: str,
        *,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> BPETokenizer:
        """Load from a directory containing either
        ``tokenizer.json`` or the legacy
        ``vocab.json`` + ``merges.txt`` pair.

        Unified format wins when present.

        Parameters
        ----------
        directory : str
            Path containing the on-disk BPE checkpoint — either the
            unified HF ``tokenizer.json`` or the legacy pair
            (``vocab.json`` + ``merges.txt``).  Optionally also
            ``special_tokens_map.json`` for the special-token registry.
        normalizer : Normalizer, optional, keyword-only
            Override the encode-time normalisation chain.  Defaults
            to :class:`~lucid.utils.tokenizer._normalizers.NFC` when
            not supplied.
        pre_tokenizer : PreTokenizer, optional, keyword-only
            Override the chunk splitter applied after normalisation.
            Defaults to
            :class:`~lucid.utils.tokenizer._pre_tokenizers.WhitespaceSplit`.
        special_tokens : SpecialTokens, optional, keyword-only
            Override the special-token registry.  When ``None`` (the
            default), parsed from ``special_tokens_map.json`` if
            present.

        Returns
        -------
        BPETokenizer
            Freshly-constructed instance ready for encode / decode.
        """
        unified = os.path.join(directory, "tokenizer.json")
        if os.path.isfile(unified):
            vocab, merges = _load_unified_tokenizer_json(unified)
        else:
            vj = os.path.join(directory, "vocab.json")
            mt = os.path.join(directory, "merges.txt")
            if not (os.path.isfile(vj) and os.path.isfile(mt)):
                raise FileNotFoundError(
                    f"BPETokenizer.from_file: neither tokenizer.json "
                    f"nor (vocab.json + merges.txt) found in {directory}"
                )
            vocab = _load_vocab_json(vj)
            merges = _load_merges_txt(mt)
        # Special-tokens map.
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
            normalizer=normalizer,
            pre_tokenizer=pre_tokenizer,
            special_tokens=st,
        )

    from_pretrained = from_file  # alias matching HF naming


def _special_tokens_from_map(sp: dict[str, object]) -> SpecialTokens:
    """Parse ``special_tokens_map.json`` into a :class:`SpecialTokens`.

    Recognises HF's standard keys (``pad_token`` / ``unk_token`` /
    ``bos_token`` / ``eos_token`` / ``cls_token`` / ``sep_token`` /
    ``mask_token``) — each can be either a plain string or
    ``{"content": "<tok>", ...}``-style dict.
    """

    def _stringify(v: object) -> str | None:
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if isinstance(v, dict) and "content" in v:
            return str(v["content"])
        return None

    extra = {}
    add = sp.get("additional_special_tokens", []) or []
    if isinstance(add, list):
        for i, v in enumerate(add):
            s = _stringify(v)
            if s is not None:
                extra[f"additional_{i}"] = s
    return SpecialTokens(
        pad=_stringify(sp.get("pad_token")),
        unk=_stringify(sp.get("unk_token")),
        bos=_stringify(sp.get("bos_token")),
        eos=_stringify(sp.get("eos_token")),
        mask=_stringify(sp.get("mask_token")),
        sep=_stringify(sp.get("sep_token")),
        cls=_stringify(sp.get("cls_token")),
        extra=extra,
    )


# ── Fast C++-backed implementation ──────────────────────────────────


class BPETokenizerFast(_BPECommonMixin, Tokenizer):
    r"""C++-backed BPE tokenizer.

    Identical algorithm + vocab format to :class:`BPETokenizer`; the
    hot loop (per-chunk merge application) runs in C++ via
    :class:`lucid._C.engine.utils.tokenizer.BPE`.  Encode outputs
    are bit-identical for the same vocab + same merges + same
    normalizer + same pre-tokenizer.

    Use this in production training / inference.  Use
    :class:`BPETokenizer` for debugging / extending the algorithm
    with custom Python-only normalizers without touching C++.

    Parameters
    ----------
    Same as :class:`BPETokenizer`.  The C++ backend is constructed
    transparently in ``__init__`` and held as :attr:`_cpp`.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        *,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        self._vocab: dict[str, int] = dict(vocab)
        self._merges: list[tuple[str, str]] = list(merges)
        self._normalizer = normalizer if normalizer is not None else NFC()
        self._pre_tokenizer = (
            pre_tokenizer if pre_tokenizer is not None else WhitespaceSplit()
        )
        # Construct the C++ tokenizer from the same vocab + merges.
        self._cpp = _C_engine.utils.tokenizer.BPE(self._vocab, self._merges)
        # Reverse lookup table for decode (the C++ side also has one,
        # but Python-side access avoids a binding round-trip for the
        # decode hot path).
        self._id_to_token: dict[int, str] = {v: k for k, v in self._vocab.items()}
        super().__init__(special_tokens=special_tokens)
        self._sync_special_tokens_to_cpp()

    def _sync_special_tokens_to_cpp(self) -> None:
        """Mirror :attr:`_special_ids` into the C++ tokenizer's
        ``SpecialTokens`` registry so the C++ encode fall-back path
        (UNK substitution) has the same view of the world.
        """
        st_cpp = _C_engine.utils.tokenizer.SpecialTokens()
        for name in ("pad", "unk", "bos", "eos", "mask", "sep", "cls"):
            tid = self._special_ids.get(name)
            if tid is not None:
                setattr(st_cpp, name, tid)
        # The engine ``SpecialTokens.extra`` binding maps name → token id;
        # its generated stub annotates the value as str, so silence the
        # int-value mismatch here.
        st_cpp.extra = {
            k: v  # type: ignore[misc]
            for k, v in self._special_ids.items()
            if k not in ("pad", "unk", "bos", "eos", "mask", "sep", "cls")
        }
        self._cpp.special_tokens = st_cpp

    # ── Tokenizer interface ────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return self._cpp.vocab_size()

    @property
    def algo(self) -> str:
        return "bpe"

    def get_vocab(self) -> dict[str, int]:
        # Pull from Python-side cache for speed; both are kept in
        # sync by ``train`` + ``__init__``.
        return dict(self._vocab)

    def id_to_token(self, token_id: int) -> str | None:
        return self._id_to_token.get(token_id)

    def _encode_one(self, text: str) -> list[int]:
        """C++ encode path: normalize + pre-tokenize in Python,
        per-chunk merge in C++.
        """
        all_ids: list[int] = []
        for chunk in self._prepare_chunks(text):
            all_ids.extend(self._cpp.encode(chunk))
        return all_ids

    def _decode_one(self, ids: list[int]) -> str:
        # C++ decode for parity.  Note: ``self._cpp.decode`` returns
        # the joined token-string surface form (no special-token
        # stripping — that's the base class's job).
        return self._cpp.decode(list(ids))

    # ── Training ────────────────────────────────────────────────────

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int = 30_000,
    ) -> None:
        """Re-train in C++.

        Materialises ``corpus`` into a list before handing off (the
        C++ binding takes ``std::vector<std::string>``); for very
        large corpora the user is responsible for chunking.  After
        training, the Python-side vocab / merges caches are refreshed
        from the C++ side so subsequent encodes see the new state.

        Parameters
        ----------
        corpus : iterable of str
            Documents to train on — each item is one document (or
            chunk thereof).  Generators are consumed exactly once
            and materialised into a list before crossing into C++.
        vocab_size : int, optional, keyword-only, default=30000
            Target total vocab size (characters + merges).  The C++
            trainer stops early if no symbol pair appears more than
            once.
        """
        corpus_list = list(corpus)
        self._cpp.train(corpus_list, vocab_size)
        self._vocab = dict(self._cpp.get_vocab())
        self._merges = [(a, b) for a, b in self._cpp.merges()]
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        self._refresh_special_ids()
        self._sync_special_tokens_to_cpp()

    # ── Persistence ─────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Same format as :meth:`BPETokenizer.save`."""
        os.makedirs(directory, exist_ok=True)
        _save_vocab_json(self._vocab, os.path.join(directory, "vocab.json"))
        _save_merges_txt(self._merges, os.path.join(directory, "merges.txt"))
        super().save(directory)

    def _save_extras(self) -> dict[str, object]:
        return {
            "model": {
                "type": "BPE",
                "vocab": self._vocab,
                "merges": [list(m) for m in self._merges],
            }
        }

    @classmethod
    def from_file(
        cls,
        directory: str,
        *,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> BPETokenizerFast:
        """Identical loader to :meth:`BPETokenizer.from_file`; the
        only difference is the returned class (and hence the encode
        backend — C++ instead of pure-Python merge loop).

        Parameters
        ----------
        directory : str
            Path containing the on-disk BPE checkpoint — either the
            unified HF ``tokenizer.json`` or the legacy pair
            (``vocab.json`` + ``merges.txt``).  Optionally also
            ``special_tokens_map.json`` for the special-token registry.
        normalizer : Normalizer, optional, keyword-only
            Override the encode-time normalisation chain.  Defaults to
            :class:`~lucid.utils.tokenizer._normalizers.NFC`.
        pre_tokenizer : PreTokenizer, optional, keyword-only
            Override the chunk splitter applied after normalisation.
            Defaults to
            :class:`~lucid.utils.tokenizer._pre_tokenizers.WhitespaceSplit`.
        special_tokens : SpecialTokens, optional, keyword-only
            Override the special-token registry.  When ``None`` (the
            default), parsed from ``special_tokens_map.json`` if
            present in ``directory``.

        Returns
        -------
        BPETokenizerFast
            Freshly-constructed C++-backed instance ready for
            encode / decode.
        """
        unified = os.path.join(directory, "tokenizer.json")
        if os.path.isfile(unified):
            vocab, merges = _load_unified_tokenizer_json(unified)
        else:
            vj = os.path.join(directory, "vocab.json")
            mt = os.path.join(directory, "merges.txt")
            if not (os.path.isfile(vj) and os.path.isfile(mt)):
                raise FileNotFoundError(
                    f"BPETokenizerFast.from_file: neither tokenizer.json "
                    f"nor (vocab.json + merges.txt) found in {directory}"
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
            normalizer=normalizer,
            pre_tokenizer=pre_tokenizer,
            special_tokens=st,
        )

    from_pretrained = from_file
