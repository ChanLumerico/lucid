"""UnigramTokenizer — SentencePiece-flavour subword tokenizer (Kudo 2018).

The Unigram Language Model tokenizer powers T5 / mBART / ALBERT /
XLNet / LLaMA / Mistral (all via SentencePiece).  Unlike BPE — which
builds a deterministic merge sequence — Unigram maintains a *fixed*
vocabulary of sub-word pieces, each with an associated
log-probability, and at encode time picks the segmentation that
maximises the product of piece probabilities (Viterbi over the
piece-with-max-log-probability lattice).

This file holds **both flavours**:

* :class:`UnigramTokenizer` — pure-Python reference implementation.
  Viterbi decode runs in Python (slow but easy to step through);
  training delegates to C++ because EM with vocab pruning is a tight
  numerical loop and a Python rewrite would be 100-1000x slower for
  any non-trivial corpus.

* :class:`UnigramTokenizerFast` — C++-backed via
  :class:`lucid._C.engine.utils.tokenizer.Unigram`.  Same on-disk
  format, bit-identical encode output, used in production.

**On-disk format.** A single unified ``tokenizer.json`` with
``model.vocab = [[piece, log_prob], ...]`` (matching the Hugging
Face Fast-tokenizers / Rust ``tokenizers`` schema).  Loaded
verbatim by :meth:`UnigramTokenizer.from_file` /
:meth:`UnigramTokenizerFast.from_file`.

**SentencePiece convention.** Words are prefixed with ``▁``
(U+2581, LOWER ONE EIGHTH BLOCK) to mark word starts so that decode
can perfectly reconstruct the original whitespace without
ambiguity.  The default :class:`SentencePiecePreTokenizer`
applies this remapping; pass a different :class:`PreTokenizer` to
disable (matches plain "non-SentencePiece" Unigram behaviour).
"""

import json
import os
from typing import Iterable

from lucid._C import engine as _C_engine

from lucid.utils.tokenizer._base import SpecialTokens, Tokenizer
from lucid.utils.tokenizer._bpe import _special_tokens_from_map
from lucid.utils.tokenizer._normalizers import NFKC, Normalizer
from lucid.utils.tokenizer._pre_tokenizers import PreTokenizer

# ── SentencePiece-style pre-tokenizer ──────────────────────────────


class SentencePiecePreTokenizer(PreTokenizer):
    """SentencePiece pre-tokenization.

    Replaces every whitespace run with a single ``▁`` (U+2581)
    prefix on each word so decode can perfectly reconstruct the
    original spacing.  This is the canonical pre-tokenizer for any
    Unigram / SentencePiece checkpoint (T5, LLaMA, mBART, ...).

    Parameters
    ----------
    add_dummy_prefix : bool, default True
        Prepend ``▁`` to the very first word (matches the canonical
        SentencePiece behaviour).  Set ``False`` for plain Unigram
        without the SentencePiece word-start marker.

    Notes
    -----
    Each emitted chunk keeps the leading ``▁`` so that decode is a
    simple ``"".join(pieces).replace("▁", " ")`` — see
    :meth:`UnigramTokenizer._decode_one`.
    """

    SP_SPACE = "▁"  # ▁

    def __init__(self, add_dummy_prefix: bool = True) -> None:
        self._add_dummy_prefix = add_dummy_prefix

    def pre_tokenize(self, text: str) -> list[tuple[str, tuple[int, int]]]:
        """Replace whitespace with ``▁`` and split on ``▁`` boundaries."""
        # Replace every whitespace run with SP_SPACE.  Optionally
        # prepend SP_SPACE so the first word also carries the
        # word-start marker.
        text = "".join(self.SP_SPACE if c.isspace() else c for c in text)
        if self._add_dummy_prefix and (not text or not text.startswith(self.SP_SPACE)):
            text = self.SP_SPACE + text
        # Split on SP_SPACE boundaries, keeping the SP_SPACE prefix
        # attached to each word for round-trip decode.
        out: list[tuple[str, tuple[int, int]]] = []
        i = 0
        n = len(text)
        while i < n:
            start = i
            # Each chunk = SP_SPACE + word characters (or just word
            # characters for the very first chunk if add_dummy_prefix
            # is False and there's no leading whitespace).
            if text[i] == self.SP_SPACE:
                i += 1
            while i < n and text[i] != self.SP_SPACE:
                i += 1
            out.append((text[start:i], (start, i)))
        return out


# ── Vocab format helpers ───────────────────────────────────────────


def _load_unigram_pieces_json(path: str) -> list[tuple[str, float]]:
    """Parse a HF unified ``tokenizer.json`` for the Unigram
    ``model.vocab`` block.  Returns ``[(piece, log_prob), ...]``."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    model = data.get("model", {})
    vocab = model.get("vocab", [])
    out: list[tuple[str, float]] = []
    for entry in vocab:
        if isinstance(entry, list) and len(entry) == 2:
            out.append((str(entry[0]), float(entry[1])))
        elif isinstance(entry, dict) and "piece" in entry and "score" in entry:
            out.append((str(entry["piece"]), float(entry["score"])))
        else:
            raise ValueError(
                f"_load_unigram_pieces_json: malformed vocab entry "
                f"{entry!r} in {path}"
            )
    return out


def _save_unigram_pieces_json(
    pieces: list[tuple[str, float]],
    path: str,
    unk_token: str,
    unk_log_prob: float,
) -> None:
    """Write the unified ``tokenizer.json`` for Unigram."""
    payload = {
        "algo": "unigram",
        "model": {
            "type": "Unigram",
            "vocab": [list(p) for p in pieces],
            "unk_token": unk_token,
            "unk_log_prob": unk_log_prob,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ── Shared mixin ───────────────────────────────────────────────────


class _UnigramCommonMixin:
    """Shared normalizer + pre-tokenizer chain for both flavours.

    Subclasses still implement their own ``_encode_one`` (Python
    Viterbi vs C++ call) but the surface preprocessing is shared.
    """

    _normalizer: Normalizer | None
    _pre_tokenizer: PreTokenizer

    def _prepare_chunks(self, text: str) -> list[str]:
        """Apply normalizer + pre-tokenizer, return chunk strings."""
        if self._normalizer is not None:
            text = self._normalizer(text)
        return [chunk for chunk, _ in self._pre_tokenizer(text)]


# ── Pure-Python Unigram ────────────────────────────────────────────


class UnigramTokenizer(_UnigramCommonMixin, Tokenizer):
    r"""Reference (pure-Python) Unigram tokenizer.

    Encodes via Viterbi over the lattice of all candidate
    sub-piece segmentations: for each chunk we build a DP table
    ``dp[i]`` = best (highest log-prob) path ending at byte offset
    ``i``, with single-codepoint UNK fallback when no piece spans
    a region.  Decode is a trivial ``"".join(pieces).replace("▁", " ")``.

    Training is delegated to C++ (see :meth:`train` for the
    rationale) — even this "pure-Python" flavour does not implement
    EM in Python because the runtime would be unusable on any
    real corpus.

    For production / latency-sensitive use, prefer
    :class:`UnigramTokenizerFast` — same vocab format, bit-identical
    encode output, but the Viterbi hot loop runs in C++.

    Parameters
    ----------
    pieces : list of (str, float)
        Ordered ``(piece_str, log_prob)`` list; index = token id.
        Larger (less-negative) log-probs are preferred during
        Viterbi decode.
    unk_token : str, default ``"<unk>"``
        Fallback piece string used when no entry in ``pieces``
        spans a region of the input.  Must appear in ``pieces`` for
        the UNK id to be defined; otherwise encode silently drops
        unmatchable codepoints.
    unk_log_prob : float, default ``-100.0``
        Log probability assigned to UNK substitutions in the
        Viterbi recurrence.  Very negative so any non-UNK path
        dominates when available.
    normalizer : Normalizer, optional
        Pre-encode text normalisation chain.  Defaults to
        :class:`~lucid.utils.tokenizer._normalizers.NFKC` (matches
        LLaMA / Mistral / T5).
    pre_tokenizer : PreTokenizer, optional
        Chunk-splitter applied after normalisation.  Defaults to
        :class:`SentencePiecePreTokenizer` for canonical
        SentencePiece behaviour with ``▁`` word-start markers.
    special_tokens : SpecialTokens, optional
        Special-token registry — see
        :class:`lucid.utils.tokenizer.SpecialTokens`.  Defaults to
        ``SpecialTokens(unk=unk_token)``.

    Notes
    -----
    The Viterbi DP runs in :math:`O(N \cdot M)` per chunk, where
    ``N`` is the chunk byte length and ``M`` is the max piece byte
    length.  UTF-8 boundary masking ensures sub-piece offsets only
    land on codepoint boundaries (no mid-codepoint cuts), which
    matches the C++ flavour bit-for-bit.

    See Also
    --------
    UnigramTokenizerFast : C++-backed flavour with identical
        encode output and a much faster :meth:`~UnigramTokenizerFast.encode`.
    """

    def __init__(
        self,
        pieces: list[tuple[str, float]],
        *,
        unk_token: str = "<unk>",
        unk_log_prob: float = -100.0,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        self._pieces: list[tuple[str, float]] = list(pieces)
        self._unk_token = unk_token
        self._unk_log_prob = unk_log_prob
        self._normalizer = normalizer if normalizer is not None else NFKC()
        self._pre_tokenizer = (
            pre_tokenizer if pre_tokenizer is not None else SentencePiecePreTokenizer()
        )
        self._piece_to_id: dict[str, int] = {}
        self._id_to_piece: dict[int, str] = {}
        self._max_piece_bytes = 0
        self._rebuild_tables()
        if special_tokens is None:
            special_tokens = SpecialTokens(unk=unk_token)
        super().__init__(special_tokens=special_tokens)

    def _rebuild_tables(self) -> None:
        """Recompute piece-to-id maps + max-piece byte cache."""
        self._piece_to_id = {p: i for i, (p, _) in enumerate(self._pieces)}
        self._id_to_piece = {i: p for i, (p, _) in enumerate(self._pieces)}
        self._max_piece_bytes = max(
            (len(p.encode("utf-8")) for p, _ in self._pieces), default=0
        )

    @property
    def vocab_size(self) -> int:
        return len(self._pieces)

    @property
    def algo(self) -> str:
        return "unigram"

    def get_vocab(self) -> dict[str, int]:
        return dict(self._piece_to_id)

    def id_to_token(self, token_id: int) -> str | None:
        return self._id_to_piece.get(token_id)

    @property
    def pieces(self) -> list[tuple[str, float]]:
        """The raw ``(piece, log_prob)`` list — useful for inspection
        and for handing off to :class:`UnigramTokenizerFast`."""
        return list(self._pieces)

    def _viterbi_encode_chunk(self, chunk: str) -> list[int]:
        """Reference Viterbi DP — operates on bytes for parity with
        the C++ flavour (UTF-8 boundary mask included)."""
        if not chunk:
            return []
        raw = chunk.encode("utf-8")
        N = len(raw)
        # Mask of "is this byte a codepoint start?"
        is_cp = [False] * (N + 1)
        i = 0
        while i < N:
            is_cp[i] = True
            c0 = raw[i]
            if c0 < 0x80:
                cp_len = 1
            elif (c0 >> 5) == 0b110:
                cp_len = 2
            elif (c0 >> 4) == 0b1110:
                cp_len = 3
            elif (c0 >> 3) == 0b11110:
                cp_len = 4
            else:
                cp_len = 1
            i += cp_len
        is_cp[N] = True

        neg_inf = float("-inf")
        dp = [neg_inf] * (N + 1)
        dp[0] = 0.0
        back: list[tuple[int, int]] = [(-1, -1)] * (N + 1)
        unk_id = self._piece_to_id.get(self._unk_token, -1)

        for i in range(1, N + 1):
            if not is_cp[i]:
                continue
            j_min = max(0, i - self._max_piece_bytes)
            for j in range(j_min, i):
                if not is_cp[j]:
                    continue
                if dp[j] == neg_inf:
                    continue
                sub_bytes = raw[j:i]
                try:
                    sub = sub_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                tid = self._piece_to_id.get(sub)
                if tid is not None:
                    score = dp[j] + self._pieces[tid][1]
                    if score > dp[i]:
                        dp[i] = score
                        back[i] = (j, tid)
                elif unk_id >= 0:
                    # Single-codepoint UNK fallback.
                    c0 = raw[j]
                    if c0 < 0x80:
                        cp_len = 1
                    elif (c0 >> 5) == 0b110:
                        cp_len = 2
                    elif (c0 >> 4) == 0b1110:
                        cp_len = 3
                    elif (c0 >> 3) == 0b11110:
                        cp_len = 4
                    else:
                        cp_len = 1
                    if i - j == cp_len:
                        score = dp[j] + self._unk_log_prob
                        if score > dp[i]:
                            dp[i] = score
                            back[i] = (j, unk_id)
        if dp[N] == neg_inf:
            return []
        ids: list[int] = []
        i = N
        while i > 0:
            j, pid = back[i]
            if pid >= 0:
                ids.append(pid)
            i = j
        ids.reverse()
        return ids

    def _encode_one(self, text: str) -> list[int]:
        """Normalize + pre-tokenize + per-chunk Viterbi encode."""
        out: list[int] = []
        for chunk in self._prepare_chunks(text):
            out.extend(self._viterbi_encode_chunk(chunk))
        return out

    def _decode_one(self, ids: list[int]) -> str:
        """Concatenate pieces and convert ``▁`` markers back to spaces."""
        # Standard SentencePiece decode: join surface forms, replace
        # the ▁ marker with a space.
        raw = "".join(self._id_to_piece[i] for i in ids if i in self._id_to_piece)
        return raw.replace(SentencePiecePreTokenizer.SP_SPACE, " ")

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int = 30_000,
    ) -> None:
        """Re-train this tokenizer from scratch on ``corpus``.

        Implements the Kudo-2018 EM-with-pruning training loop:

        1. Pre-tokenize each document (using the configured
           pre-tokenizer chain) into chunks.
        2. Seed a large candidate vocab from all sub-strings, then
           iteratively run EM to estimate per-piece probabilities
           and prune the lowest-contribution pieces until the target
           vocab size is reached.
        3. Re-load the resulting ``(piece, log_prob)`` list into
           Python state.

        Even this "pure-Python" flavour delegates the inner loop to
        C++ — EM is a tight numerical loop and a Python rewrite
        would be 100-1000x slower for any non-trivial corpus.

        Parameters
        ----------
        corpus : iterable of str
            Each item is one document (or chunk thereof).  Generators
            are consumed exactly once and materialised into a list
            before handing off to the C++ trainer.
        vocab_size : int, default 30 000
            Target total vocab size (pieces).  The trainer stops
            pruning when this is reached.
        """
        prepared: list[str] = []
        for doc in corpus:
            chunks = self._prepare_chunks(doc)
            prepared.append(" ".join(chunks))
        cpp = _C_engine.utils.tokenizer.Unigram([], self._unk_token, self._unk_log_prob)
        cpp.train(prepared, vocab_size)
        self._pieces = [(p, lp) for p, lp in cpp.pieces()]
        self._rebuild_tables()
        self._refresh_special_ids()

    def save(self, directory: str) -> None:
        """Persist as unified ``tokenizer.json`` + ``special_tokens_map.json``.

        Parameters
        ----------
        directory : str
            Output directory (created if missing).  Contents are
            HF-compatible: any other library that reads the unified
            Fast-tokenizers format will load them back unchanged.
        """
        os.makedirs(directory, exist_ok=True)
        _save_unigram_pieces_json(
            self._pieces,
            os.path.join(directory, "tokenizer.json"),
            self._unk_token,
            self._unk_log_prob,
        )
        # Also write the special_tokens_map.json via the base.
        super().save(directory)

    def _save_extras(self) -> dict[str, object]:
        """Add the ``model`` block to the unified ``tokenizer.json``."""
        return {
            "model": {
                "type": "Unigram",
                "vocab": [list(p) for p in self._pieces],
                "unk_token": self._unk_token,
                "unk_log_prob": self._unk_log_prob,
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
    ) -> UnigramTokenizer:
        """Load from a directory containing ``tokenizer.json``.

        Parameters
        ----------
        directory : str
            Directory holding the unified ``tokenizer.json`` (and
            optionally ``special_tokens_map.json``).
        normalizer : Normalizer, optional, keyword-only
            Override the encode-time normalisation chain.  Defaults
            to :class:`~lucid.utils.tokenizer._normalizers.NFKC`
            when omitted (matches LLaMA / Mistral / T5).
        pre_tokenizer : PreTokenizer, optional, keyword-only
            Override the chunk splitter applied after normalisation.
            Defaults to :class:`SentencePiecePreTokenizer` for
            canonical SentencePiece behaviour.
        special_tokens : SpecialTokens, optional, keyword-only
            Override the special-token registry.  When ``None`` (the
            default), parsed from ``special_tokens_map.json`` if
            present.

        Returns
        -------
        UnigramTokenizer
            Freshly-constructed instance ready for encode / decode.
        """
        path = os.path.join(directory, "tokenizer.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"UnigramTokenizer.from_file: tokenizer.json not found in "
                f"{directory}"
            )
        pieces = _load_unigram_pieces_json(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        model = data.get("model", {})
        unk_token = str(model.get("unk_token", "<unk>"))
        unk_log_prob = float(model.get("unk_log_prob", -100.0))
        st = special_tokens
        if st is None:
            sp_path = os.path.join(directory, "special_tokens_map.json")
            if os.path.isfile(sp_path):
                with open(sp_path, encoding="utf-8") as f:
                    sp = json.load(f)
                st = _special_tokens_from_map(sp)
        return cls(
            pieces,
            unk_token=unk_token,
            unk_log_prob=unk_log_prob,
            normalizer=normalizer,
            pre_tokenizer=pre_tokenizer,
            special_tokens=st,
        )

    from_pretrained = from_file


# ── Fast (C++-backed) Unigram ──────────────────────────────────────


class UnigramTokenizerFast(_UnigramCommonMixin, Tokenizer):
    r"""C++-backed Unigram tokenizer.

    Identical algorithm + on-disk format to
    :class:`UnigramTokenizer`; the per-chunk Viterbi loop runs in
    C++ via :class:`lucid._C.engine.utils.tokenizer.Unigram`.
    Encode outputs are bit-identical for the same pieces + same
    normalizer + same pre-tokenizer.

    Use this in production training / inference.  Use
    :class:`UnigramTokenizer` for debugging / extending the
    algorithm with custom Python-only normalizers without touching
    C++.

    Parameters
    ----------
    Same as :class:`UnigramTokenizer`.  The C++ backend is
    constructed transparently in ``__init__`` and held as
    :attr:`_cpp`.

    See Also
    --------
    UnigramTokenizer : Pure-Python reference flavour.
    """

    def __init__(
        self,
        pieces: list[tuple[str, float]],
        *,
        unk_token: str = "<unk>",
        unk_log_prob: float = -100.0,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        self._pieces: list[tuple[str, float]] = list(pieces)
        self._unk_token = unk_token
        self._unk_log_prob = unk_log_prob
        self._cpp = _C_engine.utils.tokenizer.Unigram(
            [(p, lp) for p, lp in self._pieces], unk_token, unk_log_prob
        )
        self._id_to_piece: dict[int, str] = {
            i: p for i, (p, _) in enumerate(self._pieces)
        }
        self._normalizer = normalizer if normalizer is not None else NFKC()
        self._pre_tokenizer = (
            pre_tokenizer if pre_tokenizer is not None else SentencePiecePreTokenizer()
        )
        if special_tokens is None:
            special_tokens = SpecialTokens(unk=unk_token)
        super().__init__(special_tokens=special_tokens)

    @property
    def vocab_size(self) -> int:
        return self._cpp.vocab_size()

    @property
    def algo(self) -> str:
        return "unigram"

    def get_vocab(self) -> dict[str, int]:
        return dict(self._cpp.get_vocab())

    def id_to_token(self, token_id: int) -> str | None:
        return self._id_to_piece.get(token_id)

    @property
    def pieces(self) -> list[tuple[str, float]]:
        return list(self._pieces)

    def _encode_one(self, text: str) -> list[int]:
        """Normalize + pre-tokenize in Python, per-chunk Viterbi in C++."""
        out: list[int] = []
        for chunk in self._prepare_chunks(text):
            out.extend(self._cpp.encode(chunk))
        return out

    def _decode_one(self, ids: list[int]) -> str:
        """C++ decode + ``▁`` → space replacement for parity."""
        raw = self._cpp.decode(list(ids))
        return raw.replace(SentencePiecePreTokenizer.SP_SPACE, " ")

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int = 30_000,
    ) -> None:
        """Re-train in C++ (EM with vocab pruning, Kudo 2018).

        Materialises ``corpus`` into a list before handing off (the
        C++ binding takes ``std::vector<std::string>``); for very
        large corpora the caller is responsible for chunking.  After
        training, the Python-side pieces cache is refreshed from the
        C++ side so subsequent encodes see the new state.

        Parameters
        ----------
        corpus : iterable of str
            Documents to train on.
        vocab_size : int, default 30 000
            Target piece count after EM pruning.
        """
        prepared: list[str] = []
        for doc in corpus:
            chunks = self._prepare_chunks(doc)
            prepared.append(" ".join(chunks))
        self._cpp.train(prepared, vocab_size)
        self._pieces = [(p, lp) for p, lp in self._cpp.pieces()]
        self._id_to_piece = {i: p for i, (p, _) in enumerate(self._pieces)}
        self._refresh_special_ids()

    def save(self, directory: str) -> None:
        """Same format as :meth:`UnigramTokenizer.save`."""
        os.makedirs(directory, exist_ok=True)
        _save_unigram_pieces_json(
            self._pieces,
            os.path.join(directory, "tokenizer.json"),
            self._unk_token,
            self._unk_log_prob,
        )
        super().save(directory)

    def _save_extras(self) -> dict[str, object]:
        """Add the ``model`` block to the unified ``tokenizer.json``."""
        return {
            "model": {
                "type": "Unigram",
                "vocab": [list(p) for p in self._pieces],
                "unk_token": self._unk_token,
                "unk_log_prob": self._unk_log_prob,
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
    ) -> UnigramTokenizerFast:
        """Identical loader to :meth:`UnigramTokenizer.from_file`.

        The only difference is the returned class (and hence the
        encode backend — C++ instead of Python Viterbi).

        Parameters
        ----------
        directory : str
            Directory holding the unified ``tokenizer.json`` (and
            optionally ``special_tokens_map.json``).
        normalizer : Normalizer, optional, keyword-only
            Override the encode-time normalisation chain.  Defaults
            to :class:`~lucid.utils.tokenizer._normalizers.NFKC`.
        pre_tokenizer : PreTokenizer, optional, keyword-only
            Override the chunk splitter applied after normalisation.
            Defaults to :class:`SentencePiecePreTokenizer`.
        special_tokens : SpecialTokens, optional, keyword-only
            Override the special-token registry.  When ``None`` (the
            default), parsed from ``special_tokens_map.json`` if
            present.

        Returns
        -------
        UnigramTokenizerFast
            Freshly-constructed C++-backed instance ready for
            encode / decode.
        """
        path = os.path.join(directory, "tokenizer.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"UnigramTokenizerFast.from_file: tokenizer.json not found "
                f"in {directory}"
            )
        pieces = _load_unigram_pieces_json(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        model = data.get("model", {})
        unk_token = str(model.get("unk_token", "<unk>"))
        unk_log_prob = float(model.get("unk_log_prob", -100.0))
        st = special_tokens
        if st is None:
            sp_path = os.path.join(directory, "special_tokens_map.json")
            if os.path.isfile(sp_path):
                with open(sp_path, encoding="utf-8") as f:
                    sp = json.load(f)
                st = _special_tokens_from_map(sp)
        return cls(
            pieces,
            unk_token=unk_token,
            unk_log_prob=unk_log_prob,
            normalizer=normalizer,
            pre_tokenizer=pre_tokenizer,
            special_tokens=st,
        )

    from_pretrained = from_file
