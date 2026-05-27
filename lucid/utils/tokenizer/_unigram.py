"""UnigramTokenizer — SentencePiece-flavor subword tokenizer.

The algorithm used by T5 / mBART / ALBERT / XLNet / LLaMA / Mistral
(via SentencePiece).  Two flavours:

* :class:`UnigramTokenizer` — pure-Python reference (Viterbi + EM).
* :class:`UnigramTokenizerFast` — C++-backed via
  :class:`lucid._C.engine.utils.tokenizer.Unigram`.

Both share the same on-disk format (``tokenizer.json`` with
``model.vocab = [[piece, log_prob], ...]``).  Encode is Viterbi
over the piece-with-max-log-probability segmentation; training is
EM with vocab pruning (Kudo 2018).

SentencePiece convention: words are typically prefixed with the
"▁" (U+2581) character to mark word starts — this lets decode
recover spaces without ambiguity.  The default
:class:`SentencePiecePreTokenizer` handles this; pass a different
:class:`PreTokenizer` to disable (matches plain Unigram behaviour).
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
    """SentencePiece pre-tokenization: replace whitespace runs with
    a single ``▁`` (U+2581) prefix on each word so decode can
    perfectly reconstruct the original spacing.

    Parameters
    ----------
    add_dummy_prefix : bool, default True
        Prepend ``▁`` to the very first word (matches the canonical
        SentencePiece behaviour).
    """

    SP_SPACE = "▁"  # ▁

    def __init__(self, add_dummy_prefix: bool = True) -> None:
        self._add_dummy_prefix = add_dummy_prefix

    def pre_tokenize(self, text: str) -> list[tuple[str, tuple[int, int]]]:
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
    """Shared normalizer + pre-tokenizer chain for both flavours."""

    _normalizer: Normalizer | None
    _pre_tokenizer: PreTokenizer

    def _prepare_chunks(self, text: str) -> list[str]:
        if self._normalizer is not None:
            text = self._normalizer(text)
        return [chunk for chunk, _ in self._pre_tokenizer(text)]


# ── Pure-Python Unigram ────────────────────────────────────────────


class UnigramTokenizer(_UnigramCommonMixin, Tokenizer):
    r"""Reference (pure-Python) Unigram tokenizer.

    Parameters
    ----------
    pieces : list of (str, float)
        ``(piece_str, log_prob)`` ordered list; index = token id.
    unk_token : str, default "<unk>"
        Fallback piece string.  Must appear in ``pieces`` for the
        UNK id to be defined.
    unk_log_prob : float, default -100.0
        Log probability for the UNK piece.
    normalizer : Normalizer, optional
        Default :class:`NFKC` (matches LLaMA / Mistral).
    pre_tokenizer : PreTokenizer, optional
        Default :class:`SentencePiecePreTokenizer` for canonical
        SentencePiece behaviour.
    special_tokens : SpecialTokens, optional
        Defaults to ``SpecialTokens(unk=unk_token)``.
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
            pre_tokenizer
            if pre_tokenizer is not None
            else SentencePiecePreTokenizer()
        )
        self._piece_to_id: dict[str, int] = {}
        self._id_to_piece: dict[int, str] = {}
        self._max_piece_bytes = 0
        self._rebuild_tables()
        if special_tokens is None:
            special_tokens = SpecialTokens(unk=unk_token)
        super().__init__(special_tokens=special_tokens)

    def _rebuild_tables(self) -> None:
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
        out: list[int] = []
        for chunk in self._prepare_chunks(text):
            out.extend(self._viterbi_encode_chunk(chunk))
        return out

    def _decode_one(self, ids: list[int]) -> str:
        # Standard SentencePiece decode: join surface forms, replace
        # the ▁ marker with a space.
        raw = "".join(
            self._id_to_piece[i] for i in ids if i in self._id_to_piece
        )
        return raw.replace(SentencePiecePreTokenizer.SP_SPACE, " ")

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int = 30_000,
    ) -> None:
        """Train via the C++ Unigram for speed — even the "pure
        Python" UnigramTokenizer delegates training to C++ because
        EM is a tight numerical loop and a Python implementation
        would be 100-1000× slower for any non-trivial corpus.  After
        training we pull the resulting (piece, log_prob) list back
        into the Python state.
        """
        prepared: list[str] = []
        for doc in corpus:
            chunks = self._prepare_chunks(doc)
            prepared.append(" ".join(chunks))
        cpp = _C_engine.utils.tokenizer.Unigram(
            [], self._unk_token, self._unk_log_prob
        )
        cpp.train(prepared, vocab_size)
        self._pieces = [(p, lp) for p, lp in cpp.pieces()]
        self._rebuild_tables()
        self._refresh_special_ids()

    def save(self, directory: str) -> None:
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
    ) -> "UnigramTokenizer":
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
    """C++-backed Unigram tokenizer.  See :class:`UnigramTokenizer`."""

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
            pre_tokenizer
            if pre_tokenizer is not None
            else SentencePiecePreTokenizer()
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
        out: list[int] = []
        for chunk in self._prepare_chunks(text):
            out.extend(self._cpp.encode(chunk))
        return out

    def _decode_one(self, ids: list[int]) -> str:
        raw = self._cpp.decode(list(ids))
        return raw.replace(SentencePiecePreTokenizer.SP_SPACE, " ")

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int = 30_000,
    ) -> None:
        prepared: list[str] = []
        for doc in corpus:
            chunks = self._prepare_chunks(doc)
            prepared.append(" ".join(chunks))
        self._cpp.train(prepared, vocab_size)
        self._pieces = [(p, lp) for p, lp in self._cpp.pieces()]
        self._id_to_piece = {i: p for i, (p, _) in enumerate(self._pieces)}
        self._refresh_special_ids()

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        _save_unigram_pieces_json(
            self._pieces,
            os.path.join(directory, "tokenizer.json"),
            self._unk_token,
            self._unk_log_prob,
        )
        super().save(directory)

    def _save_extras(self) -> dict[str, object]:
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
    ) -> "UnigramTokenizerFast":
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
