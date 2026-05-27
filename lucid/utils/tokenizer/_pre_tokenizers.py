"""Pre-tokenizers — split the normalised text into algorithm-ready chunks.

Every BPE / WordPiece / Unigram implementation expects its
``_encode_one`` hook to receive **one chunk** at a time (typically
one word).  The pre-tokenizer chain is what produces those chunks
from the raw post-normalised string; it sits between the
:class:`~lucid.utils.tokenizer._normalizers.Normalizer` step and
the algorithm-specific encoder.

The standard chains map to algorithm families:

* :class:`WhitespaceSplit` — split on Unicode whitespace + drop
  empties.  Used by classical BPE (without byte-level remapping)
  and by the SentencePiece pre-stage of LLaMA / T5.
* :class:`WhitespacePunctuationSplit` — split on whitespace AND
  separate every punctuation character as its own chunk.  Standard
  for WordPiece (BERT / DistilBERT / RoFormer / ALBERT) — matches
  the reference framework's ``BasicTokenizer`` after normalisation.
* :class:`ByteLevel` — re-map every UTF-8 byte to a printable
  Unicode character (the GPT-2 byte-to-Unicode table), then split
  before every ``Ġ``-prefixed token (``Ġ`` = the original space
  byte under the mapping).  Used by GPT-2 / RoBERTa / BART
  byte-level BPE.  Guarantees the BPE algorithm never sees raw
  bytes or whitespace.

**Output contract.** Every pre-tokenizer returns
``list[tuple[str, tuple[int, int]]]`` where each tuple is
``(chunk, (start, end))`` — the chunk text plus its codepoint
offsets into the input string.  Offsets feed HF-compatible
``return_offsets_mapping`` for span-level downstream tasks (NER,
extractive QA, etc.).
"""

from abc import ABC, abstractmethod
import unicodedata

# Public type alias for pre-tokenizer outputs.
Chunk = tuple[str, tuple[int, int]]


class PreTokenizer(ABC):
    """Abstract base — every pre-tokenizer implements :meth:`pre_tokenize`.

    Subclasses are callable: ``pre_tokenizer(text)`` is sugar for
    ``pre_tokenizer.pre_tokenize(text)``.

    See Also
    --------
    WhitespaceSplit, WhitespacePunctuationSplit, ByteLevel
    """

    @abstractmethod
    def pre_tokenize(self, text: str) -> list[Chunk]:
        """Split ``text`` into ``(chunk, (start, end))`` tuples.

        Parameters
        ----------
        text : str
            Post-normalised input string.

        Returns
        -------
        list of (str, (int, int))
            One tuple per chunk.  Offsets are codepoint indices into
            ``text`` (Python ``str`` indexing is codepoint-based —
            the convention used by every HF-compatible tokenizer for
            ``return_offsets_mapping``).
        """

    def __call__(self, text: str) -> list[Chunk]:
        """Sugar for :meth:`pre_tokenize`."""
        return self.pre_tokenize(text)


class WhitespaceSplit(PreTokenizer):
    """Split on Unicode whitespace, dropping empty chunks.

    The default for classical BPE pre-tokenization (LLaMA, T5
    after the SentencePiece pre-stage, GPT-3-style word BPE, ...).
    Uses Python's :meth:`str.isspace` so all Unicode whitespace
    categories count (regular space, tab, newline, NBSP, etc.).

    See Also
    --------
    WhitespacePunctuationSplit : Also splits on punctuation.
    ByteLevel : Byte-level GPT-2 / RoBERTa pre-tokenizer.
    """

    def pre_tokenize(self, text: str) -> list[Chunk]:
        """Split ``text`` on whitespace runs, dropping empty chunks."""
        out: list[Chunk] = []
        i = 0
        n = len(text)
        while i < n:
            # Skip leading whitespace.
            while i < n and text[i].isspace():
                i += 1
            if i >= n:
                break
            start = i
            # Consume word.
            while i < n and not text[i].isspace():
                i += 1
            out.append((text[start:i], (start, i)))
        return out


class WhitespacePunctuationSplit(PreTokenizer):
    """Split on whitespace AND emit each punctuation character as its
    own chunk.

    The standard pre-tokenizer for WordPiece (BERT / RoFormer /
    ALBERT / DistilBERT) — matches the canonical ``BasicTokenizer``
    after normalisation, ensuring punctuation is never merged with
    adjacent word characters during sub-word splitting.

    Notes
    -----
    Punctuation detection follows the BERT convention: ASCII
    punctuation ranges (``!``-``/``, ``:``-``@``, ``[``-`````,
    ``{``-``~``) plus every Unicode ``P*`` category — see
    :func:`_is_punctuation`.
    """

    def pre_tokenize(self, text: str) -> list[Chunk]:
        """Walk ``text`` emitting word / punctuation chunks separately."""
        out: list[Chunk] = []
        i = 0
        n = len(text)
        while i < n:
            # Skip whitespace.
            while i < n and text[i].isspace():
                i += 1
            if i >= n:
                break
            ch = text[i]
            if _is_punctuation(ch):
                # Punctuation is always its own chunk.
                out.append((ch, (i, i + 1)))
                i += 1
            else:
                start = i
                while i < n and not text[i].isspace() and not _is_punctuation(text[i]):
                    i += 1
                out.append((text[start:i], (start, i)))
        return out


class ByteLevel(PreTokenizer):
    r"""GPT-2 byte-level pre-tokenizer.

    Re-maps every UTF-8 byte to a printable Unicode codepoint via
    the canonical GPT-2 byte-to-Unicode table, then chunks the
    result along word / digit / punctuation boundaries (each chunk
    may absorb a single leading space, which after the byte mapping
    appears as the famous ``Ġ`` prefix — ``Ġ`` = ``chr(0x100 + 0)``,
    the image of byte ``0x20`` under the table).

    **Byte-to-Unicode mapping.** Bytes that are already printable
    ASCII / Latin-1 (``0x21``-``0x7E`` + ``0xA1``-``0xAC`` +
    ``0xAE``-``0xFF``) map to themselves; every other byte
    (``0x00``-``0x20``, ``0x7F``-``0xA0``, ``0xAD``) is shifted
    into the ``0x100``+ block where it lands on a printable,
    non-whitespace codepoint.  This guarantees that the BPE
    algorithm downstream sees no whitespace and no control
    characters — making "byte-level BPE" identical to "BPE over
    the mapped Unicode string".

    The mapping is bijective on bytes ``0x00``-``0xFF``, so a full
    inverse (:meth:`decode_bytes`) reconstructs the original byte
    sequence exactly.  This is what lets GPT-2 / RoBERTa / BART
    losslessly round-trip arbitrary bytes (including emoji, control
    characters, invalid UTF-8) through their tokenizers.

    Parameters
    ----------
    add_prefix_space : bool, default ``False``
        Whether to prepend a space to the input so the first word
        also gets a ``Ġ`` prefix after byte-encoding.  GPT-2 default
        is ``False``; RoBERTa / BART default is ``True``.

    Notes
    -----
    The byte-to-Unicode table is built lazily on first use and
    cached on the class — construction cost is paid once per process.

    See Also
    --------
    WhitespaceSplit : Simpler whitespace-only chunker for classical BPE.
    """

    # Compiled once per process — the table is small (256 entries).
    _byte_encoder: dict[int, str] = {}
    _byte_decoder: dict[str, int] = {}

    def __init__(self, *, add_prefix_space: bool = False) -> None:
        self._add_prefix_space = add_prefix_space
        if not ByteLevel._byte_encoder:
            ByteLevel._build_byte_tables_()

    @classmethod
    def _build_byte_tables_(cls) -> None:
        """Build the GPT-2 byte-to-unicode mapping (lazy, once).

        Bytes 0x21–0x7E + 0xA1–0xAC + 0xAE–0xFF map to themselves;
        every other byte (0–0x20, 0x7F–0xA0, 0xAD) gets shifted up
        by 0x100 into the Unicode range so the result is always
        printable + non-whitespace.
        """
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        cls._byte_encoder = {b: chr(c) for b, c in zip(bs, cs)}
        cls._byte_decoder = {chr(c): b for b, c in zip(bs, cs)}

    @classmethod
    def encode_bytes(cls, raw: bytes) -> str:
        """Re-map every byte in ``raw`` to its GPT-2 printable form."""
        if not cls._byte_encoder:
            cls._build_byte_tables_()
        return "".join(cls._byte_encoder[b] for b in raw)

    @classmethod
    def decode_bytes(cls, encoded: str) -> bytes:
        """Inverse of :meth:`encode_bytes`."""
        if not cls._byte_decoder:
            cls._build_byte_tables_()
        return bytes(cls._byte_decoder[c] for c in encoded)

    def pre_tokenize(self, text: str) -> list[Chunk]:
        """Chunk ``text`` along word/digit/punctuation boundaries
        and byte-encode each chunk."""
        # GPT-2 splits on the regex:
        #   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+
        # We approximate without ``regex`` (which isn't a stdlib
        # module): walk the string char-by-char + emit chunks at
        # word / digit / punctuation boundaries.  Each chunk
        # absorbs a leading space if one preceded it (this is what
        # the ``Ġ`` prefix represents post-byte-encode).
        if self._add_prefix_space and (not text or not text[0].isspace()):
            text = " " + text
        out: list[Chunk] = []
        i = 0
        n = len(text)
        while i < n:
            start = i
            # Optional leading space (matches GPT-2's `` ?`` capture).
            if text[i] == " ":
                i += 1
                if i >= n:
                    out.append((text[start:i], (start, i)))
                    break
            ch = text[i]
            if ch.isalpha():
                while i < n and text[i].isalpha():
                    i += 1
            elif ch.isdigit():
                while i < n and text[i].isdigit():
                    i += 1
            elif ch.isspace():
                # Consecutive whitespace (after the optional leading
                # space) — emit each as its own chunk.
                while i < n and text[i].isspace():
                    i += 1
            else:
                # Punctuation / symbols — consume a run.
                while i < n and not text[i].isalnum() and not text[i].isspace():
                    i += 1
            if i > start:
                chunk_text = text[start:i]
                # Byte-encode the chunk for the algorithm's consumption.
                encoded = ByteLevel.encode_bytes(chunk_text.encode("utf-8"))
                out.append((encoded, (start, i)))
        return out


def _is_punctuation(ch: str) -> bool:
    """Mirror BERT's ``_is_punctuation``: ASCII punctuation + every
    Unicode ``P*`` category."""
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(ch).startswith("P")
