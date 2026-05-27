"""Pre-tokenizers — split the normalised text into algorithm-ready chunks.

Every BPE / WordPiece / Unigram implementation expects its
``_encode_one`` hook to receive **one chunk** (typically one word).
The pre-tokenizer chain is what produces those chunks from the raw
post-normalised string.

The standard chains map to algorithm families:

* :class:`WhitespaceSplit` — split on whitespace + drop empties.
  Used by classical BPE (without byte-level remapping).
* :class:`WhitespacePunctuationSplit` — split on whitespace + also
  separate punctuation as its own chunks.  Standard for WordPiece
  (BERT family).
* :class:`ByteLevel` — re-map every byte to a printable Unicode
  character so the algorithm sees no whitespace at all, then split
  before every ``Ġ``-prefixed token (Ġ = the original space byte).
  Used by GPT-2 / RoBERTa byte-level BPE.

Every pre-tokenizer returns ``list[tuple[str, tuple[int, int]]]``
where each tuple is ``(chunk, (start, end))`` — the chunk text plus
its byte offsets into the original input.  Offsets feed
HF-compatible ``return_offsets_mapping`` for span-level downstream
tasks (NER, QA).
"""

from abc import ABC, abstractmethod
import unicodedata

# Public type alias for pre-tokenizer outputs.
Chunk = tuple[str, tuple[int, int]]


class PreTokenizer(ABC):
    """Base — every pre-tokenizer implements :meth:`pre_tokenize`."""

    @abstractmethod
    def pre_tokenize(self, text: str) -> list[Chunk]:
        """Split ``text`` into ``(chunk, (start, end))`` tuples.

        Offsets are byte (= character for ASCII; codepoint for
        Unicode strings since Python ``str`` indexing is codepoint-
        based, which is what HF uses by convention).
        """

    def __call__(self, text: str) -> list[Chunk]:
        return self.pre_tokenize(text)


class WhitespaceSplit(PreTokenizer):
    """Split on Unicode whitespace, dropping empty chunks.

    The default for classical BPE pre-tokenization (LLaMA, T5
    after sentencepiece pre-processing, ...).
    """

    def pre_tokenize(self, text: str) -> list[Chunk]:
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
    """Split on whitespace AND treat each punctuation character as its
    own chunk.

    The standard pre-tokenizer for WordPiece (BERT / RoFormer /
    ALBERT / DistilBERT) — matches BERT's ``BasicTokenizer`` after
    normalization.
    """

    def pre_tokenize(self, text: str) -> list[Chunk]:
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
    """GPT-2 byte-level pre-tokenizer.

    Re-maps every UTF-8 byte to a printable Unicode codepoint via the
    standard GPT-2 byte-to-unicode table (bytes ≥ 0x21 + a-z + A-Z
    map to themselves; bytes 0x00–0x20 + 0x7F–0xA0 + others are
    shifted up by 0x100 into a "safe" Unicode block).  Then splits
    on the ``Ġ``-prefixed regex used by GPT-2.

    The point: every input byte produces exactly one printable
    codepoint, and the BPE algorithm never needs to handle raw
    bytes — making "byte-level BPE" identical to "BPE over the
    mapped Unicode string".

    Parameters
    ----------
    add_prefix_space : bool, default False
        Whether to prepend a space to the input so the first word
        also gets a ``Ġ`` prefix.  GPT-2 default is ``False``;
        RoBERTa default is ``True``.
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
