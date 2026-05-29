"""Text-normalisation primitives composable into a normalizer chain.

A :class:`Normalizer` is any callable ``str -> str`` applied to the
raw input text **before** pre-tokenization.  The standard chain for
HF-style tokenizers is

    Sequence([NFD(), Lowercase(), StripAccents()])  # BERT-uncased
    Sequence([NFC()])                                # GPT-2
    Sequence([NFKC(), Replace(...)])                 # LLaMA / Mistral

so the API mirrors HF's: each primitive is its own class so config
serialisation (algo name + args) round-trips cleanly through
``tokenizer.json``.

Pure Python — these run once per encode call on the surface string
and aren't the hot path (encoding dominates).  No C++ counterpart.
"""

from abc import ABC, abstractmethod
import unicodedata


class Normalizer(ABC):
    """Abstract base — every concrete normalizer implements
    :meth:`normalize`.

    Composable via :class:`Sequence` for the canonical chain
    semantics (apply primitives left-to-right).
    """

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Return the normalised form of ``text``."""

    def __call__(self, text: str) -> str:
        return self.normalize(text)


class Sequence(Normalizer):
    """Apply a series of normalizers in order.

    The standard HF normalizer chain for any tokenizer can be
    expressed as ``Sequence([N1, N2, ...])``.
    """

    def __init__(self, normalizers: list[Normalizer]) -> None:
        self._normalizers = list(normalizers)

    def normalize(self, text: str) -> str:
        for n in self._normalizers:
            text = n.normalize(text)
        return text

    def __repr__(self) -> str:
        return f"Sequence({self._normalizers!r})"


class NFC(Normalizer):
    """Unicode NFC (Canonical Composition) — used by GPT-2 and friends.

    Combines decomposed characters back into their composed forms
    (e.g. ``a + ◌́  → á``).  No-op on most ASCII text.
    """

    def normalize(self, text: str) -> str:
        return unicodedata.normalize("NFC", text)


class NFD(Normalizer):
    """Unicode NFD (Canonical Decomposition).

    Splits composed characters into base + combining marks (the
    inverse of NFC).  Used by BERT-uncased as a precursor to
    :class:`StripAccents`.
    """

    def normalize(self, text: str) -> str:
        return unicodedata.normalize("NFD", text)


class NFKC(Normalizer):
    """Unicode NFKC (Compatibility Composition) — used by LLaMA /
    Mistral.

    Like NFC, but additionally folds compatibility characters
    (e.g. fullwidth digits → ASCII digits).
    """

    def normalize(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text)


class NFKD(Normalizer):
    r"""Unicode NFKD (Compatibility Decomposition) — splits composed
    characters AND folds compatibility forms.

    Like :class:`NFD`, but additionally decomposes compatibility
    characters (ligatures, super-/sub-scripts, fullwidth digits, …)
    into their canonical components.  Commonly chained before
    :class:`StripAccents` for the most aggressive normalisation
    pipelines.

    Notes
    -----
    NFKD is the most lossy normalisation form — ``½`` becomes
    ``1⁄2``, ``ﬁ`` becomes ``f + i``, fullwidth digits collapse to
    ASCII, etc.  Use :class:`NFD` instead if you only want canonical
    decomposition without the compatibility fold.
    """

    def normalize(self, text: str) -> str:
        return unicodedata.normalize("NFKD", text)


class Lowercase(Normalizer):
    r"""Lowercase every character via :meth:`str.lower`.

    Used by ``bert-base-uncased`` and any other uncased checkpoint;
    typically chained after :class:`NFD` + :class:`StripAccents` so
    accented uppercase letters fold to their unaccented lowercase
    forms in one pass.

    Notes
    -----
    Locale-independent — uses Python's default Unicode case folding,
    which matches HF's reference implementation but differs from
    locale-specific folds (e.g. Turkish dotless-I).
    """

    def normalize(self, text: str) -> str:
        return text.lower()


class StripAccents(Normalizer):
    """Drop every combining mark.

    Must be preceded by :class:`NFD` (or :class:`NFKD`) so accents
    have been split off the base characters; otherwise composed
    characters pass through unchanged.
    """

    def normalize(self, text: str) -> str:
        return "".join(c for c in text if not unicodedata.combining(c))


class Strip(Normalizer):
    """Strip leading / trailing whitespace.

    Parameters
    ----------
    left : bool, default True
        Strip leading whitespace.
    right : bool, default True
        Strip trailing whitespace.
    """

    def __init__(self, left: bool = True, right: bool = True) -> None:
        self._left = left
        self._right = right

    def normalize(self, text: str) -> str:
        if self._left and self._right:
            return text.strip()
        if self._left:
            return text.lstrip()
        if self._right:
            return text.rstrip()
        return text


class Replace(Normalizer):
    """Literal-string substitution.

    Parameters
    ----------
    pattern : str
        Substring to replace.
    replacement : str
        Replacement string.
    """

    def __init__(self, pattern: str, replacement: str) -> None:
        self._pattern = pattern
        self._replacement = replacement

    def normalize(self, text: str) -> str:
        return text.replace(self._pattern, self._replacement)


class BERTNormalizer(Normalizer):
    """Composite normalizer matching BERT's standard pipeline.

    Parameters
    ----------
    lowercase : bool, default True
        Apply :class:`Lowercase` after NFD (matches BERT-uncased).
    strip_accents : bool, default True
        Apply :class:`StripAccents` after NFD.
    clean_text : bool, default True
        Replace control characters with spaces + collapse
        whitespace runs into single spaces.  Matches BERT's
        ``BasicTokenizer._clean_text``.
    handle_chinese_chars : bool, default True
        Wrap every CJK ideograph with spaces so the
        whitespace pre-tokenizer treats them as standalone tokens
        (matches BERT-Chinese / Multilingual-BERT behaviour).
    """

    def __init__(
        self,
        *,
        lowercase: bool = True,
        strip_accents: bool = True,
        clean_text: bool = True,
        handle_chinese_chars: bool = True,
    ) -> None:
        self._lowercase = lowercase
        self._strip_accents = strip_accents
        self._clean_text = clean_text
        self._handle_chinese_chars = handle_chinese_chars

    def normalize(self, text: str) -> str:
        if self._clean_text:
            text = self._do_clean_text(text)
        if self._handle_chinese_chars:
            text = self._do_handle_chinese_chars(text)
        text = unicodedata.normalize("NFD", text)
        if self._strip_accents:
            text = "".join(c for c in text if not unicodedata.combining(c))
        if self._lowercase:
            text = text.lower()
        return text

    @staticmethod
    def _do_clean_text(text: str) -> str:
        """Replace control characters with spaces; collapse other
        whitespace into single spaces."""
        out: list[str] = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or _is_control(ch):
                continue
            if _is_whitespace(ch):
                out.append(" ")
            else:
                out.append(ch)
        return "".join(out)

    @staticmethod
    def _do_handle_chinese_chars(text: str) -> str:
        """Surround every CJK ideograph with spaces."""
        out: list[str] = []
        for ch in text:
            cp = ord(ch)
            if _is_cjk(cp):
                out.append(" ")
                out.append(ch)
                out.append(" ")
            else:
                out.append(ch)
        return "".join(out)


def _is_whitespace(ch: str) -> bool:
    """Mirror BERT's ``_is_whitespace``: spaces + tabs + line breaks."""
    if ch in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(ch) == "Zs"


def _is_control(ch: str) -> bool:
    """Mirror BERT's ``_is_control``: control chars excluding standard
    whitespace."""
    if ch in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(ch).startswith("C")


def _is_cjk(cp: int) -> bool:
    """CJK Unicode ranges per BERT's ``_is_chinese_char``."""
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0x2B820 <= cp <= 0x2CEAF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
    )
