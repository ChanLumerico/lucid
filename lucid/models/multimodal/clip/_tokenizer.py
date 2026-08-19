"""CLIP tokenizer — byte-level BPE with an end-of-word marker.

CLIP's scheme is a hybrid of the two already in the tree, which is why
it is neither of them.  It is **byte-level** like GPT-2's, so any input
round-trips without an unknown token; and it seeds each word as
``list(w[:-1]) + [w[-1] + "</w>"]`` like GPT-1's, so the merge table can
tell ``"in"`` inside a word from ``"in"`` ending one.  GPT-2 marks the
boundary with a leading ``Ġ`` instead, and the two markers produce
incompatible merge tables — a vocabulary built for one cannot be read by
the other.

The vocabulary's size follows from that construction, and settles a
discrepancy worth writing down.  The paper says "49,152 vocab size"; the
released models have 49408 rows.  Both are correct about different
things:

.. code-block:: text

    256    byte symbols
  + 256    byte symbols suffixed with </w>
  + 48894  merges
  + 2      <|startoftext|>, <|endoftext|>
  ------
    49408

and 49152 is the target the merge list is *sliced* to
(``merges[1:49152-256-2+1]``) before the byte symbols are counted.

**One preprocessing step is deliberately absent.**  The reference calls
``ftfy.fix_text`` to repair mojibake before anything else.  Lucid's
compute and utility paths take no third-party dependency, so it is not
called here.  For text that is already valid UTF-8 — which is what a
caller passing Python ``str`` has — the step is the identity, so the
difference is confined to inputs that were already corrupt.  It is
recorded rather than hidden because it means this tokenizer is not
bit-identical to the reference on adversarial input.
"""

import html
import re
from typing import override

from lucid.utils.tokenizer._base import SpecialTokens
from lucid.utils.tokenizer._bpe import BPETokenizer
from lucid.utils.tokenizer._normalizers import Normalizer
from lucid.utils.tokenizer._pre_tokenizers import ByteLevel, Chunk, PreTokenizer

__all__ = ["CLIPTokenizer", "CLIP_SOS", "CLIP_EOS", "END_OF_WORD"]

CLIP_SOS = "<|startoftext|>"
CLIP_EOS = "<|endoftext|>"
END_OF_WORD = "</w>"

# The reference pattern, with its two Unicode property escapes rewritten
# for the standard library.  ``\p{L}`` becomes ``[^\W\d_]`` — a word
# character that is neither a digit nor an underscore is a letter — and
# ``\p{N}`` becomes ``\d``.  The alternation order is load-bearing: the
# contraction forms must be tried before the letter run, or ``'s`` is
# consumed as punctuation followed by a letter.
_PATTERN = re.compile(
    r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d"
    r"|[^\W\d_]+|\d|[^\s\w]+",
    re.UNICODE,
)


def _whitespace_clean(text: str) -> str:
    """Collapse every run of whitespace to one space, and strip."""
    return re.sub(r"\s+", " ", text).strip()


def _basic_clean(text: str) -> str:
    """Unescape HTML entities twice, then strip.

    Twice because the reference does: web-scraped captions frequently
    carry doubly-escaped entities (``&amp;amp;``), and a single pass
    leaves ``&amp;`` in the text where the vocabulary expects ``&``.
    """
    return html.unescape(html.unescape(text)).strip()


class _CLIPPreTokenizer(PreTokenizer):
    """Split on CLIP's pattern, then byte-map each piece.

    Notes
    -----
    Two stages that have to happen in this order.  The pattern is applied
    to *text*, so it can use letter and digit classes; the byte mapping
    is applied to each piece afterwards, so the symbols the merge table
    sees are the byte-mapped ones.  Byte-mapping first would destroy the
    letter classes the pattern depends on.
    """

    @override
    def pre_tokenize(self, text: str) -> list[Chunk]:
        """Return the byte-mapped chunks of ``text`` with their offsets.

        Parameters
        ----------
        text : str
            Already normalised and lowercased.

        Returns
        -------
        list of (str, (int, int))
            One entry per matched piece.  The offsets index ``text``,
            not the byte-mapped surface — a byte-mapped chunk can be
            longer than the characters it came from, so reporting its
            own length would misalign ``return_offsets_mapping`` for
            every non-ASCII caption.
        """
        out: list[Chunk] = []
        for match in _PATTERN.finditer(text):
            surface = ByteLevel.encode_bytes(match.group().encode("utf-8"))
            out.append((surface, (match.start(), match.end())))
        return out


class CLIPTokenizer(BPETokenizer):
    r"""CLIP's byte-level BPE with ``</w>`` word boundaries.

    Parameters
    ----------
    vocab : dict[str, int]
        Token-string → id map, over byte-mapped symbols.  Must contain
        :data:`CLIP_SOS` and :data:`CLIP_EOS`.
    merges : list of tuple of str
        Ordered BPE merges; index is the rank.
    normalizer : Normalizer, optional
        Applied before the pattern.  Defaults to none — CLIP's own
        cleaning (HTML unescaping, whitespace collapse, lowercasing)
        runs regardless and is not optional.
    special_tokens : SpecialTokens, optional
        Override the default registry, which sets ``bos`` to
        ``<|startoftext|>`` and ``eos`` / ``unk`` to ``<|endoftext|>``.

    Notes
    -----
    Reference: Radford et al., ICML 2021 (arXiv:2103.00020), §2.3.

    :meth:`tokenize` is the method the model wants: it frames each
    caption with the two sentinels and pads to a fixed width, which is
    what makes ``argmax`` over the ids find ``[EOS]``.  :meth:`encode`
    is inherited and does neither — it returns the bare BPE ids, which
    is what a caller building their own framing wants.

    Examples
    --------
    >>> from lucid.models.multimodal.clip import CLIPTokenizer
    >>> vocab = {"a": 0, "b</w>": 1, "ab</w>": 2,
    ...          "<|startoftext|>": 3, "<|endoftext|>": 4}
    >>> tok = CLIPTokenizer(vocab=vocab, merges=[("a", "b</w>")])
    >>> tok.encode("ab")
    [2]
    >>> tok.tokenize("ab", context_length=5)
    [3, 2, 4, 0, 0]
    """

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        *,
        normalizer: Normalizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        """Construct the tokenizer. See the class docstring for parameters."""
        for required in (CLIP_SOS, CLIP_EOS):
            if required not in vocab:
                raise ValueError(
                    f"vocab is missing {required!r} — the sentinels frame "
                    f"every caption and [EOS] must be present for the model "
                    f"to locate the text feature"
                )
        super().__init__(
            vocab=vocab,
            merges=merges,
            normalizer=normalizer,
            pre_tokenizer=_CLIPPreTokenizer(),
            special_tokens=special_tokens
            or SpecialTokens(bos=CLIP_SOS, eos=CLIP_EOS, unk=CLIP_EOS),
        )
        self._sos_id = vocab[CLIP_SOS]
        self._eos_id = vocab[CLIP_EOS]

    @override
    def _encode_chunk(self, chunk: str) -> list[int]:
        """Seed with the end-of-word marker, then merge as usual.

        Only the seeding differs from :class:`BPETokenizer`; the merge
        loop is inherited so the two cannot drift apart.
        """
        if not chunk:
            return []
        symbols = list(chunk[:-1]) + [chunk[-1] + END_OF_WORD]
        ids: list[int] = []
        unk_id = self.unk_token_id
        for symbol in symbols:
            tid = self._vocab.get(symbol)
            if tid is not None:
                ids.append(tid)
            elif unk_id is not None:
                ids.append(unk_id)
        return self._merge_ids(ids)

    @override
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Return the BPE ids of ``text``, unframed and unpadded.

        Parameters
        ----------
        text : str
            Raw caption.
        add_special_tokens : bool, default=False, keyword-only
            The base class defaults this to ``True``.  It is flipped here
            because :meth:`tokenize` is the framing path and adding the
            sentinels in both places gives a caption two ``[SOS]``,
            shifting every position the model reads by one.

        Returns
        -------
        list of int
            One id per merged symbol.

        Notes
        -----
        The sentinels are *not* added here, unlike the base class's
        default. Framing belongs to :meth:`tokenize`, which also pads —
        adding them in both places is how a caption ends up with two
        ``[SOS]`` and the model reads its feature one position early.
        """
        return super().encode(
            _whitespace_clean(_basic_clean(text)).lower(),
            add_special_tokens=add_special_tokens,
        )

    def tokenize(self, text: str, context_length: int = 77) -> list[int]:
        """Frame a caption with the sentinels and pad it to a fixed width.

        Parameters
        ----------
        text : str
            Raw caption.
        context_length : int, default=77
            Total width, counting both sentinels.

        Returns
        -------
        list of int
            ``[SOS] … [EOS]`` followed by zeros.

        Raises
        ------
        ValueError
            If the caption does not fit.  Truncating silently would drop
            the ``[EOS]`` the model locates its feature by, so a caption
            that is too long is an error rather than a shortened result.

        Notes
        -----
        Padding is ``0``, which is a real token id in a CLIP vocabulary.
        That is harmless *because* the feature is read at ``[EOS]`` and
        the text tower is causal, so nothing after the sentinel can reach
        the position the feature is taken from.

        Examples
        --------
        >>> from lucid.models.multimodal.clip import CLIPTokenizer
        >>> vocab = {"a": 0, "b</w>": 1, "ab</w>": 2,
        ...          "<|startoftext|>": 3, "<|endoftext|>": 4}
        >>> tok = CLIPTokenizer(vocab=vocab, merges=[("a", "b</w>")])
        >>> tok.tokenize("ab", context_length=4)
        [3, 2, 4, 0]
        """
        if context_length < 2:
            raise ValueError(
                f"context_length must leave room for both sentinels, got "
                f"{context_length}"
            )
        ids = [self._sos_id, *self.encode(text), self._eos_id]
        if len(ids) > context_length:
            raise ValueError(
                f"caption needs {len(ids)} tokens but context_length is "
                f"{context_length}; truncating would drop the [EOS] the "
                f"model reads its feature at, so shorten the caption or "
                f"raise the context"
            )
        return ids + [0] * (context_length - len(ids))

    @override
    def _decode_one(self, ids: list[int]) -> str:
        """Join surfaces, turning each ``</w>`` back into a space.

        The marker *is* the word boundary, so dropping it without
        substituting a space would run every word together.
        """
        out: list[str] = []
        for i in ids:
            token = self._id_to_token.get(i)
            if token is not None and token not in (CLIP_SOS, CLIP_EOS):
                out.append(token)
        return "".join(out).replace(END_OF_WORD, " ").strip()
