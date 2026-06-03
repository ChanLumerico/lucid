"""GPT-1 family tokenizer wrappers — Byte-Level BPE.

GPT-1 (Radford et al., 2018) shipped a BPE vocabulary with the
byte-mapped Unicode pre-tokenization scheme that GPT-2 / RoBERTa /
BART later popularised.  These wrappers subclass
:class:`~lucid.utils.tokenizer.ByteLevelBPETokenizer` /
:class:`~lucid.utils.tokenizer.ByteLevelBPETokenizerFast` with GPT-1's
defaults:

* ``add_prefix_space=False`` — the first word does not implicitly
  get a leading byte-mapped space.  Set ``True`` to mimic the
  RoBERTa convention if you need uniform space-prefixing.
* No mandatory end-of-text marker.  GPT-1 was trained without one;
  pass an explicit ``special_tokens`` if your downstream code expects
  bos/eos.

Loads any HF GPT-1 ``vocab.json`` + ``merges.txt`` pair (or unified
``tokenizer.json``) without modification.

Two flavours, bit-identical encode output:

* :class:`GPTTokenizer` — pure-Python reference.
* :class:`GPTTokenizerFast` — C++-backed; the BPE merge loop runs
  in the engine-side ``BPE`` binding.
"""

from lucid.utils.tokenizer._base import SpecialTokens
from lucid.utils.tokenizer._byte_bpe import (
    ByteLevelBPETokenizer,
    ByteLevelBPETokenizerFast,
)
from lucid.utils.tokenizer._normalizers import Normalizer


class GPTTokenizer(ByteLevelBPETokenizer):
    r"""GPT-1 tokenizer — pure-Python reference.

    A thin convenience subclass of
    :class:`~lucid.utils.tokenizer.ByteLevelBPETokenizer` with GPT-1's
    default ``add_prefix_space=False``.

    Parameters
    ----------
    vocab : dict[str, int]
        Token-string → id map.  Tokens are byte-mapped Unicode
        strings (e.g. ``"Ġworld"`` for `` world``), NOT raw bytes.
    merges : list[tuple[str, str]]
        Ordered BPE merges; index = rank, lower = applied earlier.
    add_prefix_space : bool, default ``False``
        Prepend a space to the input so the first word also receives
        the ``Ġ`` byte-mapped marker.  Off in GPT-1 (matches the
        published vocab); on in RoBERTa.
    normalizer : Normalizer, optional
        Default :class:`~lucid.utils.tokenizer.normalizers.NFC`.
    special_tokens : SpecialTokens, optional
        GPT-1 ships no canonical special tokens; pass your own if
        the downstream model expects bos / eos.

    See Also
    --------
    GPTTokenizerFast : C++-backed variant with identical output.
    lucid.models.text.gpt2.GPT2Tokenizer : Successor with
        ``<|endoftext|>`` as the default bos / eos / unk marker.
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
        super().__init__(
            vocab,
            merges,
            add_prefix_space=add_prefix_space,
            normalizer=normalizer,
            special_tokens=special_tokens,
        )


class GPTTokenizerFast(ByteLevelBPETokenizerFast):
    """GPT-1 tokenizer — C++-backed.

    Bit-identical to :class:`GPTTokenizer`.  The BPE merge loop runs
    in C++ via the engine-side ``BPE`` binding for production
    throughput.

    Constructor parameters mirror :class:`GPTTokenizer` — see that
    class for the full reference.
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
        super().__init__(
            vocab,
            merges,
            add_prefix_space=add_prefix_space,
            normalizer=normalizer,
            special_tokens=special_tokens,
        )
