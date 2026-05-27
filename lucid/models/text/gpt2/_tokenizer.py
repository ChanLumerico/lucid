"""GPT-2 family tokenizer wrappers — Byte-Level BPE.

GPT-2 (Radford et al., 2019) standardised the byte-level BPE that
became the canonical tokenization scheme for the GPT-2 / GPT-3 /
RoBERTa / BART / CodeGen line of models.  These wrappers subclass
:class:`~lucid.utils.tokenizer.ByteLevelBPETokenizer` /
:class:`~lucid.utils.tokenizer.ByteLevelBPETokenizerFast` with GPT-2's
two key defaults:

* ``<|endoftext|>`` is the canonical document-boundary token.  GPT-2
  uses it for **bos**, **eos**, *and* **unk** at once (the same id
  in all three slots).
* ``add_prefix_space=False`` — matches the published GPT-2 vocab.
  Set ``True`` to mimic RoBERTa's space-prepended convention.

Loads any HF GPT-2 ``vocab.json`` + ``merges.txt`` pair or unified
``tokenizer.json`` without modification.

Two flavours, bit-identical encode output:

* :class:`GPT2Tokenizer` — pure-Python reference.
* :class:`GPT2TokenizerFast` — C++-backed via
  :class:`lucid._C.engine.utils.tokenizer.BPE`.
"""

from lucid.utils.tokenizer._base import SpecialTokens
from lucid.utils.tokenizer._byte_bpe import (
    ByteLevelBPETokenizer,
    ByteLevelBPETokenizerFast,
)
from lucid.utils.tokenizer._normalizers import Normalizer


# Canonical GPT-2 end-of-text marker.  Shared across bos / eos / unk
# in :class:`GPT2Tokenizer` defaults.
GPT2_EOS = "<|endoftext|>"


def _gpt2_special_tokens() -> SpecialTokens:
    """Return GPT-2's special-token registry.

    GPT-2 ships exactly one canonical special token,
    ``<|endoftext|>``, and uses it for **bos**, **eos**, and **unk**
    at once.  Downstream code that distinguishes the three roles can
    still do so by name — the underlying id is the same.

    Returns
    -------
    SpecialTokens
        Registry with ``bos=eos=unk=<|endoftext|>``.
    """
    return SpecialTokens(bos=GPT2_EOS, eos=GPT2_EOS, unk=GPT2_EOS)


class GPT2Tokenizer(ByteLevelBPETokenizer):
    r"""GPT-2 tokenizer — pure-Python reference.

    A thin convenience subclass of
    :class:`~lucid.utils.tokenizer.ByteLevelBPETokenizer` with
    ``<|endoftext|>`` pre-registered as the bos / eos / unk marker.

    Parameters
    ----------
    vocab : dict[str, int]
        Token-string → id map (byte-mapped Unicode keys).
    merges : list[tuple[str, str]]
        Ordered BPE merges; index = rank.
    add_prefix_space : bool, default ``False``
        Prepend a space so the first word receives the ``Ġ`` marker.
        Off in GPT-2 (matches the published vocab); on in RoBERTa.
    normalizer : Normalizer, optional
        Default :class:`~lucid.utils.tokenizer.normalizers.NFC`.
    special_tokens : SpecialTokens, optional
        Override the default GPT-2 registry (single ``<|endoftext|>``
        shared across bos / eos / unk).

    See Also
    --------
    GPT2TokenizerFast : C++-backed variant with identical output.
    lucid.models.text.gpt.GPTTokenizer : Predecessor without an
        end-of-text marker.
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
            special_tokens=(
                special_tokens if special_tokens is not None else _gpt2_special_tokens()
            ),
        )


class GPT2TokenizerFast(ByteLevelBPETokenizerFast):
    """GPT-2 tokenizer — C++-backed.

    Bit-identical to :class:`GPT2Tokenizer`.  The BPE merge loop runs
    in C++ via :class:`lucid._C.engine.utils.tokenizer.BPE` for
    production throughput.

    Constructor parameters mirror :class:`GPT2Tokenizer` — see that
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
            special_tokens=(
                special_tokens if special_tokens is not None else _gpt2_special_tokens()
            ),
        )
