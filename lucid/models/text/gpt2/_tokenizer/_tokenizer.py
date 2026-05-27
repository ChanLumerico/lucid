"""GPT-2 tokenizer wrappers — ByteLevelBPE with ``<|endoftext|>``."""

from lucid.utils.tokenizer._base import SpecialTokens
from lucid.utils.tokenizer._byte_bpe import (
    ByteLevelBPETokenizer,
    ByteLevelBPETokenizerFast,
)
from lucid.utils.tokenizer._normalizers import Normalizer


GPT2_EOS = "<|endoftext|>"


def _gpt2_special_tokens() -> SpecialTokens:
    """GPT-2's single ``<|endoftext|>`` doubles as bos / eos / unk."""
    return SpecialTokens(bos=GPT2_EOS, eos=GPT2_EOS, unk=GPT2_EOS)


class GPT2Tokenizer(ByteLevelBPETokenizer):
    """GPT-2 tokenizer (pure-Python).  ByteLevelBPE with
    ``<|endoftext|>`` as the single bos / eos / unk marker."""

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
    """GPT-2 tokenizer (C++-backed).  See :class:`GPT2Tokenizer`."""

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
