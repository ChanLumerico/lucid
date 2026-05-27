"""GPT tokenizer wrappers — ByteLevelBPE with GPT-1 defaults."""

from lucid.utils.tokenizer._base import SpecialTokens
from lucid.utils.tokenizer._byte_bpe import (
    ByteLevelBPETokenizer,
    ByteLevelBPETokenizerFast,
)
from lucid.utils.tokenizer._normalizers import Normalizer


class GPTTokenizer(ByteLevelBPETokenizer):
    """GPT-1 tokenizer (pure-Python).  ByteLevelBPE with
    ``add_prefix_space=False`` (GPT-1 convention)."""

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
    """GPT-1 tokenizer (C++-backed).  See :class:`GPTTokenizer`."""

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
