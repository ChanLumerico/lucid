"""RoFormer tokenizer wrappers — identical to BERT WordPiece."""

from lucid.utils.tokenizer._base import SpecialTokens
from lucid.utils.tokenizer._normalizers import BertNormalizer, Normalizer
from lucid.utils.tokenizer._pre_tokenizers import (
    PreTokenizer,
    WhitespacePunctuationSplit,
)
from lucid.utils.tokenizer._wordpiece import (
    WordPieceTokenizer,
    WordPieceTokenizerFast,
)


def _roformer_special_tokens() -> SpecialTokens:
    return SpecialTokens(
        unk="[UNK]",
        pad="[PAD]",
        cls="[CLS]",
        sep="[SEP]",
        mask="[MASK]",
    )


class RoFormerTokenizer(WordPieceTokenizer):
    """RoFormer tokenizer (pure-Python). BERT-style WordPiece with the
    canonical ``[UNK]/[CLS]/[SEP]/[PAD]/[MASK]`` registry."""

    def __init__(
        self,
        vocab: dict[str, int],
        *,
        unk_token: str = "[UNK]",
        continuing_prefix: str = "##",
        max_chars_per_word: int = 100,
        do_lower_case: bool = True,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        super().__init__(
            vocab,
            unk_token=unk_token,
            continuing_prefix=continuing_prefix,
            max_chars_per_word=max_chars_per_word,
            normalizer=(
                normalizer
                if normalizer is not None
                else BertNormalizer(lowercase=do_lower_case)
            ),
            pre_tokenizer=(
                pre_tokenizer
                if pre_tokenizer is not None
                else WhitespacePunctuationSplit()
            ),
            special_tokens=(
                special_tokens
                if special_tokens is not None
                else _roformer_special_tokens()
            ),
        )


class RoFormerTokenizerFast(WordPieceTokenizerFast):
    """RoFormer tokenizer (C++-backed).  See :class:`RoFormerTokenizer`."""

    def __init__(
        self,
        vocab: dict[str, int],
        *,
        unk_token: str = "[UNK]",
        continuing_prefix: str = "##",
        max_chars_per_word: int = 100,
        do_lower_case: bool = True,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        super().__init__(
            vocab,
            unk_token=unk_token,
            continuing_prefix=continuing_prefix,
            max_chars_per_word=max_chars_per_word,
            normalizer=(
                normalizer
                if normalizer is not None
                else BertNormalizer(lowercase=do_lower_case)
            ),
            pre_tokenizer=(
                pre_tokenizer
                if pre_tokenizer is not None
                else WhitespacePunctuationSplit()
            ),
            special_tokens=(
                special_tokens
                if special_tokens is not None
                else _roformer_special_tokens()
            ),
        )
