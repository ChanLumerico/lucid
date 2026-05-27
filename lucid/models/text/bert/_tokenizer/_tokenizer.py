"""BERT tokenizer wrappers — WordPiece with BERT's special-token set."""

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


def _bert_special_tokens() -> SpecialTokens:
    """BERT's canonical special-token registry."""
    return SpecialTokens(
        unk="[UNK]",
        pad="[PAD]",
        cls="[CLS]",
        sep="[SEP]",
        mask="[MASK]",
    )


class BertTokenizer(WordPieceTokenizer):
    """BERT tokenizer (pure-Python).  WordPiece with ``[UNK]/[CLS]/
    [SEP]/[PAD]/[MASK]`` registered out of the box."""

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
                special_tokens if special_tokens is not None else _bert_special_tokens()
            ),
        )


class BertTokenizerFast(WordPieceTokenizerFast):
    """BERT tokenizer (C++-backed).  See :class:`BertTokenizer`."""

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
                special_tokens if special_tokens is not None else _bert_special_tokens()
            ),
        )
