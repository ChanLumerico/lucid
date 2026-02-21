from pathlib import Path

from lucid.data.tokenizers import SpecialTokens
from lucid.models.text.bert import BERTTokenizerFast


__all__ = ["RoFormerTokenizerFast"]


class RoFormerTokenizerFast(BERTTokenizerFast):
    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        vocab_file: Path | str | None = None,
        unk_token: SpecialTokens | str = SpecialTokens.UNK,
        pad_token: SpecialTokens | str = SpecialTokens.PAD,
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        sep_token: str = "[SEP]",
        lowercase: bool = True,
        wordpieces_prefix: str = "##",
        max_input_chars_per_word: int = 100,
        clean_text: bool = True,
    ) -> None:
        super().__init__(
            vocab,
            vocab_file,
            unk_token,
            pad_token,
            cls_token,
            mask_token,
            sep_token,
            lowercase,
            wordpieces_prefix,
            max_input_chars_per_word,
            clean_text,
        )
