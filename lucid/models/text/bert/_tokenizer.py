from pathlib import Path
from typing import Any

from lucid.data.tokenizers import SpecialTokens, Tokenizer

from lucid._backend._C.tokenizers.bert.core import _C_BERTTokenizer


__all__ = ["BERTTokenizerFast"]


class BERTTokenizerFast(Tokenizer):
    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        vocab_file: Path | str | None = None,
        unk_token: SpecialTokens | str = SpecialTokens.UNK,
        pad_token: SpecialTokens | str = SpecialTokens.PAD,
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        lowercase: bool = True,
        wordpieces_prefix: str = "##",
        max_input_chars_per_word: int = 100,
        clean_text: bool = True,
    ) -> None:
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=cls_token,
            eos_token="[SEP]",
        )

        self.cls_token = cls_token
        self.sep_token = "[SEP]"
        self.mask_token = mask_token
        self.lowercase = lowercase
        self.wordpieces_prefix = wordpieces_prefix
        self.max_input_chars_per_word = max_input_chars_per_word
        self.clean_text = clean_text

        self._backend = _C_BERTTokenizer(
            vocab=vocab,
            vocab_file=Path(vocab_file) if vocab_file is not None else None,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            cls_token=self.cls_token,
            mask_token=self.mask_token,
            wordpieces_prefix=self.wordpieces_prefix,
            max_input_chars_per_word=self.max_input_chars_per_word,
            lowercase=self.lowercase,
            clean_text=self.clean_text,
        )

    @property
    def vocab_size(self) -> int:
        size = getattr(self._backend, "vocab_size")
        return int(size() if callable(size) else size)

    def tokenize(self, text: str) -> list[str]:
        return list(self._backend.tokenize(text))

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return int(self._backend.convert_token_to_id(tokens))
        return [int(x) for x in self._backend.convert_tokens_to_ids(tokens)]

    def convert_ids_to_tokens(self, ids: int | list[int]) -> str | list[str]:
        if isinstance(ids, int):
            return str(self._backend.convert_id_to_token(ids))
        return [str(x) for x in self._backend.convert_ids_to_tokens(ids)]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return str(self._backend.convert_tokens_to_string(tokens))

    def build_inputs_with_special_tokens(self, tokens: list[str]) -> list[str]:
        return list(self._backend.build_inputs_with_special_tokens(tokens))

    def encode_plus(
        self,
        text_a: str,
        text_b: str | None = None,
    ) -> dict[str, list[int]]:
        out = self._backend.encode_plus(text_a, text_b)
        return {
            "input_ids": [int(x) for x in out.input_ids],
            "token_type_ids": [int(x) for x in out.token_type_ids],
            "attention_mask": [int(x) for x in out.attention_mask],
        }

    def save_pretrained(self, save_directory: Path | str) -> list[str]:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        vocab_path = save_path / "vocab.txt"
        with vocab_path.open("w", encoding="utf-8") as f:
            for idx in range(self.vocab_size):
                f.write(self.convert_ids_to_tokens(idx) + "\n")

        config = {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "cls_token": self.cls_token,
            "mask_token": self.mask_token,
            "lowercase": self.lowercase,
            "wordpieces_prefix": self.wordpieces_prefix,
            "max_input_chars_per_word": self.max_input_chars_per_word,
            "clean_text": self.clean_text,
        }
        config_path = self._save_tokenizer_config(save_path, config)
        return [str(vocab_path), str(config_path)]

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, **kwrags: Any
    ) -> BERTTokenizerFast:
        path = Path(pretrained_model_name_or_path)
        vocab_file = path / "vocab.txt" if path.is_dir() else path
        if not vocab_file.exists():
            raise FileNotFoundError(f"Cannot find vocabulary file: {vocab_file}")

        allowed = {
            "unk_token",
            "pad_token",
            "cls_token",
            "mask_token",
            "lowercase",
            "wordpieces_prefix",
            "max_input_chars_per_word",
            "clean_text",
        }
        init_kwargs = cls._load_tokenizer_config(path, allowed_keys=allowed)
        init_kwargs.update(kwrags)
        return cls(vocab_file=vocab_file, **init_kwargs)
