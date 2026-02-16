from pathlib import Path
from typing import Any

import lucid

from lucid.data.tokenizers import SpecialTokens, Tokenizer
from lucid.types import _DeviceType

from lucid._backend._C.tokenizers.core import _C_WordPieceTokenizer


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
        sep_token: str = "[SEP]",
        lowercase: bool = True,
        wordpieces_prefix: str = "##",
        max_input_chars_per_word: int = 100,
        clean_text: bool = True,
    ) -> None:
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=cls_token,
            eos_token=sep_token,
        )

        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token
        self.lowercase = lowercase
        self.wordpieces_prefix = wordpieces_prefix
        self.max_input_chars_per_word = max_input_chars_per_word
        self.clean_text = clean_text

        self._backend = _C_WordPieceTokenizer(
            vocab=vocab,
            vocab_file=Path(vocab_file) if vocab_file is not None else None,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            bos_token=self.cls_token,
            eos_token=self.sep_token,
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
        return [self.cls_token, *tokens, self.sep_token]

    def encode_plus(
        self,
        text_a: str,
        text_b: str | None = None,
    ) -> dict[str, list[int]]:
        tokens_a = self.tokenize(text_a)
        tokens_b = self.tokenize(text_b) if text_b is not None else []

        all_tokens = [self.cls_token, *tokens_a, self.sep_token]
        token_type_ids = [0] * len(all_tokens)
        if text_b is not None:
            b_tokens = [*tokens_b, self.sep_token]
            all_tokens.extend(b_tokens)
            token_type_ids.extend([1] * len(b_tokens))

        input_ids = self.convert_tokens_to_ids(all_tokens)
        if not isinstance(input_ids, list):
            raise TypeError("Expected list[int] from token-to-id conversion.")

        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }

    def encode_pretraining_inputs(
        self,
        text_a: str,
        text_b: str | None = None,
        return_tensor: bool = False,
        device: _DeviceType = "cpu",
    ) -> dict[str, list[int] | lucid.LongTensor]:
        enc = self.encode_plus(text_a, text_b)

        special_ids = set(self.all_special_ids)
        cls_id = self.convert_tokens_to_ids(self.cls_token)
        sep_id = self.convert_tokens_to_ids(self.sep_token)
        if isinstance(cls_id, int):
            special_ids.add(cls_id)
        if isinstance(sep_id, int):
            special_ids.add(sep_id)

        special_tokens_mask = [
            1 if tid in special_ids else 0 for tid in enc["input_ids"]
        ]
        if return_tensor:
            return {
                "input_ids": lucid.LongTensor([enc["input_ids"]], device=device),
                "token_type_ids": lucid.LongTensor(
                    [enc["token_type_ids"]], device=device
                ),
                "attention_mask": lucid.LongTensor(
                    [enc["attention_mask"]], device=device
                ),
                "special_tokens_mask": lucid.LongTensor(
                    [special_tokens_mask], device=device
                ),
            }

        return {
            "input_ids": enc["input_ids"],
            "token_type_ids": enc["token_type_ids"],
            "attention_mask": enc["attention_mask"],
            "special_tokens_mask": special_tokens_mask,
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
            "sep_token": self.sep_token,
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
        base_path = path if path.is_dir() else path.parent
        vocab_file = path / "vocab.txt" if path.is_dir() else path
        if not vocab_file.exists():
            raise FileNotFoundError(f"Cannot find vocabulary file: {vocab_file}")

        allowed = {
            "unk_token",
            "pad_token",
            "cls_token",
            "sep_token",
            "mask_token",
            "lowercase",
            "wordpieces_prefix",
            "max_input_chars_per_word",
            "clean_text",
        }
        init_kwargs = cls._load_tokenizer_config(base_path, allowed_keys=allowed)
        init_kwargs.update(kwrags)
        return cls(vocab_file=vocab_file, **init_kwargs)
