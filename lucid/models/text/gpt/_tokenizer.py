from pathlib import Path
from typing import Any

from lucid.data.tokenizers import SpecialTokens
from lucid.data.tokenizers.bpe import BPETokenizerFast

__all__ = ["GPTTokenizerFast"]

_EOT: str = "<|endoftext|>"


class GPTTokenizerFast(BPETokenizerFast):
    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        merges: list[tuple[str, str]] | None = None,
        vocab_file: Path | str | None = None,
        merges_file: Path | str | None = None,
        unk_token: SpecialTokens | str = SpecialTokens.UNK,
        pad_token: SpecialTokens | str = SpecialTokens.PAD,
        eot_token: str = _EOT,
        lowercase: bool = True,
        clean_text: bool = True,
        end_of_word_suffix: str = "</w>",
    ) -> None:
        self.eot_token = eot_token
        super().__init__(
            vocab=vocab,
            merges=merges,
            vocab_file=vocab_file,
            merges_file=merges_file,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=eot_token,
            eos_token=eot_token,
            lowercase=lowercase,
            clean_text=clean_text,
            end_of_word_suffix=end_of_word_suffix,
        )

    @property
    def eot_token_id(self) -> int:
        ids = self.convert_tokens_to_ids(self.eot_token)
        assert isinstance(ids, int)
        return ids

    def build_inputs_with_special_tokens(self, tokens: list[str]) -> list[str]:
        return tokens

    def encode_plus(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int]]:
        tokens = self.tokenize(text)
        if add_special_tokens:
            tokens = tokens + [self.eot_token]

        input_ids = self.convert_tokens_to_ids(tokens)
        if not isinstance(input_ids, list):
            input_ids = [input_ids]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }

    def save_pretrained(self, save_directory: Path | str) -> list[str]:
        paths = super().save_pretrained(save_directory)
        save_path = Path(save_directory)

        cfg = self._load_json(save_path / "tokenizer_config.json")
        cfg["eot_token"] = self.eot_token
        self._save_json(cfg, save_path / "tokenizer_config.json")
        return paths

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        **kwargs: Any,
    ) -> GPTTokenizerFast:
        path = Path(pretrained_model_name_or_path)
        base_dir = path if path.is_dir() else path.parent
        vocab_file = (path / "vocab.json") if path.is_dir() else path
        merges_file = base_dir / "merges.txt"

        if not vocab_file.exists():
            raise FileNotFoundError(f"Cannot find vocabulary file: {vocab_file}")
        if not merges_file.exists():
            raise FileNotFoundError(f"Cannot find merges file: {merges_file}")

        allowed = {
            "unk_token",
            "pad_token",
            "eot_token",
            "lowercase",
            "clean_text",
            "end_of_word_suffix",
        }
        init_kwargs = cls._load_tokenizer_config(base_dir, allowed_keys=allowed)
        init_kwargs.update(kwargs)
        return cls(vocab_file=vocab_file, merges_file=merges_file, **init_kwargs)
