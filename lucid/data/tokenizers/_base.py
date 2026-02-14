import json

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Iterable, TypeVar

import lucid
from lucid.types import _DeviceType


__all__ = ["SpecialTokens", "Tokenizer"]


class SpecialTokens(Enum):
    UNK: str = "[UNK]"
    PAD: str = "[PAD]"
    BOS: str = "[BOS]"
    EOS: str = "[EOS]"


_SpecialTokensOrStr = SpecialTokens | str

T = TypeVar("T", bound="Tokenizer")


class Tokenizer(ABC):
    def __init__(
        self,
        unk_token: _SpecialTokensOrStr = SpecialTokens.UNK,
        pad_token: _SpecialTokensOrStr = SpecialTokens.PAD,
        bos_token: _SpecialTokensOrStr | None = SpecialTokens.BOS,
        eos_token: _SpecialTokensOrStr | None = SpecialTokens.EOS,
    ) -> None:
        super().__init__()
        self.unk_token = self._resolve_token(unk_token)
        self.pad_token = self._resolve_token(pad_token)
        self.bos_token = self._resolve_token(bos_token)
        self.eos_token = self._resolve_token(eos_token)

    @staticmethod
    def _resolve_token(token: _SpecialTokensOrStr | None) -> str | None:
        if isinstance(token, SpecialTokens):
            return token.value
        return token

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @abstractmethod
    def tokenize(self, text: str) -> list[str]: ...

    @abstractmethod
    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]: ...

    @abstractmethod
    def convert_ids_to_tokens(self, ids: int | list[int]) -> str | list[str]: ...

    @abstractmethod
    def convert_tokens_to_string(self, tokens: list[str]) -> str: ...

    @property
    def all_special_tokens(self) -> list[str]:
        tokens = [self.unk_token, self.pad_token, self.bos_token, self.eos_token]
        return [t for t in tokens if t is not None]

    @property
    def all_special_ids(self) -> list[int]:
        ids = self.convert_tokens_to_ids(self.all_special_tokens)
        if not isinstance(ids, list):
            raise TypeError("convert_tokens_to_ids(list[str]) must return 'list[int]'.")
        return ids

    def build_inputs_with_special_tokens(self, tokens: list[str]) -> list[str]:
        out = tokens
        if self.bos_token is not None:
            out = [self.bos_token] + out
        if self.eos_token is not None:
            out = out + [self.eos_token]
        return out

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_tensor: bool = False,
        device: _DeviceType = "cpu",
    ) -> list[int] | lucid.LongTensor:
        tokens = self.tokenize(text)
        if add_special_tokens:
            tokens = self.build_inputs_with_special_tokens(tokens)

        ids = self.convert_tokens_to_ids(tokens)
        if not isinstance(ids, list):
            raise TypeError("convert_tokens_to_ids(list[str]) must return 'list[int]'.")

        if return_tensor:
            return lucid.LongTensor(ids, device=device)
        return ids

    def decode(
        self, ids: list[int] | lucid.LongTensor, skip_special_tokens: bool = True
    ) -> str:
        if isinstance(ids, lucid.Tensor):
            if ids.dtype.base_dtype is not int:
                raise TypeError(
                    "'ids' must be a 'lucid.LongTensor'. Try 'Tensor.long()'"
                )
            if ids.ndim != 1:
                raise ValueError(f"'ids' must be 1-D tensor, got {ids.ndim}-D tensor")

            ids = ids.tolist()

        tokens = self.convert_ids_to_tokens(ids)
        if not isinstance(tokens, list):
            raise TypeError("convert_ids_to_tokens(list[int]) must return 'list[str]'.")

        if skip_special_tokens:
            special = set(self.all_special_tokens)
            tokens = [t for t in tokens if t not in special]

        return self.convert_tokens_to_string(tokens)

    def batch_encode(
        self,
        texts: Iterable[str],
        add_special_tokens: bool = True,
        padding: bool = False,
        max_length: int | None = None,
        truncation: bool = False,
        return_tensor: bool = False,
        device: _DeviceType = "cpu",
    ) -> list[list[int]] | lucid.LongTensor:
        batch_ids = [
            self.encode(t, add_special_tokens=add_special_tokens) for t in texts
        ]
        if truncation and max_length is not None:
            batch_ids: list[list[int]] = [ids[:max_length] for ids in batch_ids]

        if return_tensor:
            if not padding:
                lens = {len(ids) for ids in batch_ids}
                if len(lens) != 1:
                    raise ValueError(
                        "Cannot return 2D tensor for ragged batch. "
                        "Set 'padding=True' (or pad manually)."
                    )

            if padding:
                if self.pad_token is None:
                    raise ValueError("'pad_token' is required when 'padding=True'")

                pad_id = self.convert_ids_to_tokens(self.pad_token)
                if isinstance(pad_id, list):
                    raise TypeError("convert_tokens_to_ids(str) must return int.")

                target_len = (
                    max_length
                    if max_length is not None
                    else max(len(ids) for ids in batch_ids)
                )
                padded: list[list[int]] = []
                for ids in batch_ids:
                    cur = ids[:target_len]
                    cur = cur + [pad_id] * max(0, target_len - len(cur))
                    padded.append(cur)

                batch_ids = padded

            return lucid.LongTensor(batch_ids, device=device)

        return batch_ids

    def batch_decode(
        self,
        batch_ids: Iterable[list[int]] | lucid.LongTensor,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        return [
            self.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in batch_ids
        ]

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_tensor: bool = False,
        device: _DeviceType = "cpu",
    ) -> list[int] | lucid.LongTensor:
        return self.encode(text, add_special_tokens, return_tensor, device)

    def fit(self, texts: Iterable[str], **kwargs) -> Tokenizer:
        raise NotImplementedError(
            f"{type(self).__name__} does not support vocabulary training via 'fit'."
        )

    @classmethod
    def train_from_iterator(cls, texts: Iterable[str], **kwargs) -> Tokenizer:
        raise NotImplementedError(
            f"{cls.__name__} does not support vocabulary training via "
            "'train_from_iterator'."
        )

    @staticmethod
    def _save_json(data: dict, path: Path | str) -> str:
        out_path = Path(path)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return str(out_path)

    @staticmethod
    def _load_json(path: Path | str) -> dict[str, object]:
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise TypeError(f"Expected JSON object at '{path}'.")
        return data

    def _save_tokenizer_config(self, save_directory: Path | str, config: dict) -> str:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        return self._save_json(config, save_path / "tokenizer_config.json")

    @classmethod
    def _load_tokenizer_config(
        cls,
        pretrained_model_name_or_path: Path | str,
        allowed_keys: set[str] | None = None,
    ) -> dict[str, object]:
        path = Path(pretrained_model_name_or_path)
        config_file = (
            path / "tokenizer_config.json"
            if path.is_dir()
            else path.with_name("tokenizer_config.json")
        )
        if not config_file.exists():
            return {}
        config = cls._load_json(config_file)
        if allowed_keys is None:
            return config
        return {k: v for k, v in config.items() if k in allowed_keys}

    def _build_id_to_token_map(
        self,
        vocab: dict[str, int],
        unk_token: str = SpecialTokens.UNK.value,
        verbose: bool = False,
    ) -> list[str]:
        from tqdm.auto import tqdm

        max_id = max(vocab.values(), default=-1)
        id_to_token = ["" for _ in range(max_id + 1)]

        items = vocab.items()
        if verbose:
            items = tqdm(
                items,
                total=len(vocab),
                desc=f"{type(self).__name__} id->token map",
                dynamic_ncols=True,
                unit="tok",
            )
        for token, idx in items:
            if idx < 0:
                raise ValueError("Vocabulary indices must be non-negative integers.")
            if idx >= len(id_to_token):
                id_to_token.extend([""] * (idx - len(id_to_token) + 1))
            if id_to_token[idx]:
                raise ValueError(f"Duplicate token index detected: {idx}")
            id_to_token[idx] = token

        for i, token in enumerate(id_to_token):
            if not token:
                id_to_token[i] = unk_token
        return id_to_token

    @abstractmethod
    def save_pretrained(self, save_directory: Path | str) -> list[str]: ...

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, **kwrags
    ) -> Tokenizer: ...
