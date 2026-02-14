"""
Lucid BERT tokenizer C++ bindings
"""

from __future__ import annotations
import collections.abc
import typing

__all__: list[str] = list()

class _C_BERTEncodePairResult:
    def __init__(self) -> None: ...
    @property
    def attention_mask(self) -> list[int]: ...
    @attention_mask.setter
    def attention_mask(
        self, arg0: collections.abc.Sequence[typing.SupportsInt]
    ) -> None: ...
    @property
    def input_ids(self) -> list[int]: ...
    @input_ids.setter
    def input_ids(self, arg0: collections.abc.Sequence[typing.SupportsInt]) -> None: ...
    @property
    def token_type_ids(self) -> list[int]: ...
    @token_type_ids.setter
    def token_type_ids(
        self, arg0: collections.abc.Sequence[typing.SupportsInt]
    ) -> None: ...

class _C_BERTTokenizer:
    def __init__(
        self,
        vocab: collections.abc.Mapping[str, typing.SupportsInt] | None = None,
        vocab_file: os.PathLike | str | bytes | None = None,
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        wordpieces_prefix: str = "##",
        max_input_chars_per_word: typing.SupportsInt = 100,
        lowercase: bool = True,
        clean_text: bool = True,
    ) -> None: ...
    def build_inputs_with_special_tokens(
        self, tokens: collections.abc.Sequence[str]
    ) -> list[str]: ...
    def build_inputs_with_special_tokens_pair(
        self,
        tokens_a: collections.abc.Sequence[str],
        tokens_b: collections.abc.Sequence[str],
    ) -> list[str]: ...
    def cls_token(self) -> str: ...
    def convert_id_to_token(self, id: typing.SupportsInt) -> str: ...
    def convert_ids_to_tokens(
        self, ids: collections.abc.Sequence[typing.SupportsInt]
    ) -> list[str]: ...
    def convert_token_to_id(self, token: str) -> int: ...
    def convert_tokens_to_ids(
        self, tokens: collections.abc.Sequence[str]
    ) -> list[int]: ...
    def convert_tokens_to_string(
        self, tokens: collections.abc.Sequence[str]
    ) -> str: ...
    def create_token_type_ids_from_sequences(
        self,
        tokens_a: collections.abc.Sequence[str],
        tokens_b: collections.abc.Sequence[str] | None = None,
    ) -> list[int]: ...
    def encode_plus(
        self, text_a: str, text_b: str | None = None
    ) -> _C_BERTEncodePairResult: ...
    def mask_token(self) -> str: ...
    def sep_token(self) -> str: ...
    def tokenize(self, text: str) -> list[str]: ...
    def vocab_size(self) -> int: ...
