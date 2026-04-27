from __future__ import annotations
import collections.abc
import typing

__all__: list[str] = list()

class _C_BPETokenizer:
    def __init__(
        self,
        vocab: collections.abc.Mapping[str, typing.SupportsInt] | None = None,
        merges: collections.abc.Sequence[tuple[str, str]] | None = None,
        vocab_file: os.PathLike | str | bytes | None = None,
        merges_file: os.PathLike | str | bytes | None = None,
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        bos_token: str | None = None,
        eos_token: str | None = None,
        lowercase: bool = True,
        clean_text: bool = True,
        end_of_word_suffix: str = "</w>",
    ) -> None: ...
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
    def encode_ids(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def fit(
        self,
        texts: collections.abc.Sequence[str],
        vocab_size: typing.SupportsInt,
        min_frequency: typing.SupportsInt = 2,
    ) -> _C_BPETokenizer: ...
    def get_merges(self) -> list[tuple[str, str]]: ...
    def get_vocab(self) -> dict[str, int]: ...
    def tokenize(self, text: str) -> list[str]: ...
    def vocab_size(self) -> int: ...

class _C_ByteBPETokenizer(_C_BPETokenizer):
    def __init__(
        self,
        vocab: collections.abc.Mapping[str, typing.SupportsInt] | None = None,
        merges: collections.abc.Sequence[tuple[str, str]] | None = None,
        vocab_file: os.PathLike | str | bytes | None = None,
        merges_file: os.PathLike | str | bytes | None = None,
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        bos_token: str | None = None,
        eos_token: str | None = None,
        lowercase: bool = False,
        clean_text: bool = True,
        add_prefix_space: bool = False,
        end_of_word_suffix: str = "",
    ) -> None: ...
    def add_prefix_space(self) -> bool: ...
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
    def encode_ids(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def fit(
        self,
        texts: collections.abc.Sequence[str],
        vocab_size: typing.SupportsInt,
        min_frequency: typing.SupportsInt = 2,
    ) -> _C_ByteBPETokenizer: ...
    def get_merges(self: _C_BPETokenizer) -> list[tuple[str, str]]: ...
    def get_vocab(self: _C_BPETokenizer) -> dict[str, int]: ...
    def tokenize(self, text: str) -> list[str]: ...
    def vocab_size(self) -> int: ...

class _C_WordPieceTokenizer:
    def __init__(
        self,
        vocab: collections.abc.Mapping[str, typing.SupportsInt] | None = None,
        vocab_file: os.PathLike | str | bytes | None = None,
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        bos_token: str | None = None,
        eos_token: str | None = None,
        wordpieces_prefix: str = "##",
        max_input_chars_per_word: typing.SupportsInt = 100,
        lowercase: bool = True,
        clean_text: bool = True,
    ) -> None: ...
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
    def fit(
        self,
        texts: collections.abc.Sequence[str],
        vocab_size: typing.SupportsInt,
        min_frequency: typing.SupportsInt = 2,
    ) -> _C_WordPieceTokenizer: ...
    def tokenize(self, text: str) -> list[str]: ...
    def vocab_size(self) -> int: ...
