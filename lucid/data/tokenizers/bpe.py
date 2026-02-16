import re

from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import lucid

from lucid.data.tokenizers import Tokenizer, SpecialTokens
from lucid.data.tokenizers.utils import basic_tokenize, clean_text
from lucid.types import _DeviceType

from lucid._backend._C.tokenizers.core import _C_BPETokenizer, _C_ByteBPETokenizer


__all__ = [
    "BPETokenizer",
    "BPETokenizerFast",
    "ByteBPETokenizer",
    "ByteBPETokenizerFast",
]


class BPETokenizer(Tokenizer):
    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        merges: list[tuple[str, str]] | None = None,
        vocab_file: Path | str | None = None,
        merges_file: Path | str | None = None,
        unk_token: SpecialTokens | str = SpecialTokens.UNK,
        pad_token: SpecialTokens | str = SpecialTokens.PAD,
        bos_token: SpecialTokens | str | None = None,
        eos_token: SpecialTokens | str | None = None,
        lowercase: bool = True,
        clean_text: bool = True,
        end_of_word_suffix: str = "</w>",
    ) -> None:
        if vocab is not None and vocab_file is not None:
            raise ValueError("Provide only one of 'vocab' or 'vocab_file'.")
        if merges is not None and merges_file is not None:
            raise ValueError("Provide only one of 'merges' or 'merges_file'.")

        super().__init__(unk_token, pad_token, bos_token, eos_token)
        if vocab_file is not None:
            vocab = self._load_vocab(vocab_file)
        if merges_file is not None:
            merges = self._load_merges(merges_file)

        self.lowercase = lowercase
        self.clean_text = clean_text
        self.end_of_word_suffix = end_of_word_suffix

        self.vocab: dict[str, int] = dict(vocab or {})
        self.merges: list[tuple[str, str]] = list(merges or [])
        self.merge_ranks: dict[tuple[str, str], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }

        self._ensure_special_tokens()
        self.ids_to_tokens = self._build_id_to_token_map(self.vocab, verbose=False)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def fit(
        self,
        texts: Iterable[str],
        vocab_size: int,
        min_frequency: int = 2,
        verbose: bool = False,
    ) -> BPETokenizer:
        if min_frequency < 1:
            raise ValueError(f"'min_frequency' must be >= 1.")
        special_tokens = self.all_special_tokens
        if vocab_size < len(special_tokens):
            raise ValueError(
                "'vocab_size' must be >= "
                f"number of special tokens ({len(special_tokens)})."
            )

        word_freq = self._build_word_frequency(texts)
        if not word_freq:
            raise ValueError("Cannot train BPE vocabulary from empty corpus.")

        word_splits: dict[str, list[str]] = {}
        token_set: set[str] = set()

        for word in word_freq:
            chars = list(word)
            if not chars:
                continue
            chars[-1] = chars[-1] + self.end_of_word_suffix
            word_splits[word] = chars
            token_set.update(chars)

        merges: list[tuple[str, str]] = []
        target_size = vocab_size - len(special_tokens)

        while len(token_set) < target_size:
            pair_counts: Counter[tuple[str, str]] = Counter()
            for word, split in word_splits.items():
                freq = word_freq[word]
                for i in range(len(split) - 1):
                    pair_counts[(split[i], split[i + 1])] += freq

            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_frequency:
                break

            merges.append(best_pair)
            merged_token = best_pair[0] + best_pair[1]
            token_set.add(merged_token)

            for word, split in word_splits.items():
                i = 0
                new_split: list[str] = []
                while i < len(split):
                    if (
                        i < len(split) - 1
                        and split[i] == best_pair[0]
                        and split[i + 1] == best_pair[1]
                    ):
                        new_split.append(merged_token)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1

                word_splits[word] = new_split

        vocab_tokens = sorted(token_set)[:target_size]
        full_vocab = special_tokens + vocab_tokens

        self.vocab = {tok: i for i, tok in enumerate(full_vocab)}
        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.ids_to_tokens = self._build_id_to_token_map(self.vocab, verbose=verbose)

        return self

    @classmethod
    def train_from_iterator(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        min_frequency: int = 2,
        verbose: bool = False,
        **kwargs: Any,
    ) -> BPETokenizer:
        tok = cls(**kwargs)
        tok.fit(
            texts=texts,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            verbose=verbose,
        )
        return tok

    def tokenize(self, text: str) -> list[str]:
        if self.clean_text:
            text = clean_text(text)
        basic_tokens = basic_tokenize(text, lowercase=self.lowercase)

        out: list[str] = []
        for tok in basic_tokens:
            out.extend(self._bpe_tokenize(tok))
        return out

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        unk_id = self._unk_id()
        if isinstance(tokens, str):
            return self.vocab.get(tokens, unk_id)
        return [self.vocab.get(t, unk_id) for t in tokens]

    def convert_ids_to_tokens(self, ids: int | list[int]) -> str | list[str]:
        if isinstance(ids, int):
            return self._id_to_token(ids)
        return [self._id_to_token(i) for i in ids]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        text = "".join(tokens)
        text = text.replace(self.end_of_word_suffix, " ")
        for punct in [".", ",", "!", "?", ";", ":", "%", ")", "]", "}"]:
            text = text.replace(f" {punct}", punct)
        for punct in ["(", "[", "{"]:
            text = text.replace(f"{punct} ", punct)

        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"([.!?])\1+", r"\1", text)

        return text

    def save_pretrained(self, save_directory: Path | str) -> list[str]:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        vocab_path = save_path / "vocab.json"
        merges_path = save_path / "merges.txt"

        self._save_json(self.vocab, vocab_path)
        with merges_path.open("w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")

        config = {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "lowercase": self.lowercase,
            "clean_text": self.clean_text,
            "end_of_word_suffix": self.end_of_word_suffix,
        }
        config_path = self._save_tokenizer_config(save_path, config)
        return [str(vocab_path), str(merges_path), str(config_path)]

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, **kwargs: Any
    ) -> BPETokenizer:
        path = Path(pretrained_model_name_or_path)
        if path.is_dir():
            base_dir = path
            vocab_file = base_dir / "vocab.json"
            merges_file = base_dir / "merges.txt"
        else:
            base_dir = path.parent
            vocab_file = path
            if vocab_file.name != "vocab.json":
                raise ValueError(
                    "Expected a directory containing 'vocab.json'/'merges.txt' "
                    "or a direct path to 'vocab.json'."
                )
            merges_file = base_dir / "merges.txt"

        if not vocab_file.exists():
            raise FileNotFoundError(f"Cannot find vocabulary file: {vocab_file}")
        if not merges_file.exists():
            raise FileNotFoundError(f"Cannot find merges file: {merges_file}")

        allowed = {
            "unk_token",
            "pad_token",
            "bos_token",
            "eos_token",
            "lowercase",
            "clean_text",
            "end_of_word_suffix",
        }
        init_kwargs = cls._load_tokenizer_config(base_dir, allowed_keys=allowed)
        init_kwargs.update(kwargs)

        return cls(vocab_file=vocab_file, merges_file=merges_file, **init_kwargs)

    @staticmethod
    def _load_vocab(vocab_file: Path | str) -> dict[str, int]:
        data = Tokenizer._load_json(vocab_file)
        out: dict[str, int] = {}
        for k, v in data.items():
            out[str(k)] = int(v)
        return out

    @staticmethod
    def _load_merges(merges_file: Path | str) -> list[tuple[str, str]]:
        merges: list[tuple[str, str]] = []
        with Path(merges_file).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()

                if len(parts) != 2:
                    continue
                merges.append((parts[0], parts[1]))

        return merges

    def _ensure_special_tokens(self) -> None:
        for tok in self.all_special_tokens:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)

    def _unk_id(self) -> int:
        if self.unk_token is None or self.unk_token not in self.vocab:
            raise ValueError("Unknown token is not in vocabulary.")
        return self.vocab[self.unk_token]

    def _id_to_token(self, idx: int) -> str:
        if idx < 0 or idx >= len(self.ids_to_tokens):
            return self.unk_token or SpecialTokens.UNK.value

        t = self.ids_to_tokens[idx]
        return t if t else (self.unk_token or SpecialTokens.UNK.value)

    def _bpe_tokenize(self, token: str) -> list[str]:
        if not token:
            return []

        symbols = list(token)
        symbols[-1] = symbols[-1] + self.end_of_word_suffix

        while len(symbols) > 1:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            ranked = [(self.merge_ranks.get(p, 10**18), p) for p in pairs]

            best_rank, best_pair = min(ranked, key=lambda x: x[0])
            if best_rank == 10**18:
                break

            merged = best_pair[0] + best_pair[1]
            new_symbols: list[str] = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == best_pair[0]
                    and symbols[i + 1] == best_pair[1]
                ):
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1

            symbols = new_symbols
        return symbols

    def _build_word_frequency(self, texts: Iterable[str]) -> Counter[str]:
        freq: Counter[str] = Counter()
        for text in texts:
            if self.clean_text:
                text = clean_text(text)
            for tok in basic_tokenize(text, lowercase=self.lowercase):
                if tok:
                    freq[tok] += 1
        return freq


class BPETokenizerFast(Tokenizer):
    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        merges: list[tuple[str, str]] | None = None,
        vocab_file: Path | str | None = None,
        merges_file: Path | str | None = None,
        unk_token: SpecialTokens | str = SpecialTokens.UNK,
        pad_token: SpecialTokens | str = SpecialTokens.PAD,
        bos_token: SpecialTokens | str | None = None,
        eos_token: SpecialTokens | str | None = None,
        lowercase: bool = True,
        clean_text: bool = True,
        end_of_word_suffix: str = "</w>",
    ) -> None:
        if vocab is not None and vocab_file is not None:
            raise ValueError("Provide only one of 'vocab' or 'vocab_file'.")
        if merges is not None and merges_file is not None:
            raise ValueError("Provide only one of 'merges' or 'merges_file'.")

        super().__init__(unk_token, pad_token, bos_token, eos_token)

        backend_vocab_file: Path | None = None
        backend_merges_file: Path | None = None
        if vocab_file is not None:
            vocab = BPETokenizer._load_vocab(vocab_file)
        else:
            backend_vocab_file = None
        if merges_file is not None:
            merges = BPETokenizer._load_merges(merges_file)
        else:
            backend_merges_file = None

        self.lowercase = lowercase
        self.clean_text = clean_text
        self.end_of_word_suffix = end_of_word_suffix

        self.vocab: dict[str, int] = dict(vocab or {})
        self.merges: list[tuple[str, str]] = list(merges or [])

        self._backend = _C_BPETokenizer(
            vocab=vocab,
            merges=merges,
            vocab_file=backend_vocab_file,
            merges_file=backend_merges_file,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            lowercase=self.lowercase,
            clean_text=self.clean_text,
            end_of_word_suffix=self.end_of_word_suffix,
        )
        self._sync_state_from_backend()

    @property
    def vocab_size(self) -> int:
        size = getattr(self._backend, "vocab_size")
        return int(size() if callable(size) else size)

    def tokenize(self, text: str) -> list[str]:
        return list(self._backend.tokenize(text))

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_tensor: bool = False,
        device: _DeviceType = "cpu",
    ) -> list[int] | lucid.LongTensor:
        ids = [int(x) for x in self._backend.encode_ids(text, add_special_tokens)]
        if return_tensor:
            return lucid.LongTensor(ids, device=device)
        return ids

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

    def fit(
        self,
        texts: Iterable[str],
        vocab_size: int,
        min_frequency: int = 2,
    ) -> BPETokenizerFast:
        self._backend.fit(list(texts), int(vocab_size), int(min_frequency))
        self._sync_state_from_backend()
        return self

    @classmethod
    def train_from_iterator(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        min_frequency: int = 2,
        **kwargs: Any,
    ) -> BPETokenizerFast:
        tokenizer = cls(**kwargs)
        tokenizer.fit(
            texts=texts,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )
        return tokenizer

    def save_pretrained(self, save_directory: Path | str) -> list[str]:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        vocab_path = save_path / "vocab.json"
        merges_path = save_path / "merges.txt"

        if not self.vocab:
            self._sync_state_from_backend()

        self._save_json(self.vocab, vocab_path)
        with merges_path.open("w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")

        config = {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "lowercase": self.lowercase,
            "clean_text": self.clean_text,
            "end_of_word_suffix": self.end_of_word_suffix,
        }
        config_path = self._save_tokenizer_config(save_path, config)
        return [str(vocab_path), str(merges_path), str(config_path)]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        **kwargs: Any,
    ) -> BPETokenizerFast:
        path = Path(pretrained_model_name_or_path)
        if path.is_dir():
            base_dir = path
            vocab_file = base_dir / "vocab.json"
            merges_file = base_dir / "merges.txt"
        else:
            base_dir = path.parent
            vocab_file = path
            if vocab_file.name != "vocab.json":
                raise ValueError(
                    "Expected a directory containing 'vocab.json'/'merges.txt' "
                    "or a direct path to 'vocab.json'."
                )
            merges_file = base_dir / "merges.txt"

        if not vocab_file.exists():
            raise FileNotFoundError(f"Cannot find vocabulary file: {vocab_file}")
        if not merges_file.exists():
            raise FileNotFoundError(f"Cannot find merges file: {merges_file}")

        allowed = {
            "unk_token",
            "pad_token",
            "bos_token",
            "eos_token",
            "lowercase",
            "clean_text",
            "end_of_word_suffix",
        }
        init_kwargs = cls._load_tokenizer_config(base_dir, allowed_keys=allowed)
        init_kwargs.update(kwargs)

        return cls(vocab_file=vocab_file, merges_file=merges_file, **init_kwargs)

    def _sync_state_from_backend(self) -> None:
        backend_vocab = self._backend.get_vocab()
        self.vocab = {str(k): int(v) for k, v in backend_vocab.items()}

        backend_merges = self._backend.get_merges()
        self.merges = [(str(a), str(b)) for a, b in backend_merges]


class ByteBPETokenizer(BPETokenizer):
    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        merges: list[tuple[str, str]] | None = None,
        vocab_file: Path | str | None = None,
        merges_file: Path | str | None = None,
        unk_token: SpecialTokens | str = SpecialTokens.UNK,
        pad_token: SpecialTokens | str = SpecialTokens.PAD,
        bos_token: SpecialTokens | str | None = None,
        eos_token: SpecialTokens | str | None = None,
        lowercase: bool = False,
        clean_text: bool = True,
        add_prefix_space: bool = False,
        end_of_word_suffix: str = "",
    ) -> None:
        loaded_from_artifacts = (
            vocab is not None
            or merges is not None
            or vocab_file is not None
            or merges_file is not None
        )
        self.add_prefix_space = add_prefix_space

        self.byte_encoder: dict[int, str] = self._byte_encoder()
        self.byte_decoder: dict[str, int] = {v: k for k, v in self.byte_encoder.items()}

        super().__init__(
            vocab,
            merges,
            vocab_file,
            merges_file,
            unk_token,
            pad_token,
            bos_token,
            eos_token,
            lowercase,
            clean_text,
            end_of_word_suffix,
        )
        if loaded_from_artifacts:
            self._validate_byte_vocab(self.vocab)

    @staticmethod
    @lru_cache(maxsize=1)
    def _byte_encoder() -> dict[int, str]:
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1

        return {b: chr(c) for b, c in zip(bs, cs)}

    @staticmethod
    def _validate_byte_vocab(vocab: dict[str, int]) -> None:
        if not vocab:
            return
        required = set(ByteBPETokenizer._byte_encoder().values())
        missing = required.difference(vocab.keys())
        if missing:
            raise ValueError(
                "Invalid ByteBPE vocabulary: missing required byte symbols "
                f"({len(missing)} missing)."
            )

    def _normalize_text(self, text: str) -> str:
        if self.clean_text:
            text = clean_text(text)
        if self.lowercase:
            text = text.lower()
        if self.add_prefix_space and text and not text.startswith(" "):
            text = " " + text
        return text

    def _bytes_to_symbols(self, text: str) -> list[str]:
        pieces = re.findall(r"\S+\s*|\s+", text)
        out: list[str] = []
        for p in pieces:
            if not p:
                continue
            b = p.encode("utf-8")
            out.append("".join(self.byte_encoder[x] for x in b))
        return out

    def _build_word_frequency(self, texts: Iterable[str]) -> Counter[str]:
        freq: Counter[str] = Counter()
        for text in texts:
            norm = self._normalize_text(text)
            for tok in self._bytes_to_symbols(norm):
                if tok:
                    freq[tok] += 1
        return freq

    def fit(
        self,
        texts: Iterable[str],
        vocab_size: int,
        min_frequency: int = 2,
        verbose: bool = False,
    ) -> ByteBPETokenizer:
        min_bbpe_vocab = len(self.all_special_tokens) + 256
        if vocab_size < min_bbpe_vocab:
            raise ValueError(
                "'vocab_size' must be >= number of special tokens + 256 for ByteBPE "
                f"(got {vocab_size}, need at least {min_bbpe_vocab})."
            )
        out = super().fit(
            texts=texts,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            verbose=verbose,
        )
        self._validate_byte_vocab(self.vocab)
        return out

    def tokenize(self, text: str) -> list[str]:
        norm = self._normalize_text(text)
        byte_tokens = self._bytes_to_symbols(norm)

        out: list[str] = []
        for tok in byte_tokens:
            out.extend(self._bpe_tokenize(tok))

        return out

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        bs = bytearray()
        special_tokens = set(self.all_special_tokens)
        for tok in tokens:
            if tok in special_tokens:
                continue

            piece = tok
            if self.end_of_word_suffix and piece.endswith(self.end_of_word_suffix):
                piece = piece[: -len(self.end_of_word_suffix)]

            for ch in piece:
                b = self.byte_decoder.get(ch)
                if b is not None:
                    bs.append(b)

        return bs.decode("utf-8", errors="replace")

    def save_pretrained(self, save_directory: Path | str) -> list[str]:
        paths = super().save_pretrained(save_directory)

        save_path = Path(save_directory)
        cfg = self._load_json(save_path / "tokenizer_config.json")
        cfg["add_prefix_space"] = self.add_prefix_space

        self._save_json(cfg, save_path / "tokenizer_config.json")
        return paths

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Path | str, **kwargs: Any
    ) -> ByteBPETokenizer:
        path = Path(pretrained_model_name_or_path)
        base_dir = path if path.is_dir() else path.parent

        allowed = {
            "unk_token",
            "pad_token",
            "bos_token",
            "eos_token",
            "lowercase",
            "clean_text",
            "end_of_word_suffix",
            "add_prefix_space",
        }
        init_kwargs = cls._load_tokenizer_config(base_dir, allowed_keys=allowed)
        init_kwargs.update(kwargs)

        if path.is_dir():
            vocab_file = base_dir / "vocab.json"
            merges_file = base_dir / "merges.txt"
        else:
            vocab_file = path
            merges_file = base_dir / "merges.txt"

        return cls(vocab_file=vocab_file, merges_file=merges_file, **init_kwargs)


class ByteBPETokenizerFast(BPETokenizerFast):
    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        merges: list[tuple[str, str]] | None = None,
        vocab_file: Path | str | None = None,
        merges_file: Path | str | None = None,
        unk_token: SpecialTokens | str = SpecialTokens.UNK,
        pad_token: SpecialTokens | str = SpecialTokens.PAD,
        bos_token: SpecialTokens | str | None = None,
        eos_token: SpecialTokens | str | None = None,
        lowercase: bool = False,
        clean_text: bool = True,
        add_prefix_space: bool = False,
        end_of_word_suffix: str = "",
    ) -> None:
        loaded_from_artifacts = (
            vocab is not None
            or merges is not None
            or vocab_file is not None
            or merges_file is not None
        )
        if vocab is not None and vocab_file is not None:
            raise ValueError("Provide only one of 'vocab' or 'vocab_file'.")
        if merges is not None and merges_file is not None:
            raise ValueError("Provide only one of 'merges' or 'merges_file'.")

        Tokenizer.__init__(self, unk_token, pad_token, bos_token, eos_token)

        backend_vocab_file: Path | None = None
        backend_merges_file: Path | None = None
        if vocab_file is not None:
            vocab = BPETokenizer._load_vocab(vocab_file)
        else:
            backend_vocab_file = None
        if merges_file is not None:
            merges = BPETokenizer._load_merges(merges_file)
        else:
            backend_merges_file = None

        self.lowercase = lowercase
        self.clean_text = clean_text
        self.add_prefix_space = add_prefix_space
        self.end_of_word_suffix = end_of_word_suffix

        self.vocab: dict[str, int] = dict(vocab or {})
        self.merges: list[tuple[str, str]] = list(merges or [])

        self._backend = _C_ByteBPETokenizer(
            vocab=vocab,
            merges=merges,
            vocab_file=backend_vocab_file,
            merges_file=backend_merges_file,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            lowercase=self.lowercase,
            clean_text=self.clean_text,
            add_prefix_space=self.add_prefix_space,
            end_of_word_suffix=self.end_of_word_suffix,
        )
        self._sync_state_from_backend()
        if loaded_from_artifacts:
            ByteBPETokenizer._validate_byte_vocab(self.vocab)

    def fit(
        self,
        texts: Iterable[str],
        vocab_size: int,
        min_frequency: int = 2,
    ) -> ByteBPETokenizerFast:
        min_bbpe_vocab = len(self.all_special_tokens) + 256
        if vocab_size < min_bbpe_vocab:
            raise ValueError(
                "'vocab_size' must be >= number of special tokens + 256 for ByteBPE "
                f"(got {vocab_size}, need at least {min_bbpe_vocab})."
            )
        out = super().fit(
            texts=texts,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )
        ByteBPETokenizer._validate_byte_vocab(self.vocab)
        return out

    def save_pretrained(self, save_directory: Path | str) -> list[str]:
        paths = super().save_pretrained(save_directory)
        save_path = Path(save_directory)

        cfg = self._load_json(save_path / "tokenizer_config.json")
        cfg["add_prefix_space"] = self.add_prefix_space

        self._save_json(cfg, save_path / "tokenizer_config.json")
        return paths

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        **kwargs: Any,
    ) -> "ByteBPETokenizerFast":
        path = Path(pretrained_model_name_or_path)
        if path.is_dir():
            base_dir = path
            vocab_file = base_dir / "vocab.json"
            merges_file = base_dir / "merges.txt"
        else:
            base_dir = path.parent
            vocab_file = path

            if vocab_file.name != "vocab.json":
                raise ValueError(
                    "Expected a directory containing 'vocab.json'/'merges.txt' "
                    "or a direct path to 'vocab.json'."
                )
            merges_file = base_dir / "merges.txt"

        if not vocab_file.exists():
            raise FileNotFoundError(f"Cannot find vocabulary file: {vocab_file}")
        if not merges_file.exists():
            raise FileNotFoundError(f"Cannot find merges file: {merges_file}")

        allowed = {
            "unk_token",
            "pad_token",
            "bos_token",
            "eos_token",
            "lowercase",
            "clean_text",
            "add_prefix_space",
            "end_of_word_suffix",
        }
        init_kwargs = cls._load_tokenizer_config(base_dir, allowed_keys=allowed)
        init_kwargs.update(kwargs)

        return cls(vocab_file=vocab_file, merges_file=merges_file, **init_kwargs)
