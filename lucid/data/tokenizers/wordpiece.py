import unicodedata

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from lucid.data.tokenizers import Tokenizer, SpecialTokens

from lucid._backend._C.tokenizers.core import _C_WordPieceTokenizer


__all__ = ["WordPieceTokenizer", "WordPieceTokenizerFast"]


class WordPieceTokenizer(Tokenizer):
    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        vocab_file: Path | str | None = None,
        unk_token: SpecialTokens | str = SpecialTokens.UNK,
        pad_token: SpecialTokens | str = SpecialTokens.PAD,
        bos_token: SpecialTokens | str | None = None,
        eos_token: SpecialTokens | str | None = None,
        lowercase: bool = True,
        wordpieces_prefix: str = "##",
        max_input_chars_per_word: int = 100,
        clean_text: bool = True,
        verbose: bool = False,
    ) -> None:
        if vocab is not None and vocab_file is not None:
            raise ValueError("Provide only one of 'vocab' or 'vocab_file'.")

        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
        )

        if vocab_file is not None:
            vocab = self._load_vocab(vocab_file)
        if vocab is None:
            vocab = {}

        self.lowercase = lowercase
        self.wordpieces_prefix = wordpieces_prefix
        self.max_input_chars_per_word = max_input_chars_per_word
        self.clean_text = clean_text

        self.vocab: dict[str, int] = dict(vocab)
        self._ensure_special_tokens()
        self.ids_to_tokens = self._build_id_to_token_map(self.vocab, verbose=verbose)

    def fit(
        self,
        texts: Iterable[str],
        vocab_size: int,
        min_frequency: int = 2,
        verbose: bool = False,
    ) -> WordPieceTokenizer:
        trained_vocab = self._train_vocab(
            texts=texts,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )
        self.vocab = trained_vocab
        unk_token = self.unk_token or SpecialTokens.UNK.value

        self.ids_to_tokens = self._build_id_to_token_map(
            self.vocab, unk_token=unk_token, verbose=verbose
        )
        return self

    @classmethod
    def train_from_iterator(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        min_frequency: int = 2,
        verbose: bool = False,
        **kwargs,
    ) -> WordPieceTokenizer:
        tokenizer = cls(**kwargs)
        tokenizer.fit(
            texts, vocab_size=vocab_size, min_frequency=min_frequency, verbose=verbose
        )
        return tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def tokenize(self, text: str) -> list[str]:
        if self.clean_text:
            text = self._clean_text(text)

        basic_tokens = self._basic_tokenize(text)
        tokens: list[str] = []
        for token in basic_tokens:
            tokens.extend(self._wordpiece_tokenize(token))
        return tokens

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        unk_id = self._unk_id()
        if isinstance(tokens, str):
            return self.vocab.get(tokens, unk_id)
        return [self.vocab.get(token, unk_id) for token in tokens]

    def convert_ids_to_tokens(self, ids: int | list[int]) -> str | list[str]:
        if isinstance(ids, int):
            return self._id_to_token(ids)
        return [self._id_to_token(i) for i in ids]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        out = " ".join(tokens)
        out = out.replace(f" {self.wordpieces_prefix}", "")

        for punct in [".", ",", "!", "?", ";", ":", "%", ")", "]", "}"]:
            out = out.replace(f" {punct}", punct)
        for punct in ["(", "[", "{"]:
            out = out.replace(f"{punct} ", punct)

        return out.strip()

    def save_pretrained(self, save_directory: Path | str) -> list[str]:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        vocab_path = save_path / "vocab.txt"
        config_path = save_path / "tokenizer_config.json"

        with vocab_path.open("w", encoding="utf-8") as f:
            for token in self.ids_to_tokens:
                f.write(token + "\n")

        config = {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "lowercase": self.lowercase,
            "wordpieces_prefix": self.wordpieces_prefix,
            "max_input_chars_per_word": self.max_input_chars_per_word,
            "clean_text": self.clean_text,
        }
        self._save_tokenizer_config(save_path, config)

        return [str(vocab_path), str(config_path)]

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, **kwrags
    ) -> WordPieceTokenizer:
        path = Path(pretrained_model_name_or_path)
        if path.is_dir():
            vocab_file = path / "vocab.txt"
        else:
            vocab_file = path

        if not vocab_file.exists():
            raise FileNotFoundError(f"Cannot find vocabulary file: {vocab_file}")

        allowed = {
            "unk_token",
            "pad_token",
            "bos_token",
            "eos_token",
            "lowercase",
            "wordpieces_prefix",
            "max_input_chars_per_word",
            "clean_text",
        }
        init_kwargs = cls._load_tokenizer_config(path, allowed_keys=allowed)

        init_kwargs.update(kwrags)
        return cls(vocab_file=vocab_file, **init_kwargs)

    @staticmethod
    def _load_vocab(vocab_file: Path | str) -> dict[str, int]:
        vocab: dict[str, int] = {}
        with Path(vocab_file).open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.rstrip("\n")
                if token:
                    vocab[token] = idx
        return vocab

    def _ensure_special_tokens(self) -> None:
        for token in self.all_special_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def _train_vocab(
        self,
        texts: Iterable[str],
        vocab_size: int,
        min_frequency: int,
    ) -> dict[str, int]:
        special_tokens = self.all_special_tokens
        if vocab_size < len(special_tokens):
            raise ValueError(
                f"'vocab_size' must be >= number of special tokens ({len(special_tokens)})."
            )
        if min_frequency < 1:
            raise ValueError("'min_frequency' must be >= 1.")

        word_freq = self._build_word_frequency(texts)
        if not word_freq:
            raise ValueError("Cannot train WordPiece vocabulary from empty corpus.")

        word_splits, token_set = self._initialize_splits(word_freq)
        merged_tokens: list[str] = []

        target_size = vocab_size - len(special_tokens)
        while len(token_set) + len(merged_tokens) < target_size:
            best_pair = self._select_best_pair(word_splits, word_freq)
            if best_pair is None:
                break
            merged = self._merge_pair(best_pair, word_splits)
            if merged not in token_set and merged not in merged_tokens:
                merged_tokens.append(merged)

        vocab_tokens = list(dict.fromkeys(sorted(token_set) + merged_tokens))
        vocab_tokens = vocab_tokens[:target_size]

        full_vocab = special_tokens + vocab_tokens
        return {token: i for i, token in enumerate(full_vocab)}

    def _build_word_frequency(self, texts: Iterable[str]) -> Counter[str]:
        word_freq: Counter[str] = Counter()
        for text in texts:
            if self.clean_text:
                text = self._clean_text(text)
            for token in self._basic_tokenize(text):
                if token:
                    word_freq[token] += 1
        return word_freq

    def _initialize_splits(
        self, word_freq: Counter[str]
    ) -> tuple[dict[str, list[str]], set[str]]:
        word_splits: dict[str, list[str]] = {}
        token_set: set[str] = set()

        for word in word_freq:
            if not word:
                continue
            split = [word[0]] + [self.wordpieces_prefix + ch for ch in word[1:]]
            word_splits[word] = split
            token_set.update(split)

        return word_splits, token_set

    def _select_best_pair(
        self, word_splits: dict[str, list[str]], word_freq: Counter[str]
    ) -> tuple[str, str] | None:
        token_freq: Counter[str] = Counter()
        pair_freq: Counter[tuple[str, str]] = Counter()

        for word, split in word_splits.items():
            freq = word_freq[word]
            for token in split:
                token_freq[token] += freq
            for i in range(len(split) - 1):
                pair_freq[(split[i], split[i + 1])] += freq

        best_pair: tuple[str, str] | None = None
        best_score = -1.0
        best_pair_count = -1

        for pair, pfreq in pair_freq.items():
            if pfreq < 1:
                continue
            left, right = pair
            denom = token_freq[left] * token_freq[right]
            if denom == 0:
                continue
            score = pfreq / denom
            if (
                score > best_score
                or (score == best_score and pfreq > best_pair_count)
                or (
                    score == best_score
                    and pfreq == best_pair_count
                    and pair < (best_pair or pair)
                )
            ):
                best_score = score
                best_pair_count = pfreq
                best_pair = pair

        if best_pair is None or best_pair_count < 1:
            return None
        return best_pair

    def _merge_pair(
        self, pair: tuple[str, str], word_splits: dict[str, list[str]]
    ) -> str:
        left, right = pair
        if right.startswith(self.wordpieces_prefix):
            merged_token = left + right[len(self.wordpieces_prefix) :]
        else:
            merged_token = left + right

        for word, split in word_splits.items():
            if len(split) < 2:
                continue
            new_split: list[str] = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == left and split[i + 1] == right:
                    new_split.append(merged_token)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            word_splits[word] = new_split

        return merged_token

    def _unk_id(self) -> int:
        unk = self.unk_token
        if unk is None:
            raise ValueError("'unk_token' must not be None for WordPiece tokenizer.")
        if unk not in self.vocab:
            raise ValueError(f"Unknown token '{unk}' is not in vocabulary.")
        return self.vocab[unk]

    def _id_to_token(self, idx: int) -> str:
        if idx < 0 or idx >= len(self.ids_to_tokens):
            return self.unk_token or SpecialTokens.UNK.value
        token = self.ids_to_tokens[idx]
        return token if token else (self.unk_token or SpecialTokens.UNK.value)

    def _basic_tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for token in text.strip().split():
            if self.lowercase:
                token = token.lower()
            tokens.extend(self._split_on_punctuation(token))
        return tokens

    @staticmethod
    def _split_on_punctuation(token: str) -> list[str]:
        out: list[str] = []
        current: list[str] = []
        for ch in token:
            if WordPieceTokenizer._is_punctuation(ch):
                if current:
                    out.append("".join(current))
                    current = []
                out.append(ch)
            else:
                current.append(ch)
        if current:
            out.append("".join(current))
        return out

    def _wordpiece_tokenize(self, token: str) -> list[str]:
        if len(token) > self.max_input_chars_per_word:
            return [self.unk_token or SpecialTokens.UNK.value]

        chars = list(token)
        start = 0
        sub_tokens: list[str] = []

        while start < len(chars):
            end = len(chars)
            cur_substr: str | None = None
            while start < end:
                substr = "".join(chars[start:end])
                if start > 0:
                    substr = self.wordpieces_prefix + substr
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1

            if cur_substr is None:
                return [self.unk_token or SpecialTokens.UNK.value]

            sub_tokens.append(cur_substr)
            start = end

        return sub_tokens

    @staticmethod
    def _is_whitespace(ch: str) -> bool:
        if ch in (" ", "\t", "\n", "\r"):
            return True
        return unicodedata.category(ch) == "Zs"

    @staticmethod
    def _is_control(ch: str) -> bool:
        if ch in ("\t", "\n", "\r"):
            return False
        return unicodedata.category(ch) in {"Cc", "Cf"}

    @staticmethod
    def _is_punctuation(ch: str) -> bool:
        cp = ord(ch)
        if (
            (33 <= cp <= 47)
            or (58 <= cp <= 64)
            or (91 <= cp <= 96)
            or (123 <= cp <= 126)
        ):
            return True
        return unicodedata.category(ch).startswith("P")

    @staticmethod
    def _clean_text(text: str) -> str:
        out: list[str] = []
        for ch in text:
            cp = ord(ch)
            if cp in (0, 0xFFFD) or WordPieceTokenizer._is_control(ch):
                continue
            if WordPieceTokenizer._is_whitespace(ch):
                out.append(" ")
            else:
                out.append(ch)
        return "".join(out)


class WordPieceTokenizerFast(Tokenizer):
    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        vocab_file: Path | str | None = None,
        unk_token: SpecialTokens | str = SpecialTokens.UNK,
        pad_token: SpecialTokens | str = SpecialTokens.PAD,
        bos_token: SpecialTokens | str | None = None,
        eos_token: SpecialTokens | str | None = None,
        lowercase: bool = True,
        wordpieces_prefix: str = "##",
        max_input_chars_per_word: int = 100,
        clean_text: bool = True,
    ) -> None:
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
        )

        self.lowercase = lowercase
        self.wordpieces_prefix = wordpieces_prefix
        self.max_input_chars_per_word = max_input_chars_per_word
        self.clean_text = clean_text

        if _C_WordPieceTokenizer is None:
            raise ImportError(
                "Cannot import native C++ WordPiece backend "
                "'lucid._backend._C.tokenizers.core'. "
                "Build extensions first."
            )

        self._backend = _C_WordPieceTokenizer(
            vocab=vocab,
            vocab_file=Path(vocab_file) if vocab_file is not None else None,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
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

    def fit(
        self,
        texts: Iterable[str],
        vocab_size: int,
        min_frequency: int = 2,
        **_: Any,
    ) -> WordPieceTokenizerFast:
        self._backend.fit(list(texts), int(vocab_size), int(min_frequency))
        return self

    @classmethod
    def train_from_iterator(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        min_frequency: int = 2,
        **kwargs: Any,
    ) -> WordPieceTokenizerFast:
        tokenizer = cls(**kwargs)
        tokenizer.fit(texts, vocab_size=vocab_size, min_frequency=min_frequency)
        return tokenizer

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
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
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
    ) -> WordPieceTokenizerFast:
        path = Path(pretrained_model_name_or_path)
        vocab_file = path / "vocab.txt" if path.is_dir() else path
        if not vocab_file.exists():
            raise FileNotFoundError(f"Cannot find vocabulary file: {vocab_file}")

        allowed = {
            "unk_token",
            "pad_token",
            "bos_token",
            "eos_token",
            "lowercase",
            "wordpieces_prefix",
            "max_input_chars_per_word",
            "clean_text",
        }
        init_kwargs = cls._load_tokenizer_config(path, allowed_keys=allowed)
        init_kwargs.update(kwrags)
        return cls(vocab_file=vocab_file, **init_kwargs)
