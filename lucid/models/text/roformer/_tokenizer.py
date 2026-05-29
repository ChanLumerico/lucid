"""RoFormer family tokenizer wrappers — BERT-style WordPiece.

RoFormer (Su et al., 2021) is a BERT-architecture model with rotary
positional embeddings — but at the tokenizer level it shares BERT's
vocabulary format and special-token convention exactly.  These
wrappers exist for *uniformity* with the rest of the text-model zoo:
every family ships a ``{Family}Tokenizer`` + ``{Family}TokenizerFast``
pair so call sites are interchangeable across model families.

Internally :class:`RoFormerTokenizer` /
:class:`RoFormerTokenizerFast` delegate to
:class:`~lucid.utils.tokenizer.WordPieceTokenizer` /
:class:`~lucid.utils.tokenizer.WordPieceTokenizerFast` with the
canonical ``[UNK]/[CLS]/[SEP]/[PAD]/[MASK]`` registry plus
:class:`~lucid.utils.tokenizer.normalizers.BERTNormalizer` (lowercase
by default) and
:class:`~lucid.utils.tokenizer.pre_tokenizers.WhitespacePunctuationSplit`.

Existing published RoFormer ``vocab.txt`` checkpoints load
unchanged.
"""

from lucid.utils.tokenizer._base import SpecialTokens
from lucid.utils.tokenizer._normalizers import BERTNormalizer, Normalizer
from lucid.utils.tokenizer._pre_tokenizers import (
    PreTokenizer,
    WhitespacePunctuationSplit,
)
from lucid.utils.tokenizer._wordpiece import (
    WordPieceTokenizer,
    WordPieceTokenizerFast,
)


def _roformer_special_tokens() -> SpecialTokens:
    """Return RoFormer's special-token registry (identical to BERT).

    Returns
    -------
    SpecialTokens
        Registry with ``unk=[UNK]``, ``pad=[PAD]``, ``cls=[CLS]``,
        ``sep=[SEP]``, ``mask=[MASK]``.  RoFormer reuses BERT's
        vocab + special-token layout verbatim.
    """
    return SpecialTokens(
        unk="[UNK]",
        pad="[PAD]",
        cls="[CLS]",
        sep="[SEP]",
        mask="[MASK]",
    )


class RoFormerTokenizer(WordPieceTokenizer):
    r"""RoFormer tokenizer — pure-Python reference.

    Identical algorithm to :class:`~lucid.models.text.bert.BERTTokenizer`
    (WordPiece longest-match with ``[UNK]/[CLS]/[SEP]/[PAD]/[MASK]``
    registered out of the box).  Kept as a separate class so every
    text-model family exposes a uniform
    ``from lucid.models.text.<family> import {Family}Tokenizer``
    surface — useful for generic model-loading helpers.

    Parameters
    ----------
    vocab : dict[str, int]
        BERT-style vocab with ``##`` continuation prefix.
    unk_token : str, default ``"[UNK]"``
    continuing_prefix : str, default ``"##"``
    max_chars_per_word : int, default 100
    do_lower_case : bool, default ``True``
        Set ``False`` for cased RoFormer checkpoints.
    normalizer : Normalizer, optional
    pre_tokenizer : PreTokenizer, optional
    special_tokens : SpecialTokens, optional

    See Also
    --------
    RoFormerTokenizerFast : C++-backed variant with identical output.
    lucid.models.text.bert.BERTTokenizer : Sibling wrapper sharing
        the same algorithm and defaults.
    """

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
                else BERTNormalizer(lowercase=do_lower_case)
            ),
            pre_tokenizer=(
                pre_tokenizer
                if pre_tokenizer is not None
                else WhitespacePunctuationSplit()
            ),
            special_tokens=(
                special_tokens
                if special_tokens is not None
                else _roformer_special_tokens()
            ),
        )


class RoFormerTokenizerFast(WordPieceTokenizerFast):
    """RoFormer tokenizer — C++-backed.

    Bit-identical to :class:`RoFormerTokenizer`.  The greedy
    longest-match loop runs in C++ via
    :class:`lucid._C.engine.utils.tokenizer.WordPiece`.

    Constructor parameters mirror :class:`RoFormerTokenizer` — see
    that class for the full reference.
    """

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
                else BERTNormalizer(lowercase=do_lower_case)
            ),
            pre_tokenizer=(
                pre_tokenizer
                if pre_tokenizer is not None
                else WhitespacePunctuationSplit()
            ),
            special_tokens=(
                special_tokens
                if special_tokens is not None
                else _roformer_special_tokens()
            ),
        )
