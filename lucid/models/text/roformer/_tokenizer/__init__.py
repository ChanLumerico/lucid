"""RoFormer family tokenizer wrappers.

RoFormer (Su et al., 2021) shares BERT's WordPiece vocab format +
special-token set; only the underlying model differs (rotary position
embeddings).  These wrappers thus reuse :class:`WordPieceTokenizer` /
:class:`WordPieceTokenizerFast` with the BERT default registry.
"""

from lucid.models.text.roformer._tokenizer._tokenizer import (
    RoFormerTokenizer,
    RoFormerTokenizerFast,
)

__all__ = ["RoFormerTokenizer", "RoFormerTokenizerFast"]
