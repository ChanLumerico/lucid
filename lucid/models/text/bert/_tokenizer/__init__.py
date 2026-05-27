"""BERT family tokenizer wrappers.

Both flavours subclass :class:`WordPieceTokenizer` /
:class:`WordPieceTokenizerFast` with BERT's canonical special-token
set baked in as the default:

* ``[UNK]`` — unknown
* ``[CLS]`` — classification head
* ``[SEP]`` — segment separator
* ``[PAD]`` — padding
* ``[MASK]`` — MLM mask

Existing pretrained BERT ``vocab.txt`` files load via
:meth:`~BertTokenizer.from_pretrained` without modification.
"""

from lucid.models.text.bert._tokenizer._tokenizer import (
    BertTokenizer,
    BertTokenizerFast,
)

__all__ = ["BertTokenizer", "BertTokenizerFast"]
