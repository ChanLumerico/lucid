"""``lucid.utils.tokenizer`` — text tokenization sub-package.

Every tokenizer comes in two flavours:

* ``XxxTokenizer`` — pure-Python reference implementation.  Easy to
  read, easy to debug, easy to extend with custom normalizers.
* ``XxxTokenizerFast`` — C++-backed via
  ``lucid._C.engine.utils.tokenizer``.  Same vocab format, same
  encode output bit-for-bit, but the algorithm hot loop runs in C++.

Both flavours share the Hugging Face-compatible on-disk format
(``vocab.json`` + ``merges.txt`` legacy or unified
``tokenizer.json``) so any published HF checkpoint loads without
modification.  The :class:`Tokenizer` base class layers a uniform
HF-style API on top — :meth:`~Tokenizer.encode` /
:meth:`~Tokenizer.decode` / batch versions / HF-style
:meth:`~Tokenizer.__call__` with padding + truncation +
``return_tensors='lucid'``.

Algorithms (current + planned)
------------------------------

==========  ===============  =============================  ==========
algo        modules           used by                        status
==========  ===============  =============================  ==========
BPE         ``_bpe``         (raw BPE — base for byte-BPE)  ✅ landed
ByteLevel   ``_byte_bpe``    GPT, GPT-2, RoBERTa, BART      ⏳ planned
WordPiece   ``_wordpiece``   BERT, RoFormer, DistilBERT     ⏳ planned
Unigram     ``_unigram``     T5, LLaMA, Mistral, mBART      ⏳ planned
==========  ===============  =============================  ==========

Per-model wrappers live in each ``lucid.models.text.<family>``
package's ``_tokenizer/`` directory and subclass the algorithm
matching that family (e.g. ``BERTTokenizer`` ←
:class:`WordPieceTokenizer`).
"""

from lucid.utils.tokenizer._base import SpecialTokens, Tokenizer
from lucid.utils.tokenizer._bpe import BPETokenizer, BPETokenizerFast
from lucid.utils.tokenizer._byte import ByteTokenizer, ByteTokenizerFast
from lucid.utils.tokenizer._byte_bpe import (
    ByteLevelBPETokenizer,
    ByteLevelBPETokenizerFast,
)
from lucid.utils.tokenizer._char import CharTokenizer, CharTokenizerFast
from lucid.utils.tokenizer._regex import RegexTokenizer, RegexTokenizerFast
from lucid.utils.tokenizer._unigram import (
    UnigramTokenizer,
    UnigramTokenizerFast,
)
from lucid.utils.tokenizer._whitespace import (
    WhitespaceTokenizer,
    WhitespaceTokenizerFast,
)
from lucid.utils.tokenizer._word import WordTokenizer, WordTokenizerFast
from lucid.utils.tokenizer._wordpiece import (
    WordPieceTokenizer,
    WordPieceTokenizerFast,
)
from lucid.utils.tokenizer import _normalizers as normalizers
from lucid.utils.tokenizer import _pre_tokenizers as pre_tokenizers

__all__ = [
    # Base
    "Tokenizer",
    "SpecialTokens",
    # Tier 0 — primitive / no-vocab
    "ByteTokenizer",
    "ByteTokenizerFast",
    "CharTokenizer",
    "CharTokenizerFast",
    # Tier 1 — rule-based / vocab-lookup
    "WhitespaceTokenizer",
    "WhitespaceTokenizerFast",
    "WordTokenizer",
    "WordTokenizerFast",
    "RegexTokenizer",
    "RegexTokenizerFast",
    # Tier 2 — subword
    "BPETokenizer",
    "BPETokenizerFast",
    "ByteLevelBPETokenizer",
    "ByteLevelBPETokenizerFast",
    "WordPieceTokenizer",
    "WordPieceTokenizerFast",
    "UnigramTokenizer",
    "UnigramTokenizerFast",
    # Sub-modules
    "normalizers",
    "pre_tokenizers",
]
