"""GPT family tokenizer wrappers.

The original GPT (Radford et al., 2018) used a BPE vocab over
byte-mapped Unicode characters (the same family GPT-2 / RoBERTa
later popularised).  These wrappers subclass
:class:`ByteLevelBPETokenizer` /
:class:`ByteLevelBPETokenizerFast` with GPT-1's defaults:

* ``add_prefix_space = False``
* no end-of-text marker by default (GPT-1 was trained without one)

Loads/saves any HF GPT-1 ``vocab.json`` + ``merges.txt`` pair.
"""

from lucid.models.text.gpt._tokenizer._tokenizer import (
    GPTTokenizer,
    GPTTokenizerFast,
)

__all__ = ["GPTTokenizer", "GPTTokenizerFast"]
