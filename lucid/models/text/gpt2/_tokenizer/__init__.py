"""GPT-2 family tokenizer wrappers.

GPT-2 (Radford et al., 2019) standardised the byte-level BPE that
became the default for GPT-2 / GPT-3 / RoBERTa / BART / CodeGen.
These wrappers bake in:

* ``<|endoftext|>`` as the bos / eos / unk token
* ``add_prefix_space = False`` (GPT-2 default; pass ``True`` to mimic
  RoBERTa's space-prepended convention)

Loads/saves any HF GPT-2 ``vocab.json`` + ``merges.txt`` pair or
unified ``tokenizer.json``.
"""

from lucid.models.text.gpt2._tokenizer._tokenizer import (
    GPT2Tokenizer,
    GPT2TokenizerFast,
)

__all__ = ["GPT2Tokenizer", "GPT2TokenizerFast"]
