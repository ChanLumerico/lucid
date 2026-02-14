Tokenizers
==========

.. toctree::
    :maxdepth: 1
    :hidden:

    Tokenizer.rst
    
    WordPieceTokenizer <WordPieceTokenizer.rst>
    WordPieceTokenizerFast <WordPieceTokenizerFast.rst>

The `lucid.data.tokenizers` package provides tokenizer interfaces for text pipelines.
It focuses on converting text into token ids and reconstructing text from ids in a way
that integrates cleanly with Lucid tensors and data loaders.

Overview
--------

Key responsibilities of a tokenizer in Lucid:

- Split text into token units.
- Convert tokens to integer ids for model input.
- Convert ids back to readable text.
- Handle special tokens such as `[PAD]`, `[UNK]`, `[BOS]`, and `[EOS]`.
- Save and reload tokenizer state with pretrained-style APIs.

The base class is :class:`lucid.data.tokenizers.Tokenizer`.

Quick Start
-----------

Single input:

.. code-block:: python

    from lucid.data.tokenizers import Tokenizer

    tokenizer = MyTokenizer.from_pretrained("path/to/tokenizer")
    ids = tokenizer.encode("hello lucid", add_special_tokens=True)
    text = tokenizer.decode(ids)

Batch input (2D tensor output):

.. code-block:: python

    import lucid

    batch_ids = tokenizer.batch_encode(
        ["hello lucid", "hi"],
        padding=True,
        return_tensor=True,
        device="cpu",
    )
    assert isinstance(batch_ids, lucid.Tensor)  # shape: (batch, seq_len)

Save / load:

.. code-block:: python

    tokenizer.save_pretrained("out/my_tokenizer")
    tokenizer2 = MyTokenizer.from_pretrained("out/my_tokenizer")

.. note::

    - `encode(..., return_tensor=True)` returns a 1D `lucid.LongTensor`.
    - `batch_encode(..., return_tensor=True)` returns a 2D tensor when lengths are aligned
      (or when `padding=True` is set).
    - `decode` accepts both `list[int]` and `lucid.LongTensor` inputs.
