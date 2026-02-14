tokenizers.WordPieceTokenizerFast
=================================
|cpp-badge|

.. autoclass:: lucid.data.tokenizers.WordPieceTokenizerFast

`WordPieceTokenizerFast` is the high-performance WordPiece tokenizer implementation
in Lucid. It shares the same Python-facing interface as `WordPieceTokenizer`, but
delegates tokenization and vocabulary operations to a native C++ backend exposed via
the `lucid.data.tokenizers._C` extension module.

Class Signature
---------------

.. code-block:: python

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
        ) -> None

Typical Usage
-------------

.. code-block:: python

    from lucid.data.tokenizers import WordPieceTokenizerFast

    tokenizer = WordPieceTokenizerFast.from_pretrained(".data/bert/pretrained")

    ids = tokenizer.encode("Hello, Lucid!")
    text = tokenizer.decode(ids)

    batch_ids = tokenizer.batch_encode(
        ["hello world", "tokenizer fast path"],
        add_special_tokens=True,
    )

.. note::

    - This class requires the compiled C++ extension (`lucid.data.tokenizers._C`).
    - If the extension is not built, importing or constructing the tokenizer will fail
      with an import/runtime error.
    - For development builds, run `python setup.py build_ext --inplace` before use.
