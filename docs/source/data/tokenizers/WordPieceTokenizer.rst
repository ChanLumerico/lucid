data.tokenizers.WordPieceTokenizer
==================================

.. autoclass:: lucid.data.tokenizers.WordPieceTokenizer

`WordPieceTokenizer` is the pure Python WordPiece tokenizer implementation in Lucid.
It provides training and inference utilities with a Hugging Face-style interface and
can be used directly when a native extension is not required.

Class Signature
---------------

.. code-block:: python

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
        ) -> None

Typical Usage
-------------

.. code-block:: python

    from lucid.data.tokenizers import WordPieceTokenizer

    tokenizer = WordPieceTokenizer.from_pretrained(".data/bert/pretrained")

    ids = tokenizer.encode("Hello, Lucid!")
    text = tokenizer.decode(ids)

    tokenizer.fit(
        texts=["hello world", "lucid tokenizer"],
        vocab_size=1000,
        min_frequency=2,
    )

.. note::

    - `WordPieceTokenizerFast` provides the same high-level API but runs core logic in C++.
    - Prefer `WordPieceTokenizerFast` for large-scale encoding workloads when the extension
      module is available.
