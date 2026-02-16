BPETokenizerFast
================
|cpp-badge|

.. autoclass:: lucid.data.tokenizers.BPETokenizerFast

`BPETokenizerFast` is the high-performance BPE tokenizer implementation in Lucid.
It follows the same Python-facing interface as `BPETokenizer`, but delegates core
tokenization, vocabulary lookup, and merge-rule application to the native C++ backend.

Class Signature
---------------

.. code-block:: python

    class BPETokenizerFast(Tokenizer):
        def __init__(
            self,
            vocab: dict[str, int] | None = None,
            merges: list[tuple[str, str]] | None = None,
            vocab_file: Path | str | None = None,
            merges_file: Path | str | None = None,
            unk_token: SpecialTokens | str = SpecialTokens.UNK,
            pad_token: SpecialTokens | str = SpecialTokens.PAD,
            bos_token: SpecialTokens | str | None = None,
            eos_token: SpecialTokens | str | None = None,
            lowercase: bool = True,
            clean_text: bool = True,
            end_of_word_suffix: str = "</w>",
        ) -> None

Parameters
----------

- **vocab** (*dict[str, int] | None*): Optional in-memory token-to-id mapping used
  to initialize the tokenizer.
- **merges** (*list[tuple[str, str]] | None*): Optional BPE merge rule list as
  `(left, right)` token pairs.
- **vocab_file** (*Path | str | None*): Path to `vocab.json`. If provided, `vocab`
  must be `None`.
- **merges_file** (*Path | str | None*): Path to `merges.txt`. If provided, `merges`
  must be `None`.
- **unk_token** (*SpecialTokens | str*): Token used for unknown/out-of-vocabulary
  pieces.
- **pad_token** (*SpecialTokens | str*): Token used for sequence padding.
- **bos_token** (*SpecialTokens | str | None*): Optional beginning-of-sequence token.
- **eos_token** (*SpecialTokens | str | None*): Optional end-of-sequence token.
- **lowercase** (*bool*): Whether to lowercase input text during basic tokenization.
- **clean_text** (*bool*): Whether to normalize whitespace and remove control
  characters before tokenization.
- **end_of_word_suffix** (*str*): Suffix attached to each word-final symbol during
  BPE training/inference (default: `"</w>"`).

Typical Usage
-------------

.. code-block:: python

    from lucid.data.tokenizers import BPETokenizerFast

    tokenizer = BPETokenizerFast.from_pretrained("some_path")

    ids = tokenizer.encode("Machine learning improves tokenization quality.")
    text = tokenizer.decode(ids)

    batch_ids = tokenizer.batch_encode(
        ["hello world", "bpe fast path"],
        add_special_tokens=True,
    )

.. note::

    - This class requires the compiled C++ extension backend.
    - If the extension is not built, importing or constructing this tokenizer will fail.
