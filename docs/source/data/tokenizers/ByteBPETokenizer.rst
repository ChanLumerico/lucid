ByteBPETokenizer
================

.. autoclass:: lucid.data.tokenizers.ByteBPETokenizer

`ByteBPETokenizer` is the pure Python byte-level BPE tokenizer implementation in Lucid.
It extends BPE tokenization by operating on UTF-8 byte-symbol sequences, allowing robust
handling of multilingual and out-of-vocabulary text without requiring explicit Unicode
normalization vocab coverage.

Class Signature
---------------

.. code-block:: python

    class ByteBPETokenizer(Tokenizer):
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
            lowercase: bool = False,
            clean_text: bool = True,
            add_prefix_space: bool = False,
            end_of_word_suffix: str = "",
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
- **lowercase** (*bool*): Whether to lowercase input text before byte-symbol conversion.
- **clean_text** (*bool*): Whether to normalize whitespace and remove control
  characters before tokenization.
- **add_prefix_space** (*bool*): Whether to prepend a leading space to text that does
  not already start with whitespace.
- **end_of_word_suffix** (*str*): Optional suffix attached to final symbol during BPE
  training/inference.

Typical Usage
-------------

.. code-block:: python

    from lucid.data.tokenizers import ByteBPETokenizer

    texts = [
        "Machine learning helps us build useful systems.",
        "Byte-level BPE should handle UTF-8 like café, naïve, and 한글.",
    ]

    tokenizer = ByteBPETokenizer.train_from_iterator(
        texts=texts,
        vocab_size=1000,
        min_frequency=2,
        add_prefix_space=False,
    )

    ids = tokenizer.encode("Machine learning helps café and 한글.")
    text = tokenizer.decode(ids)

    tokenizer.save_pretrained(".data/bbpe/pretrained")
    tokenizer2 = ByteBPETokenizer.from_pretrained(".data/bbpe/pretrained")

.. note::

    - This is the Python implementation of byte-level BPE.
    - Saved tokenizer directories are expected to contain:
      `vocab.json`, `merges.txt`, and `tokenizer_config.json`.
