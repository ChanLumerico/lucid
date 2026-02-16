BPETokenizer
============

.. autoclass:: lucid.data.tokenizers.BPETokenizer

`BPETokenizer` is the pure Python Byte Pair Encoding tokenizer implementation in Lucid.
It supports vocabulary training from text iterators, token/id conversion, and pretrained-style
save/load flows using `vocab.json` and `merges.txt`.

Class Signature
---------------

.. code-block:: python

    class BPETokenizer(Tokenizer):
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

    from lucid.data.tokenizers import BPETokenizer

    texts = [
        "Machine learning helps us build useful systems.",
        "Tokenization quality strongly affects language model performance.",
    ]

    tokenizer = BPETokenizer.train_from_iterator(
        texts=texts,
        vocab_size=1000,
        min_frequency=2,
    )

    ids = tokenizer.encode("Machine learning improves tokenization quality.")
    text = tokenizer.decode(ids)

    tokenizer.save_pretrained(".data/bpe/pretrained")
    tokenizer2 = BPETokenizer.from_pretrained(".data/bpe/pretrained")

.. note::

    - `BPETokenizer` is currently the Python implementation (no C++ fast backend yet).
    - Saved tokenizer directories are expected to contain:
      `vocab.json`, `merges.txt`, and `tokenizer_config.json`.
    - `from_pretrained` accepts either a directory path or a direct path to `vocab.json`.
