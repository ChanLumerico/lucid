ByteBPETokenizerFast
====================
|cpp-badge|

.. autoclass:: lucid.data.tokenizers.ByteBPETokenizerFast

`ByteBPETokenizerFast` is the high-performance byte-level BPE tokenizer in Lucid.
It follows the same Python-facing interface as `ByteBPETokenizer`, while delegating
core tokenization, merge application, and id conversion to the native C++ backend.

Class Signature
---------------

.. code-block:: python

    class ByteBPETokenizerFast(Tokenizer):
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

    from lucid.data.tokenizers import ByteBPETokenizerFast

    tokenizer = ByteBPETokenizerFast.from_pretrained("some_path")

    ids = tokenizer.encode("Machine learning helps café and 한글.")
    text = tokenizer.decode(ids)

    batch_ids = tokenizer.batch_encode(
        ["hello world", "byte-level bpe fast path"],
        add_special_tokens=True,
    )

.. note::

    - This class requires the compiled C++ extension backend.
    - If the extension is not built, importing or constructing this tokenizer will fail.
