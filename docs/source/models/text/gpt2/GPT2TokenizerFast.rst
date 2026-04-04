GPT2TokenizerFast
=================
|cpp-badge|

.. autoclass:: lucid.models.GPT2TokenizerFast

`GPT2TokenizerFast` is the high-performance GPT-2 tokenizer in Lucid.
It wraps the native C++ Byte-level BPE backend, operating on raw UTF-8 bytes
rather than characters. This eliminates unknown tokens for any Unicode input and
matches the tokenization scheme of the original GPT-2 release.

Class Signature
---------------

.. code-block:: python

    class GPT2TokenizerFast(ByteBPETokenizerFast):
        def __init__(
            self,
            vocab: dict[str, int] | None = None,
            merges: list[tuple[str, str]] | None = None,
            vocab_file: Path | str | None = None,
            merges_file: Path | str | None = None,
            unk_token: SpecialTokens | str = SpecialTokens.UNK,
            pad_token: SpecialTokens | str = SpecialTokens.PAD,
            eot_token: str = "<|endoftext|>",
            lowercase: bool = False,
            clean_text: bool = True,
            add_prefix_space: bool = False,
        ) -> None

Typical Usage
-------------

.. code-block:: python

    from lucid.models import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("path/to/gpt2_tokenizer")

    # Single-sentence encoding
    encoded = tokenizer.encode_plus("GPT-2 uses byte-level BPE tokenization.")
    print(encoded["input_ids"])
    print(encoded["attention_mask"])

    # Plain encode / decode
    ids = tokenizer.encode("Language models are powerful.")
    text = tokenizer.decode(ids)

    # Train from scratch
    corpus = ["The quick brown fox.", "GPT-2 operates on raw bytes."]
    tokenizer = GPT2TokenizerFast.train_from_iterator(corpus, vocab_size=1000)
    tokenizer.save_pretrained("./my_gpt2_tokenizer")

Methods
-------

.. automethod:: lucid.models.GPT2TokenizerFast.encode_plus
.. automethod:: lucid.models.GPT2TokenizerFast.encode
.. automethod:: lucid.models.GPT2TokenizerFast.decode
.. automethod:: lucid.models.GPT2TokenizerFast.save_pretrained
.. automethod:: lucid.models.GPT2TokenizerFast.from_pretrained

.. note::

    - This class requires compiled C++ extensions under `lucid._backend._C`.
    - Unlike `GPTTokenizerFast` (character-level BPE), this tokenizer operates on
      raw UTF-8 bytes, so it never produces unknown tokens for valid Unicode input.
    - `lowercase` defaults to `False` — GPT-2 preserves casing.
    - `eot_token` (`<|endoftext|>`) is registered as both `bos_token` and
      `eos_token` internally, and appended when `add_special_tokens=True`.
    - `encode_plus` returns `input_ids` and `attention_mask` only;
      there is no `token_type_ids`.
    - Vocabulary artifacts are saved as `vocab.json` + `merges.txt` +
      `tokenizer_config.json`.
