GPTTokenizerFast
================
|cpp-badge|

.. autoclass:: lucid.models.GPTTokenizerFast

`GPTTokenizerFast` is the high-performance GPT-1 tokenizer in Lucid.
It wraps the native C++ BPE backend and follows the original GPT-1 tokenization
scheme: character-level BPE with an `</w>` end-of-word suffix and a single
`<|endoftext|>` token used as the document boundary delimiter.

Class Signature
---------------

.. code-block:: python

    class GPTTokenizerFast(BPETokenizerFast):
        def __init__(
            self,
            vocab: dict[str, int] | None = None,
            merges: list[tuple[str, str]] | None = None,
            vocab_file: Path | str | None = None,
            merges_file: Path | str | None = None,
            unk_token: SpecialTokens | str = SpecialTokens.UNK,
            pad_token: SpecialTokens | str = SpecialTokens.PAD,
            eot_token: str = "<|endoftext|>",
            lowercase: bool = True,
            clean_text: bool = True,
            end_of_word_suffix: str = "</w>",
        ) -> None

Typical Usage
-------------

.. code-block:: python

    from lucid.models import GPTTokenizerFast

    tokenizer = GPTTokenizerFast.from_pretrained("path/to/gpt_tokenizer")

    # Single-sentence encoding
    encoded = tokenizer.encode_plus("The GPT model learns by predicting the next word.")
    print(encoded["input_ids"])
    print(encoded["attention_mask"])

    # Plain encode / decode
    ids = tokenizer.encode("Language models are powerful.")
    text = tokenizer.decode(ids)

    # Train from scratch
    corpus = ["The quick brown fox.", "GPT learns autoregressive language modeling."]
    tokenizer = GPTTokenizerFast.train_from_iterator(corpus, vocab_size=1000)
    tokenizer.save_pretrained("./my_gpt_tokenizer")

Methods
-------

.. automethod:: lucid.models.GPTTokenizerFast.encode_plus
.. automethod:: lucid.models.GPTTokenizerFast.encode
.. automethod:: lucid.models.GPTTokenizerFast.decode
.. automethod:: lucid.models.GPTTokenizerFast.save_pretrained
.. automethod:: lucid.models.GPTTokenizerFast.from_pretrained

.. note::

    - This class requires compiled C++ extensions under `lucid._backend._C`.
    - `encode_plus` returns a dict with `input_ids` and `attention_mask` only —
      there is no `token_type_ids` because GPT is a decoder-only model with no
      next-sentence-prediction objective.
    - `eot_token` (`<|endoftext|>`) serves as the document boundary delimiter
      and is registered as both `bos_token` and `eos_token` internally.
      It is appended to each sequence when `add_special_tokens=True`.
    - `build_inputs_with_special_tokens` returns the token list as-is (no per-sample
      BOS prepended), consistent with GPT-1 pretraining where `eot_token` only marks
      document boundaries.
    - Vocabulary artifacts are saved as `vocab.json` + `merges.txt` +
      `tokenizer_config.json`, identical to the `BPETokenizerFast` format.
