BERTTokenizerFast
=================
|cpp-badge|

.. autoclass:: lucid.models.BERTTokenizerFast

`BERTTokenizerFast` is the high-performance BERT tokenizer in Lucid.
It follows a Hugging Face-style BERT tokenization flow based on WordPiece and
uses a native C++ backend for fast tokenization and id conversion.

Class Signature
---------------

.. code-block:: python

    class BERTTokenizerFast(Tokenizer):
        def __init__(
            self,
            vocab: dict[str, int] | None = None,
            vocab_file: Path | str | None = None,
            unk_token: SpecialTokens | str = SpecialTokens.UNK,
            pad_token: SpecialTokens | str = SpecialTokens.PAD,
            cls_token: str = "[CLS]",
            mask_token: str = "[MASK]",
            lowercase: bool = True,
            wordpieces_prefix: str = "##",
            max_input_chars_per_word: int = 100,
            clean_text: bool = True,
        ) -> None

Typical Usage
-------------

.. code-block:: python

    from lucid.models import BERTTokenizerFast

    tokenizer = BERTTokenizerFast.from_pretrained(".data/bert/pretrained")

    single = tokenizer.encode_plus("Hello world!")
    print(single["input_ids"])
    print(single["token_type_ids"])
    print(single["attention_mask"])

    pair = tokenizer.encode_plus("Hello world!", "How are you?")
    print(pair["input_ids"])
    print(pair["token_type_ids"])
    print(pair["attention_mask"])

    ids = tokenizer.encode("A simple sentence.")
    text = tokenizer.decode(ids)

Methods
-------

.. automethod:: lucid.models.BERTTokenizerFast.encode_plus
.. automethod:: lucid.models.BERTTokenizerFast.encode_pretraining_inputs
.. automethod:: lucid.models.BERTTokenizerFast.encode
.. automethod:: lucid.models.BERTTokenizerFast.decode
.. automethod:: lucid.models.BERTTokenizerFast.save_pretrained
.. automethod:: lucid.models.BERTTokenizerFast.from_pretrained

.. note::

    - This class requires compiled C++ extensions under `lucid._backend._C`.
    - `encode_plus` returns a dict with `input_ids`, `token_type_ids`,
      and `attention_mask`.
    - `encode_pretraining_inputs` also returns `special_tokens_mask`, which is
      directly usable for MLM masking preprocessing.
    - This tokenizer does not apply MLM masking (e.g., 15% masking); masking is
      typically handled in data-collation or training preprocessing.
