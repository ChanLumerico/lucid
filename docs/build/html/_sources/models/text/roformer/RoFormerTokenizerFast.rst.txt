RoFormerTokenizerFast
=====================
|cpp-badge|

.. autoclass:: lucid.models.RoFormerTokenizerFast

`RoFormerTokenizerFast` is a RoFormer-named tokenizer wrapper that inherits
the same high-performance WordPiece pipeline as `BERTTokenizerFast`.

Class Signature
---------------

.. code-block:: python

    class RoFormerTokenizerFast(BERTTokenizerFast)

Typical Usage
-------------

.. code-block:: python

    from lucid.models import RoFormerTokenizerFast

    tokenizer = RoFormerTokenizerFast.from_pretrained("some_path")
    encoded = tokenizer.encode_plus("RoFormer uses rotary position embedding.")
    print(encoded["input_ids"])
    print(encoded["token_type_ids"])
    print(encoded["attention_mask"])

Methods
-------

.. automethod:: lucid.models.RoFormerTokenizerFast.encode_plus
.. automethod:: lucid.models.RoFormerTokenizerFast.encode_pretraining_inputs
.. automethod:: lucid.models.RoFormerTokenizerFast.encode
.. automethod:: lucid.models.RoFormerTokenizerFast.decode
.. automethod:: lucid.models.RoFormerTokenizerFast.save_pretrained
.. automethod:: lucid.models.RoFormerTokenizerFast.from_pretrained

.. note::

    - This class currently reuses `BERTTokenizerFast` behavior as-is.
    - It accepts the same vocabulary format and special token settings.
    - As with BERT tokenizer usage, MLM masking is handled outside the tokenizer.

