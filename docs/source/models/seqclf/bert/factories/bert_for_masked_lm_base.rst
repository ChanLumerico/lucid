bert_for_masked_lm_base
=======================

.. autofunction:: lucid.models.bert_for_masked_lm_base

The `bert_for_masked_lm_base` function creates a BERT-Base model configured for
masked language modeling.

**Total Parameters**: 109,514,298

Function Signature
------------------

.. code-block:: python

    @register_model
    def bert_for_masked_lm_base(
        vocab_size: int = 30522,
        **kwargs,
    ) -> BERTForMaskedLM

Parameters
----------
- **vocab_size** (*int*, optional): Vocabulary size. Default is 30,522.
- **kwargs**: Additional keyword arguments forwarded to BERT configuration.

Returns
-------
- **BERTForMaskedLM**:
  A masked language modeling model.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.bert_for_masked_lm_base()
    >>> print(model)
    BERTForMaskedLM(...)
