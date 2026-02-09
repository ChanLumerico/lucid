bert_for_masked_lm_large
========================

.. autofunction:: lucid.models.bert_for_masked_lm_large

The `bert_for_masked_lm_large` function creates a BERT-Large model configured for
masked language modeling.

**Total Parameters**: 335,174,458
**Total FLOPs**: 87.19G

Function Signature
------------------

.. code-block:: python

    @register_model
    def bert_for_masked_lm_large(
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
    >>> model = models.bert_for_masked_lm_large()
    >>> print(model)
    BERTForMaskedLM(...)
