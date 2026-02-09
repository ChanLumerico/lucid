bert_for_pre_training_large
===========================

.. autofunction:: lucid.models.bert_for_pre_training_large

The `bert_for_pre_training_large` function creates a BERT-Large model configured for
joint MLM + NSP pre-training.

**Total Parameters**: 336,226,108

Function Signature
------------------

.. code-block:: python

    @register_model
    def bert_for_pre_training_large(
        vocab_size: int = 30522,
        **kwargs,
    ) -> BERTForPreTraining

Parameters
----------
- **vocab_size** (*int*, optional): Vocabulary size. Default is 30,522.
- **kwargs**: Additional keyword arguments forwarded to BERT configuration.

Returns
-------
- **BERTForPreTraining**:
  A pre-training model with MLM and NSP heads.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.bert_for_pre_training_large()
    >>> print(model)
    BERTForPreTraining(...)
