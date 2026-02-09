bert_for_pre_training_base
==========================

.. autofunction:: lucid.models.bert_for_pre_training_base

The `bert_for_pre_training_base` function creates a BERT-Base model configured for
joint MLM + NSP pre-training.

**Total Parameters**: 110,106,428
**Total FLOPs**: 28.50G

Function Signature
------------------

.. code-block:: python

    @register_model
    def bert_for_pre_training_base(
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
    >>> model = models.bert_for_pre_training_base()
    >>> print(model)
    BERTForPreTraining(...)
