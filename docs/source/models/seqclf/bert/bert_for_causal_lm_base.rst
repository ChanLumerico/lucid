bert_for_causal_lm_base
=======================

.. autofunction:: lucid.models.bert_for_causal_lm_base

The `bert_for_causal_lm_base` function creates a decoder-style BERT-Base model
for causal language modeling.

**Total Parameters**: 109,514,298
**Total FLOPs**: 28.50G

Function Signature
------------------

.. code-block:: python

    @register_model
    def bert_for_causal_lm_base(
        vocab_size: int = 30522,
        **kwargs,
    ) -> BERTForCausalLM

Parameters
----------
- **vocab_size** (*int*, optional): Vocabulary size. Default is 30,522.
- **kwargs**: Additional keyword arguments forwarded to BERT configuration.

Returns
-------
- **BERTForCausalLM**:
  A causal language modeling model with decoder behavior.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.bert_for_causal_lm_base()
    >>> print(model)
    BERTForCausalLM(...)
