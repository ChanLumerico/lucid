BERTForCausalLM
===============

.. autoclass:: lucid.models.BERTForCausalLM

The `BERTForCausalLM` class configures BERT in decoder-style mode and
applies a causal language modeling head.

Class Signature
---------------

.. code-block:: python

    class BERTForCausalLM(config: BERTConfig)

Parameters
----------
- **config** (*BERTConfig*): BERT configuration with decoder/caching options.

Methods
-------

.. automethod:: lucid.models.BERTForCausalLM.forward

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.bert_for_causal_lm_base()
    >>> print(model)
    BERTForCausalLM(...)
