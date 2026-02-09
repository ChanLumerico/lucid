BERTForMaskedLM
===============

.. autoclass:: lucid.models.BERTForMaskedLM

The `BERTForMaskedLM` class attaches a masked language modeling head
to the BERT backbone.

Class Signature
---------------

.. code-block:: python

    class BERTForMaskedLM(config: BERTConfig)

Parameters
----------
- **config** (*BERTConfig*): BERT configuration for masked language modeling.

Methods
-------

.. automethod:: lucid.models.BERTForMaskedLM.forward

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.bert_for_masked_lm_base()
    >>> print(model)
    BERTForMaskedLM(...)
