BERTForPreTraining
==================

.. autoclass:: lucid.models.BERTForPreTraining

The `BERTForPreTraining` class combines masked language modeling and
next-sentence prediction heads on top of a BERT backbone.

Class Signature
---------------

.. code-block:: python

    class BERTForPreTraining(config: BERTConfig)

Parameters
----------
- **config** (*BERTConfig*): BERT configuration for encoder + pooled output.

Methods
-------

.. automethod:: lucid.models.BERTForPreTraining.forward

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.bert_for_pre_training_base()
    >>> print(model)
    BERTForPreTraining(...)
