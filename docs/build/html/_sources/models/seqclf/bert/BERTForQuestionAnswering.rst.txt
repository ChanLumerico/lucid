BERTForQuestionAnswering
========================

.. autoclass:: lucid.models.BERTForQuestionAnswering

The `BERTForQuestionAnswering` class predicts start and end logits for
extractive question answering.

Class Signature
---------------

.. code-block:: python

    class BERTForQuestionAnswering(config: BERTConfig)

Parameters
----------
- **config** (*BERTConfig*): BERT configuration for token span prediction.

Methods
-------

.. automethod:: lucid.models.BERTForQuestionAnswering.forward

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.bert_for_question_answering_base()
    >>> print(model)
    BERTForQuestionAnswering(...)
