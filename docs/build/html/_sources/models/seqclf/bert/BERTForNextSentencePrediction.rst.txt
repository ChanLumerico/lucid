BERTForNextSentencePrediction
=============================

.. autoclass:: lucid.models.BERTForNextSentencePrediction

The `BERTForNextSentencePrediction` class applies the NSP head over
pooled BERT outputs.

Class Signature
---------------

.. code-block:: python

    class BERTForNextSentencePrediction(config: BERTConfig)

Parameters
----------
- **config** (*BERTConfig*): BERT configuration with pooling enabled.

Methods
-------

.. automethod:: lucid.models.BERTForNextSentencePrediction.forward

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.bert_for_next_sentence_prediction_base()
    >>> print(model)
    BERTForNextSentencePrediction(...)
