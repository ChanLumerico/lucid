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
   :no-index:

Compute NSP logits from pooled BERT outputs.

.. automethod:: lucid.models.BERTForNextSentencePrediction.get_loss
   :no-index:

Compute next-sentence prediction classification loss.

.. automethod:: lucid.models.BERTForNextSentencePrediction.predict_labels
   :no-index:

Return predicted NSP labels by argmax over logits.

.. automethod:: lucid.models.BERTForNextSentencePrediction.predict_proba
   :no-index:

Return NSP class probabilities via softmax.

.. automethod:: lucid.models.BERTForNextSentencePrediction.get_accuracy
   :no-index:

Compute NSP classification accuracy.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.bert_for_next_sentence_prediction_base()
    >>> print(model)
    BERTForNextSentencePrediction(...)

.. code-block:: python

    >>> logits = model(input_ids=input_ids)
    >>> loss = model.get_loss(labels=labels, input_ids=input_ids)
    >>> acc = model.get_accuracy(labels=labels, input_ids=input_ids)
