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
   :no-index:

Compute start and end logits for extractive answer span prediction.

.. automethod:: lucid.models.BERTForQuestionAnswering.get_loss
   :no-index:

Compute QA training loss from start and end target positions.

.. automethod:: lucid.models.BERTForQuestionAnswering.predict_spans
   :no-index:

Return argmax start and end indices per sample.

.. automethod:: lucid.models.BERTForQuestionAnswering.get_best_spans
   :no-index:

Return best start/end span candidates with scores under a max answer length.

.. automethod:: lucid.models.BERTForQuestionAnswering.get_accuracy
   :no-index:

Compute exact-match span accuracy (both start and end must match).

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.bert_for_question_answering_base()
    >>> print(model)
    BERTForQuestionAnswering(...)

.. code-block:: python

    >>> start_logits, end_logits = model(input_ids=input_ids, attention_mask=attention_mask)
    >>> loss = model.get_loss(start_positions, end_positions, input_ids=input_ids, attention_mask=attention_mask)
    >>> best_start, best_end, best_score = model.get_best_spans(input_ids=input_ids, attention_mask=attention_mask)
