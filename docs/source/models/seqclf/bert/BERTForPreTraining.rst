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
   :no-index:

Return pretraining outputs as `(prediction_scores, seq_relationship_scores)`,
combining MLM and NSP heads.

.. automethod:: lucid.models.BERTForPreTraining.get_mlm_loss
   :no-index:

Compute the masked language modeling (MLM) loss from token labels.

.. automethod:: lucid.models.BERTForPreTraining.get_nsp_loss
   :no-index:

Compute the next sentence prediction (NSP) loss from sequence labels.

.. automethod:: lucid.models.BERTForPreTraining.get_loss
   :no-index:

Compute the combined pretraining loss as a weighted sum of MLM and NSP losses.

.. automethod:: lucid.models.BERTForPreTraining.create_masked_lm_inputs
   :no-index:

Create BERT-style masked inputs and MLM labels for pretraining batches.

.. automethod:: lucid.models.BERTForPreTraining.predict_mlm_token_ids
   :no-index:

Predict token ids for the MLM branch.

.. automethod:: lucid.models.BERTForPreTraining.predict_nsp_labels
   :no-index:

Predict binary NSP labels.

.. automethod:: lucid.models.BERTForPreTraining.get_mlm_accuracy
   :no-index:

Compute masked-token accuracy for MLM labels.

.. automethod:: lucid.models.BERTForPreTraining.get_nsp_accuracy
   :no-index:

Compute classification accuracy for NSP labels.

.. automethod:: lucid.models.BERTForPreTraining.get_accuracy
   :no-index:

Return `(mlm_accuracy, nsp_accuracy, weighted_accuracy)` for joint pretraining.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.bert_for_pre_training_base()
    >>> print(model)
    BERTForPreTraining(...)

.. code-block:: python

    >>> prediction_scores, seq_relationship_scores = model(
    ...     input_ids=input_ids,
    ...     attention_mask=attention_mask,
    ...     token_type_ids=token_type_ids,
    ... )

.. code-block:: python

    >>> mlm_loss = model.get_mlm_loss(
    ...     mlm_labels=mlm_labels,
    ...     input_ids=input_ids,
    ...     attention_mask=attention_mask,
    ...     token_type_ids=token_type_ids,
    ... )
    >>> nsp_loss = model.get_nsp_loss(
    ...     nsp_labels=nsp_labels,
    ...     input_ids=input_ids,
    ...     attention_mask=attention_mask,
    ...     token_type_ids=token_type_ids,
    ... )
    >>> total_loss = model.get_loss(
    ...     mlm_labels=mlm_labels,
    ...     nsp_labels=nsp_labels,
    ...     input_ids=input_ids,
    ...     attention_mask=attention_mask,
    ...     token_type_ids=token_type_ids,
    ... )

.. code-block:: python

    >>> masked_input_ids, mlm_labels = model.create_masked_lm_inputs(
    ...     input_ids=input_ids,
    ...     attention_mask=attention_mask,
    ... )
    >>> mlm_pred_ids = model.predict_mlm_token_ids(
    ...     input_ids=masked_input_ids,
    ...     attention_mask=attention_mask,
    ...     token_type_ids=token_type_ids,
    ... )
    >>> nsp_pred = model.predict_nsp_labels(
    ...     input_ids=input_ids,
    ...     attention_mask=attention_mask,
    ...     token_type_ids=token_type_ids,
    ... )
    >>> mlm_acc = model.get_mlm_accuracy(
    ...     mlm_labels=mlm_labels,
    ...     input_ids=masked_input_ids,
    ...     attention_mask=attention_mask,
    ...     token_type_ids=token_type_ids,
    ... )
    >>> nsp_acc = model.get_nsp_accuracy(
    ...     nsp_labels=nsp_labels,
    ...     input_ids=input_ids,
    ...     attention_mask=attention_mask,
    ...     token_type_ids=token_type_ids,
    ... )
    >>> mlm_acc, nsp_acc, weighted_acc = model.get_accuracy(
    ...     mlm_labels=mlm_labels,
    ...     nsp_labels=nsp_labels,
    ...     input_ids=masked_input_ids,
    ...     attention_mask=attention_mask,
    ...     token_type_ids=token_type_ids,
    ... )
