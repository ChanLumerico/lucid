bert_for_question_answering_base
================================

.. autofunction:: lucid.models.bert_for_question_answering_base

The `bert_for_question_answering_base` function creates a BERT-Base model for
extractive question answering with span logits.

**Total Parameters**: 108,893,186

Function Signature
------------------

.. code-block:: python

    @register_model
    def bert_for_question_answering_base(
        vocab_size: int = 30522,
        **kwargs,
    ) -> BERTForQuestionAnswering

Parameters
----------
- **vocab_size** (*int*, optional): Vocabulary size. Default is 30,522.
- **kwargs**: Additional keyword arguments forwarded to BERT configuration.

Returns
-------
- **BERTForQuestionAnswering**:
  A question-answering model returning start/end span logits.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.bert_for_question_answering_base()
    >>> print(model)
    BERTForQuestionAnswering(...)
