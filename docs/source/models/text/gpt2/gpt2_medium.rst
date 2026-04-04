gpt2_medium
===========

.. autofunction:: lucid.models.gpt2_medium

The `gpt2_medium` function creates a `GPT2` instance with the Medium configuration,
matching the original GPT-2 Medium (345M) release by OpenAI.

**Total Parameters**: 354,798,592

Function Signature
------------------

.. code-block:: python

    @register_model
    def gpt2_medium(**kwargs) -> GPT2

Parameters
----------
- **kwargs**:
  Additional keyword arguments forwarded to `GPT2Config.medium()` to override
  any default field (e.g. `hidden_dropout_prob`, `vocab_size`).

Returns
-------
- **GPT2**:
  A `GPT2` instance initialized with the Medium configuration.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.gpt2_medium()
    >>> print(model)
    GPT2(...)

.. code-block:: python

    >>> import lucid
    >>> input_ids = lucid.randint(0, 50257, (1, 32))
    >>> hidden_states, _ = model(input_ids)
    >>> hidden_states.shape
    (1, 32, 1024)
