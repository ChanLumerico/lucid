gpt2_small
==========

.. autofunction:: lucid.models.gpt2_small

The `gpt2_small` function creates a `GPT2` instance with the Small configuration,
matching the original GPT-2 Small (117M) release by OpenAI.

**Total Parameters**: 124,439,808

Function Signature
------------------

.. code-block:: python

    @register_model
    def gpt2_small(**kwargs) -> GPT2

Parameters
----------
- **kwargs**:
  Additional keyword arguments forwarded to `GPT2Config.small()` to override
  any default field (e.g. `hidden_dropout_prob`, `vocab_size`).

Returns
-------
- **GPT2**:
  A `GPT2` instance initialized with the Small configuration.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.gpt2_small()
    >>> print(model)
    GPT2(...)

.. code-block:: python

    >>> import lucid
    >>> input_ids = lucid.randint(0, 50257, (1, 32))
    >>> hidden_states, _ = model(input_ids)
    >>> hidden_states.shape
    (1, 32, 768)
