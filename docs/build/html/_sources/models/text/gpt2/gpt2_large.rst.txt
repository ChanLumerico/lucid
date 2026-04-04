gpt2_large
==========

.. autofunction:: lucid.models.gpt2_large

The `gpt2_large` function creates a `GPT2` instance with the Large configuration,
matching the original GPT-2 Large (774M) release by OpenAI.

**Total Parameters**: 774,030,080

Function Signature
------------------

.. code-block:: python

    @register_model
    def gpt2_large(**kwargs) -> GPT2

Parameters
----------
- **kwargs**:
  Additional keyword arguments forwarded to `GPT2Config.large()` to override
  any default field (e.g. `hidden_dropout_prob`, `vocab_size`).

Returns
-------
- **GPT2**:
  A `GPT2` instance initialized with the Large configuration.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.gpt2_large()
    >>> print(model)
    GPT2(...)

.. code-block:: python

    >>> import lucid
    >>> input_ids = lucid.randint(0, 50257, (1, 32))
    >>> hidden_states, _ = model(input_ids)
    >>> hidden_states.shape
    (1, 32, 1280)
