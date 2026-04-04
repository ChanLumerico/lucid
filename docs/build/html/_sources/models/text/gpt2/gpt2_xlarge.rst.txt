gpt2_xlarge
===========

.. autofunction:: lucid.models.gpt2_xlarge

The `gpt2_xlarge` function creates a `GPT2` instance with the XL configuration,
matching the original GPT-2 XL (1558M) release by OpenAI.

**Total Parameters**: 1,557,611,200

Function Signature
------------------

.. code-block:: python

    @register_model
    def gpt2_xlarge(**kwargs) -> GPT2

Parameters
----------
- **kwargs**:
  Additional keyword arguments forwarded to `GPT2Config.xl()` to override
  any default field (e.g. `hidden_dropout_prob`, `vocab_size`).

Returns
-------
- **GPT2**:
  A `GPT2` instance initialized with the XL configuration.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.gpt2_xlarge()
    >>> print(model)
    GPT2(...)

.. code-block:: python

    >>> import lucid
    >>> input_ids = lucid.randint(0, 50257, (1, 32))
    >>> hidden_states, _ = model(input_ids)
    >>> hidden_states.shape
    (1, 32, 1600)
