GPTConfig
=========

.. autoclass:: lucid.models.GPTConfig

The `GPTConfig` dataclass stores model hyperparameters used to build
the GPT backbone and task-specific wrappers.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class GPTConfig:
        vocab_size: int
        max_position_embeddings: int
        hidden_size: int
        num_attention_heads: int
        num_hidden_layers: int
        intermediate_size: int
        hidden_act: str
        hidden_dropout_prob: float
        attention_prob_dropout_prob: float
        layer_norm_eps: float
        initializer_range: float
        use_cache: bool
        bos_token_id: int
        eos_token_id: int
        pad_token_id: int | None

Parameters
----------
- **vocab_size** (*int*): Vocabulary size. Default is 40478.
- **max_position_embeddings** (*int*): Maximum supported sequence length. Default is 512.
- **hidden_size** (*int*): Hidden dimension of token states. Default is 768.
- **num_attention_heads** (*int*): Number of attention heads. Default is 12.
- **num_hidden_layers** (*int*): Number of Transformer blocks. Default is 12.
- **intermediate_size** (*int*): Feed-forward inner dimension. Default is 3072.
- **hidden_act** (*str*): Activation used in feed-forward layers. Default is ``"gelu"``.
- **hidden_dropout_prob** (*float*): Dropout probability in hidden layers. Default is 0.1.
- **attention_prob_dropout_prob** (*float*): Dropout for attention probabilities. Default is 0.1.
- **layer_norm_eps** (*float*): Epsilon for layer normalization. Default is 1e-5.
- **initializer_range** (*float*): Std for weight initialization. Default is 0.02.
- **use_cache** (*bool*): Whether KV-cache is enabled for generation. Default is ``True``.
- **bos_token_id** (*int*): Beginning-of-sequence token id. Default is 40476.
- **eos_token_id** (*int*): End-of-sequence token id. Default is 40477.
- **pad_token_id** (*int | None*, optional): Padding token id. Default is ``None``.

Preset Constructors
-------------------

`GPTConfig` provides a class method for the GPT-1 original configuration:

- **GPTConfig.base(...)**: Returns the GPT-1 base config
  (hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072).

Keyword arguments are forwarded to override any default field.

Basic Usage
-----------

.. code-block:: python

    from lucid.models import GPTConfig

    config = GPTConfig.base()
    custom = GPTConfig.base(vocab_size=30000, hidden_dropout_prob=0.2)
