GPT2Config
==========

.. autoclass:: lucid.models.GPT2Config

`GPT2Config` extends `GPTConfig` with GPT-2-specific defaults:
a larger vocabulary (50257), and a longer context window (1024). 
All other fields are inherited from `GPTConfig`.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class GPT2Config(GPTConfig):
        vocab_size: int
        max_position_embeddings: int
        hidden_act: str

Parameters
----------
- **vocab_size** (*int*): Vocabulary size. Default is 50257.
- **max_position_embeddings** (*int*): Maximum supported sequence length. Default is 1024.
- **hidden_act** (*str*): Activation used in feed-forward layers. Default is `"gelu"`.

All other parameters are inherited from `GPTConfig` with identical defaults.

Preset Constructors
-------------------

`GPT2Config` provides class methods for all four GPT-2 size variants:

- **GPT2Config.small(...)**: 117M params — hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072.
- **GPT2Config.medium(...)**: 345M params — hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096.
- **GPT2Config.large(...)**: 774M params — hidden_size=1280, num_hidden_layers=36, num_attention_heads=20, intermediate_size=5120.
- **GPT2Config.xl(...)**: 1558M params — hidden_size=1600, num_hidden_layers=48, num_attention_heads=25, intermediate_size=6400.

Keyword arguments are forwarded to override any default field.

Basic Usage
-----------

.. code-block:: python

    from lucid.models import GPT2Config

    config = GPT2Config.small()
    large  = GPT2Config.large(hidden_dropout_prob=0.05)
    custom = GPT2Config(vocab_size=32000, hidden_size=512, num_hidden_layers=6,
                        num_attention_heads=8, intermediate_size=2048)
