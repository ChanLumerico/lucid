BERTConfig
==========

.. autoclass:: lucid.models.BERTConfig

The `BERTConfig` dataclass stores model hyperparameters used to build
BERT backbones and task-specific wrappers.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class BERTConfig:
        vocab_size: int
        hidden_size: int
        num_attention_heads: int
        num_hidden_layers: int
        intermediate_size: int
        hidden_act: Callable[[Tensor], Tensor]
        hidden_dropout_prob: float
        attention_probs_dropout_prob: float
        max_position_embeddings: int
        tie_word_embedding: bool
        type_vocab_size: int
        initializer_range: float
        layer_norm_eps: float
        use_cache: bool
        is_decoder: bool
        add_cross_attention: bool
        chunk_size_feed_forward: int
        pad_token_id: int = 0
        bos_token_id: int | None = None
        eos_token_id: int | None = None
        classifier_dropout: float | None = None
        add_pooling_layer: bool = True

Parameters
----------
- **vocab_size** (*int*): Vocabulary size.
- **hidden_size** (*int*): Hidden dimension of token states.
- **num_attention_heads** (*int*): Number of attention heads.
- **num_hidden_layers** (*int*): Number of Transformer blocks.
- **intermediate_size** (*int*): Feed-forward inner dimension.
- **hidden_act** (*Callable*): Activation used in feed-forward layers.
- **hidden_dropout_prob** (*float*): Dropout probability in hidden layers.
- **attention_probs_dropout_prob** (*float*): Dropout for attention probabilities.
- **max_position_embeddings** (*int*): Maximum supported sequence length.
- **tie_word_embedding** (*bool*): Whether to tie input/output token embeddings.
- **type_vocab_size** (*int*): Token-type embedding size.
- **initializer_range** (*float*): Std for weight initialization.
- **layer_norm_eps** (*float*): Epsilon for layer normalization.
- **use_cache** (*bool*): Whether caching is enabled for decoder usage.
- **is_decoder** (*bool*): Whether to run as decoder.
- **add_cross_attention** (*bool*): Whether to enable cross-attention blocks.
- **chunk_size_feed_forward** (*int*): Chunk size for feed-forward computation.
- **pad_token_id** (*int*, optional): Padding token id. Default is 0.
- **bos_token_id** (*int | None*, optional): Beginning-of-sequence token id.
- **eos_token_id** (*int | None*, optional): End-of-sequence token id.
- **classifier_dropout** (*float | None*, optional): Dropout for classification heads.
- **add_pooling_layer** (*bool*, optional): Whether to add BERT pooler. Default is True.
