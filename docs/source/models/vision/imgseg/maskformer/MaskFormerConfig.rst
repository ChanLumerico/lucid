MaskFormerConfig
================

.. autoclass:: lucid.models.MaskFormerConfig

`MaskFormerConfig` stores the full model setup used by :class:`lucid.models.MaskFormer`.
It includes output space, backbone, decoder shape, and loss-related coefficients.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class MaskFormerConfig:
        num_labels: int
        fpn_feature_size: int
        mask_feature_size: int
        backbone_config: dict | None = None
        num_channels: int = 3
        num_queries: int = 100
        encoder_layer: int = 6
        encoder_ffn_dim: int = 2048
        encoder_attention_heads: int = 8
        decoder_config: dict | None = None
        decoder_layers: int = 6
        decoder_ffn_dim: int = 2048
        decoder_attention_heads: int = 8
        decoder_hidden_size: int | None = None
        decoder_num_queries: int | None = None
        encoder_layerdrop: float = 0.0
        decoder_layerdrop: float = 0.0
        is_encoder_decoder: bool = True
        activation_function: str = "relu"
        d_model: int = 256
        dropout: float = 0.1
        attention_dropout: float = 0.1
        activation_dropout: float = 0.0
        init_std: float = 0.02
        init_xavier_std: float = 1.0
        dilation: bool = False
        class_cost: float = 1.0
        mask_loss_coefficient: float = 1.0
        dice_loss_coefficient: float = 1.0
        eos_coefficient: float = 0.1
        no_object_weight: float = 0.1
        output_attentions: bool = False
        output_hidden_states: bool = False

Parameters
----------
- **num_labels** (*int*): Number of semantic classes (foreground categories).
- **fpn_feature_size** (*int*): Pyramid feature channel width.
- **mask_feature_size** (*int*): Hidden width for mask embedding MLP head.
- **backbone_config** (*dict | None*): Backbone metadata (`model_type`, `depths`, `hidden_sizes`).
- **num_channels** (*int*): Input channel count.
- **num_queries** (*int*): Number of segmentation queries.
- **encoder_layer** (*int*): Number of encoder blocks.
- **encoder_ffn_dim** (*int*): Encoder MLP hidden width.
- **encoder_attention_heads** (*int*): Encoder attention heads.
- **decoder_config** (*dict | None*): Decoder preset config (DETR-style).
- **decoder_layers** (*int*): Number of decoder layers.
- **decoder_ffn_dim** (*int*): Decoder MLP hidden width.
- **decoder_attention_heads** (*int*): Decoder attention heads.
- **decoder_hidden_size** (*int | None*): Optional decoder hidden size override.
- **decoder_num_queries** (*int | None*): Optional query count override.
- **encoder_layerdrop** (*float*): Layer drop probability for encoder.
- **decoder_layerdrop** (*float*): Layer drop probability for decoder.
- **is_encoder_decoder** (*bool*): If True, treats model as encoder-decoder style.
- **activation_function** (*str*): Activation function name.
- **d_model** (*int*): Transformer model width.
- **dropout** (*float*): Dropout probability.
- **attention_dropout** (*float*): Attention dropout probability.
- **activation_dropout** (*float*): Activation dropout probability.
- **init_std** (*float*): Normal init standard deviation.
- **init_xavier_std** (*float*): Xavier gain.
- **dilation** (*bool*): Dilated backbone option (reserved in this implementation).
- **class_cost** (*float*): Weight for classification loss in Hungarian matching.
- **mask_loss_coefficient** (*float*): Weight for mask loss term.
- **dice_loss_coefficient** (*float*): Weight for dice loss term.
- **eos_coefficient** (*float*): Weight for end-of-object class.
- **no_object_weight** (*float*): No-object weight for matcher/class losses.
- **output_attentions** (*bool*): Whether to return attention maps.
- **output_hidden_states** (*bool*): Whether to return hidden states.

Usage
-----

.. code-block:: python

    import lucid.models as models

    cfg = models.MaskFormerConfig(
        num_labels=150,
        fpn_feature_size=256,
        mask_feature_size=256,
        backbone_config={"model_type": "resnet", "depths": [3, 4, 6, 3], "hidden_sizes": [256, 512, 1024, 2048]},
        num_queries=100,
        encoder_layer=6,
        decoder_layers=6,
        decoder_attention_heads=8,
    )

    model = models.MaskFormer(cfg)
