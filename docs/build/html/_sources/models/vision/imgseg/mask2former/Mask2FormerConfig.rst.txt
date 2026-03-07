Mask2FormerConfig
=================

.. autoclass:: lucid.models.Mask2FormerConfig

`Mask2FormerConfig` stores the complete model setup used by
:class:`lucid.models.Mask2Former`.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class Mask2FormerConfig:
        num_labels: int
        feature_size: int = 256
        mask_feature_size: int = 256
        hidden_dim: int = 256
        backbone_config: dict | None = None
        num_channels: int = 3
        num_queries: int = 100
        encoder_layers: int = 6
        encoder_feedforward_dim: int = 1024
        decoder_layers: int = 10
        dim_feedforward: int = 2048
        num_attention_heads: int = 8
        feature_strides: list[int] = [4, 8, 16, 32]
        common_stride: int = 4
        enforce_input_projection: bool = False
        activation_function: str = "relu"
        pre_norm: bool = False
        dropout: float = 0.0
        init_std: float = 0.02
        init_xavier_std: float = 1.0
        dilation: bool = False
        class_weight: float = 2.0
        mask_weight: float = 5.0
        dice_weight: float = 5.0
        no_object_weight: float = 0.1
        train_num_points: int = 12544
        oversample_ratio: float = 3.0
        importance_sample_ratio: float = 0.75
        use_auxiliary_loss: bool = True
        output_auxiliary_logits: bool | None = None
        output_attentions: bool = False
        output_hidden_states: bool = False

Usage
-----

.. code-block:: python

    import lucid.models as models

    cfg = models.Mask2FormerConfig(
        num_labels=150,
        hidden_dim=256,
        feature_size=256,
        mask_feature_size=256,
        backbone_config={
            "model_type": "swin",
            "embed_dim": 96,
            "depths": [2, 2, 18, 2],
            "num_heads": [3, 6, 12, 24],
            "image_size": 224,
            "window_size": 7,
        },
    )

    model = models.Mask2Former(cfg)
