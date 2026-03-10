ViTConfig
=========

.. autoclass:: lucid.models.ViTConfig

`ViTConfig` stores the patch embedding and transformer encoder settings used by
:class:`lucid.models.ViT`. It defines the input image size, patch size,
embedding width, encoder depth, attention heads, MLP width, and dropout.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class ViTConfig:
        image_size: int = 224
        patch_size: int = 16
        in_channels: int = 3
        num_classes: int = 1000
        embedding_dim: int = 768
        depth: int = 12
        num_heads: int = 12
        mlp_dim: int = 3072
        dropout_rate: float = 0.1

Parameters
----------

- **image_size** (*int*):
  Input image size. Vision Transformer assumes square inputs.
- **patch_size** (*int*):
  Patch size used by the strided patch embedding convolution.
- **in_channels** (*int*):
  Number of input image channels.
- **num_classes** (*int*):
  Number of output classes.
- **embedding_dim** (*int*):
  Transformer token embedding width.
- **depth** (*int*):
  Number of encoder layers.
- **num_heads** (*int*):
  Number of attention heads in each encoder layer.
- **mlp_dim** (*int*):
  Hidden width of the feedforward block inside each encoder layer.
- **dropout_rate** (*float*):
  Dropout applied to token embeddings and encoder layers.

Validation
----------

- `image_size`, `patch_size`, `in_channels`, `num_classes`, `embedding_dim`,
  `depth`, `num_heads`, and `mlp_dim` must all be greater than `0`.
- `image_size` must be divisible by `patch_size`.
- `embedding_dim` must be divisible by `num_heads`.
- `dropout_rate` must be in the range `[0, 1)`.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.ViTConfig(
        image_size=32,
        patch_size=8,
        in_channels=1,
        num_classes=10,
        embedding_dim=64,
        depth=2,
        num_heads=4,
        mlp_dim=128,
        dropout_rate=0.0,
    )
    model = models.ViT(config)
