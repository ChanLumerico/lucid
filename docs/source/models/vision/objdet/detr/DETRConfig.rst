DETRConfig
==========

.. autoclass:: lucid.models.DETRConfig

`DETRConfig` stores the backbone, transformer, query count, and loss settings
used to build :class:`lucid.models.DETR`.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class DETRConfig:
        backbone: nn.Module
        transformer: nn.Module
        num_classes: int
        num_queries: int = 100
        aux_loss: bool = True
        matcher: nn.Module | None = None
        class_loss_coef: float = 1.0
        bbox_loss_coef: float = 5.0
        giou_loss_coef: float = 2.0
        eos_coef: float = 0.1

Parameters
----------

- **backbone** (`nn.Module`):
  Feature extractor module. It must define a positive integer `num_channels`
  attribute used by the `1x1` input projection.
- **transformer** (`nn.Module`):
  DETR encoder-decoder module. It must define a positive integer `d_model`
  attribute used by positional encoding, query embeddings, and prediction heads.
- **num_classes** (`int`):
  Number of foreground classes.
- **num_queries** (`int`):
  Number of learned object queries.
- **aux_loss** (`bool`):
  Whether intermediate decoder outputs should contribute auxiliary losses.
- **matcher** (`nn.Module | None`):
  Optional Hungarian matcher override. If omitted, the standard matcher is built
  from the configured loss coefficients.
- **class_loss_coef** (`float`):
  Classification loss weight.
- **bbox_loss_coef** (`float`):
  Box L1 loss weight.
- **giou_loss_coef** (`float`):
  Generalized IoU loss weight.
- **eos_coef** (`float`):
  Weight for the "no-object" class in cross-entropy.

Validation
----------

- `backbone` must be an `nn.Module` with positive integer `num_channels`.
- `transformer` must be an `nn.Module` with positive integer `d_model`.
- `num_classes` and `num_queries` must be greater than `0`.
- `aux_loss` must be a boolean.
- `matcher` must be an `nn.Module` or `None`.
- `class_loss_coef`, `bbox_loss_coef`, and `giou_loss_coef` must be non-negative.
- `eos_coef` must be in `[0, 1]`.

Usage
-----

.. code-block:: python

    import lucid.models as models
    from lucid.models.vision.detr import _BackboneBase, _Transformer

    config = models.DETRConfig(
        backbone=_BackboneBase(models.resnet_50()),
        transformer=_Transformer(d_model=256, n_head=8),
        num_classes=91,
        num_queries=100,
    )
    model = models.DETR(config)
