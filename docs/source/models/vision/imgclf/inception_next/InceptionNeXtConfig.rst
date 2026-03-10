InceptionNeXtConfig
===================

.. autoclass:: lucid.models.InceptionNeXtConfig

`InceptionNeXtConfig` stores the stage layout and classifier settings used by
:class:`lucid.models.InceptionNeXt`. It defines the per-stage depth profile,
stage widths, token mixers, MLP ratios, and regularization values.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class InceptionNeXtConfig:
        num_classes: int = 1000
        depths: tuple[int, ...] | list[int] = (3, 3, 9, 3)
        dims: tuple[int, ...] | list[int] = (96, 192, 384, 768)
        token_mixers: object | None = None
        mlp_ratios: int | tuple[int, ...] | list[int] = (4, 4, 4, 3)
        head_fn: object | None = None
        drop_rate: float = 0.0
        drop_path_rate: float = 0.0
        ls_init_value: float = 1e-6

Parameters
----------

- **num_classes** (*int*):
  Number of output classes.
- **depths**:
  Per-stage block counts for the InceptionNeXt hierarchy.
- **dims**:
  Per-stage channel widths for the InceptionNeXt hierarchy.
- **token_mixers**:
  Callable token mixer, or sequence of token mixer callables, used by each stage.
- **mlp_ratios**:
  Positive MLP expansion ratio, or per-stage sequence of expansion ratios.
- **head_fn**:
  Callable used to construct the classifier head.
- **drop_rate** (*float*):
  Dropout probability used by the classifier head.
- **drop_path_rate** (*float*):
  Global drop-path rate distributed across all blocks.
- **ls_init_value** (*float*):
  Initial layer-scale value used inside each block.

Validation
----------

- `num_classes` must be greater than 0.
- `depths` and `dims` must be non-empty sequences of positive integers with the same length.
- `token_mixers` must be a callable or a same-length sequence of callables.
- `mlp_ratios` must be a positive integer or a same-length sequence of positive integers.
- `head_fn` must be callable.
- `drop_rate` must be in the range `[0, 1)`.
- `drop_path_rate` must be in the range `[0, 1]`.
- `ls_init_value` must be greater than or equal to 0.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.InceptionNeXtConfig(
        num_classes=10,
        depths=(2, 2, 6, 2),
        dims=(40, 80, 160, 320),
        mlp_ratios=2,
        drop_path_rate=0.1,
    )
    model = models.InceptionNeXt(config)
