NCSNConfig
==========

.. autoclass:: lucid.models.NCSNConfig

`NCSNConfig` stores the RefineNet width, conditional noise schedule settings,
and optional initial sigma buffer used by :class:`lucid.models.NCSN`.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class NCSNConfig:
        in_channels: int = 3
        nf: int = 128
        num_classes: int = 10
        dilations: Sequence[int] = (1, 2, 4, 8)
        scale_by_sigma: bool = True
        sigmas: Tensor | Sequence[float] | None = None

Parameters
----------

- **in_channels** (*int*):
  Number of image channels.
- **nf** (*int*):
  Base feature width of the RefineNet-style backbone.
- **num_classes** (*int*):
  Number of noise levels and conditional normalization labels.
- **dilations** (*Sequence[int]*):
  Four dilation values used by the RCU stages.
- **scale_by_sigma** (*bool*):
  Whether the output score is divided by the selected sigma.
- **sigmas** (*Tensor | Sequence[float] | None*):
  Optional sigma schedule to preload into the registered `sigmas` buffer.

Validation
----------

- `in_channels`, `nf`, and `num_classes` must be greater than `0`.
- `dilations` must contain exactly four positive integers.
- `scale_by_sigma` must be a boolean.
- `sigmas`, when provided, must be 1D, positive, and have length `num_classes`.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.NCSNConfig(
        in_channels=3,
        nf=128,
        num_classes=10,
        sigmas=models.NCSN.make_sigmas(50.0, 0.01, 10),
    )
    model = models.NCSN(config)
