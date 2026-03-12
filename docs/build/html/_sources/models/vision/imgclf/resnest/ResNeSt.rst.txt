ResNeSt
=======

.. toctree::
    :maxdepth: 1
    :hidden:

    ResNeStConfig.rst
    resnest_14.rst
    resnest_26.rst
    resnest_50.rst
    resnest_101.rst
    resnest_200.rst
    resnest_269.rst
    resnest_50_4s2x40d.rst
    resnest_50_1s4x24d.rst

|convnet-badge|

.. autoclass:: lucid.models.ResNeSt

The `ResNeSt` class extends the residual bottleneck design with split-attention
convolutions and a deep stem. Its structure is described by `ResNeStConfig`,
which captures the stage depths together with radix, cardinality, base width,
and other backbone options used by the ResNeSt family.

Class Signature
---------------

.. code-block:: python

    class ResNeSt(ResNet):
        def __init__(self, config: ResNeStConfig)

Parameters
----------

- **config** (*ResNeStConfig*):
  Configuration object describing the stage depths, split-attention settings,
  classifier size, deep-stem width, and shared residual-stage options.

Attributes
----------

- **config** (*ResNeStConfig*):
  The configuration used to construct the model.
- **base_width** (*int*):
  Width parameter used to compute grouped bottleneck channels.
- **stem_width** (*int*):
  Width of the deep stem.
- **cardinality** (*int*):
  Number of groups in split-attention bottlenecks.
- **radix** (*int*):
  Number of attention splits per grouped bottleneck.
- **avd** (*bool*):
  Whether anti-aliasing average downsampling is enabled inside the bottleneck.

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> import lucid.models as models
    >>> config = models.ResNeStConfig(
    ...     layers=[3, 4, 6, 3],
    ...     base_width=40,
    ...     stem_width=32,
    ...     cardinality=2,
    ...     radix=4,
    ...     num_classes=10,
    ... )
    >>> model = models.ResNeSt(config)
    >>> output = model(lucid.zeros(1, 3, 224, 224))
    >>> print(output.shape)
    (1, 10)

.. note::

   - `ResNeSt` always uses a deep stem internally.
   - Factory helpers such as `resnest_50` and `resnest_50_4s2x40d` provide the historical presets.
