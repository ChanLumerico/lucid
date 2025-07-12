util.FPN
========

.. autoclass:: lucid.models.objdet.util.FPN

The `FPN` (Feature Pyramid Network) module constructs a multi-scale feature pyramid 
from a list of input feature maps. It enhances semantic representation at all scales, 
typically used in detection backbones.

Constructor
-----------

.. code-block:: python

    def __init__(in_channels_list: list[int], out_channels: int = 256) -> None

Parameters
----------

- **in_channels_list** (*list[int]*): 
  List of channel dimensions for input feature maps 
  (e.g., from backbone layers `C2`, `C3`, `C4`, `C5`).
- **out_channels** (*int*, optional): 
  Number of output channels per feature map in the pyramid. Defaults to `256`.

Returns
-------

- **FPN** (*nn.Module*): 
  A module that fuses multi-scale feature maps into semantically rich pyramidal outputs.

Forward Input & Output
----------------------

.. code-block:: python

    def forward(features: list[Tensor]) -> list[Tensor]

- **features** (*list[Tensor]*): 
  Input feature maps from the backbone, typically of decreasing spatial size.
- **Returns** (*list[Tensor]*): 
  A list of feature maps of shape :math:`(N, \text{out\_channels}, H_l, W_l)` 
  at each pyramid level, with lateral and top-down connections fused.

.. note::

   - Uses 1x1 lateral convs and 3x3 smoothing convs.
   - All outputs are aligned to the same number of channels (`out_channels`).
   - Maintains same resolution per level as the input list.

Example
-------

.. code-block:: python

    >>> from lucid.models.objdet.util import FPN
    >>> inputs = [lucid.random.randn(1, c, s, s) for c, s in zip([64, 128, 256, 512], [64, 32, 16, 8])]
    >>> fpn = FPN(in_channels_list=[64, 128, 256, 512])
    >>> outputs = fpn(inputs)
    >>> for feat in outputs:
    ...     print(feat.shape)
    (1, 256, 64, 64)
    (1, 256, 32, 32)
    (1, 256, 16, 16)
    (1, 256, 8, 8)
