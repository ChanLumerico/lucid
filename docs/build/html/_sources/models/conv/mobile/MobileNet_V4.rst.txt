MobileNet_V4
============

.. toctree::
    :maxdepth: 1
    :hidden:

    mobilenet_v4_conv_small.rst
    mobilenet_v4_conv_medium.rst
    mobilenet_v4_conv_large.rst
    mobilenet_v4_hybrid_medium.rst
    mobilenet_v4_hybrid_large.rst

.. autoclass:: lucid.models.MobileNet_V4

Overview
--------
The `MobileNet_V4` class provides the foundational architecture for the MobileNet-v4 model family.
Building upon earlier MobileNet designs, it emphasizes both efficiency and flexibility, making it ideal
for mobile and embedded applications. By utilizing a configurable design through a dictionary-based
parameterization, this base class allows developers to easily experiment with different architectural
variants, striking a balance between computational cost and model performance.

.. image:: mobilenet_v4.png
    :width: 600
    :alt: MobileNet-v4 architecture
    :align: center

Class Signature
---------------
.. code-block:: python

    class MobileNet_V4(nn.Module):
        def __init__(self, cfg: Dict[str, dict], num_classes: int = 1000) -> None

Parameters
----------
- **cfg** (*Dict[str, dict]*):  
  A dictionary that holds the configuration for various network blocks. Each key in the dictionary
  represents a distinct module or layer group, and its corresponding value is another dictionary 
  that specifies parameters (such as kernel size, expansion factors, filter counts, strides, etc.) 
  for that module.

- **num_classes** (*int*, optional):  
  Specifies the number of output classes for the classification task. The default value is 1000,
  which is typically used for datasets like ImageNet.
