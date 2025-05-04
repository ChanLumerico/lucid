MobileNet_V2
============

.. toctree::
    :maxdepth: 1
    :hidden:

    mobilenet_v2.rst

.. raw:: html

   <span
     style="
       display: inline-block; padding: 0.15em 0.6em;
       border-radius: 999px; border: 1px solid #ffa600;
       color: #ffa600; background-color: transparent;
       font-size: 0.72em; font-weight: 500;
     "
   >
     ConvNet
   </span>

   <span
     style="
       display: inline-block; padding: 0.15em 0.6em;
       border-radius: 999px; border: 1px solid #707070;
       color: #707070; background-color: transparent;
       font-size: 0.72em; font-weight: 500;
     "
   >
     Image Classification
   </span>

.. autoclass:: lucid.models.MobileNet_V2

Overview
--------

The `MobileNetV2` class implements the MobileNet-v2 architecture, an advancement 
over MobileNet-v1. It introduces the concept of **inverted residual blocks** and 
**linear bottlenecks**, which significantly enhance efficiency and accuracy for 
mobile and embedded vision applications. This architecture is particularly optimized 
for lightweight and low-power tasks, making it suitable for real-time applications 
on mobile devices. 

.. image:: mobilenet_v2.png
    :width: 600
    :alt: MobileNet-v2 architecture
    :align: center

Class Signature
---------------

.. code-block:: python

    class MobileNet_V2(nn.Module):
        def __init__(self, num_classes: int = 1000) -> None

Parameters
----------
- **num_classes** (*int*, optional):
  Number of output classes for the classification task. Default is 1000, 
  commonly used for ImageNet.

