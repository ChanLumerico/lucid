ResNeXt
=======

.. toctree::
    :maxdepth: 1
    :hidden:

    resnext_50_32x4d.rst
    resnext_101_32x4d.rst
    resnext_101_32x8d.rst
    resnext_101_32x16d.rst
    resnext_101_32x32d.rst
    resnext_101_64x4d.rst

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

.. autoclass:: lucid.models.ResNeXt

The `ResNeXt` class extends the `ResNet` architecture by incorporating group convolutions, 
allowing for an increase in model capacity while maintaining computational efficiency. 
This is achieved through the use of cardinality, a hyperparameter that specifies the number 
of groups in convolutions.

.. image:: resnext.png
    :width: 600
    :alt: ResNeXt architecture
    :align: center

Class Signature
---------------

.. code-block:: python

    class lucid.nn.ResNeXt(
        block: nn.Module,
        layers: list[int],
        cardinality: int,
        base_width: int,
        num_classes: int = 1000,
    )

Parameters
----------
- **block** (*nn.Module*):
  The building block module used for the ResNeXt layers. 
  Typically, this is a bottleneck block.

- **layers** (*list[int]*):
  Specifies the number of blocks in each stage of the network.

- **cardinality** (*int*):
  Number of groups for grouped convolutions. Higher cardinality 
  increases model capacity without significantly increasing computational cost.

- **base_width** (*int*):
  The base width of feature channels in each group.

- **num_classes** (*int*, optional):
  Number of output classes for the final fully connected layer. 
  Default: 1000.

Attributes
----------
- **layers** (*list[nn.Module]*):
  A list of stages, each containing a sequence of grouped 
  convolutional blocks.

- **cardinality** (*int*):
  Stores the number of groups used in the grouped convolutions.

- **base_width** (*int*):
  Stores the base width of the feature maps for each group.

Forward Calculation
-------------------

The forward pass of the `ResNeXt` model includes:

1. **Stem**: Initial convolutional layers for feature extraction.
2. **Grouped Convolution Stages**: Each stage applies grouped convolutions based on the 
   `cardinality` parameter.
3. **Global Pooling**: A global average pooling layer reduces spatial dimensions.
4. **Classifier**: A fully connected layer maps the features to class scores.

.. math::

    \text{output} = \text{FC}(\text{GAP}(\text{GroupedConvBlocks}(\text{Stem}(\text{input}))))

.. note::

   - The `ResNeXt` architecture introduces cardinality as an additional dimension 
     to control model capacity.
   - Increasing the cardinality improves feature learning while maintaining 
     computational efficiency.
