SENet
=====

.. toctree::
    :maxdepth: 1
    :hidden:

    se_resnet_18.rst
    se_resnet_34.rst
    se_resnet_50.rst
    se_resnet_101.rst
    se_resnet_152.rst

    se_resnext_50_32x4d.rst
    se_resnext_101_32x4d.rst
    se_resnext_101_32x8d.rst
    se_resnext_101_64x4d.rst

|convnet-badge| |imgclf-badge|

.. autoclass:: lucid.models.SENet

The `SENet` class serves as a foundational implementation for creating Squeeze-and-Excitation 
Network architectures.

.. image:: senet.png
    :width: 600
    :alt: SENet architecture
    :align: center

Class Signature
---------------

.. code-block:: python

    class SENet(
        block: Type[nn.Module], layers: List[int], num_classes: int = 1000, reduction: int = 16
    ):

Parameters
----------

- **block** (*Type[nn.Module]*):
  The block type to use, incorporating the squeeze-and-excitation mechanism.

- **layers** (*List[int]*):
  A list of integers specifying the number of blocks at each stage of the SENet.

- **num_classes** (*int*, optional):
  Number of output classes for the final fully connected layer. Default is 1000.

- **reduction** (*int*, optional):
  Reduction ratio for the squeeze-and-excitation mechanism. Default is 16.

Attributes
----------

- **conv** (*Sequential*):
  Initial convolutional block including a convolutional layer, batch normalization, 
  ReLU activation, and max pooling.

- **stage1, stage2, stage3, stage4** (*Sequential*):
  Stages of the SENet, each containing multiple blocks as specified in the `layers` parameter, 
  with squeeze-and-excitation operations.

- **avgpool** (*AdaptiveAvgPool2d*):
  Global average pooling layer before the fully connected layer.

- **fc** (*Linear*):
  Fully connected layer for classification.

Forward Calculation
--------------------

The forward method defines the computation flow:

.. code-block:: python

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
