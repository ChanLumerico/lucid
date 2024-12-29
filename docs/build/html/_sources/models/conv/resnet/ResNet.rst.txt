ResNet
======

.. toctree::
    :maxdepth: 1
    :hidden:

    resnet_18.rst
    resnet_34.rst
    resnet_50.rst
    resnet_101.rst
    resnet_152.rst
    resnet_200.rst
    resnet_269.rst
    resnet_1001.rst

.. autoclass:: lucid.models.ResNet

The `ResNet` class serves as a foundational implementation for creating ResNet architectures.

Class Signature
---------------

.. code-block:: python

    class ResNet(block: Type[Module], layers: List[int], num_classes: int = 1000):

Parameters
----------

- **block** (*Type[Module]*):
  The block type to use (e.g., `BasicBlock` or `Bottleneck`).

- **layers** (*List[int]*):
  A list of integers specifying the number of blocks at each stage of the ResNet.

- **num_classes** (*int*, optional):
  Number of output classes for the final fully connected layer. Default is 1000.

Attributes
----------

- **conv1** (*Conv2d*):
  The initial convolutional layer.

- **bn1** (*BatchNorm2d*):
  Batch normalization applied after the initial convolution.

- **relu** (*ReLU*):
  ReLU activation function.

- **maxpool** (*MaxPool2d*):
  Max pooling operation applied after the first convolutional layer.

- **layer1, layer2, layer3, layer4** (*Sequential*):
  Stages of the ResNet, each containing multiple blocks as specified 
  in the `layers` parameter.

- **avgpool** (*AdaptiveAvgPool2d*):
  Global average pooling layer before the fully connected layer.

- **fc** (*Linear*):
  Fully connected layer for classification.

Forward Calculation
--------------------

The forward method defines the computation flow:

.. code-block:: python

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
