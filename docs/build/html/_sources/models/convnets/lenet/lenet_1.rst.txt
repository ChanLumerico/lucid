models.lenet_1
==============

.. autofunction:: lucid.models.lenet_1

Overview
--------

The `lenet_1` function returns an instance of the LeNet-1 architecture, 
a simplified version of the LeNet series designed for small-scale image classification tasks. 

It is suitable for grayscale input images with a size of :math:`28 \times 28`, 
such as those in the MNIST dataset.

Function Signature
------------------

.. code-block:: python

   def lenet_1(**kwargs) -> LeNet

Returns
-------

- **LeNet**: 
  An instance of the `LeNet` class configured for the LeNet-1 architecture.

Architecture
------------

The LeNet-1 architecture consists of:

1. **Feature Extractor**:
   Two convolutional layers:
   - Conv1: 1 input channel, 4 output channels, :math:`5 \times 5` kernel, stride 1.
   - Conv2: 4 input channels, 12 output channels, :math:`5 \times 5` kernel, stride 1.
     
   Two average pooling layers with a :math:`2 \times 2` kernel and stride 2.

2. **Classifier**:
   One fully connected layer:
   - Input features: :math:`12 \times 4 \times 4 = 192`.
   - Output features: 10 (number of classes).

**Total Parameters**: 3,246

Example Usage
-------------

Below is an example of using the `lenet_1` function to create a LeNet-1 model:

.. code-block:: python

   import lucid.models as models

   # Create the LeNet-1 model
   model = models.lenet_1()

   # Create a sample input tensor (e.g., 28x28 grayscale image)
   input_tensor = lucid.Tensor([...])

   # Perform a forward pass
   output = model(input_tensor)
   print(output)
