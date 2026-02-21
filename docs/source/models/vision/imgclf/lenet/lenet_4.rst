lenet_4
=======

.. autofunction:: lucid.models.lenet_4

Overview
--------

The `lenet_4` function returns an instance of the LeNet-4 architecture, 
an enhanced version of the LeNet series designed for small-scale image classification tasks. 

It is suitable for grayscale input images with a size of :math:`28 \times 28`, 
such as those in the MNIST dataset.

**Total Parameters**: 18,378

Function Signature
------------------

.. code-block:: python

   @register_model
   def lenet_4(**kwargs) -> LeNet

Returns
-------

- **LeNet**: 
  An instance of the `LeNet` class configured for the LeNet-4 architecture.

Architecture
------------

The LeNet-4 architecture consists of:

1. **Feature Extractor**:
   Two convolutional layers:
   - Conv1: 1 input channel, 4 output channels, :math:`5 \times 5` kernel, stride 1.
   - Conv2: 4 input channels, 12 output channels, :math:`5 \times 5` kernel, stride 1.
   
   Two average pooling layers with a :math:`2 \times 2` kernel and stride 2.

2. **Classifier**:
   Two fully connected layers:
   - FC1: Input features: :math:`12 \times 4 \times 4 = 192`, output features: 84.
   - FC2: Input features: 84, output features: 10 (number of classes).

Example Usage
-------------

Below is an example of using the `lenet_4` function to create a LeNet-4 model:

.. code-block:: python

   import lucid.models as models

   # Create the LeNet-4 model
   model = models.lenet_4()

   # Create a sample input tensor (e.g., 28x28 grayscale image)
   input_tensor = lucid.Tensor([...])

   # Perform a forward pass
   output = model(input_tensor)
   print(output)
