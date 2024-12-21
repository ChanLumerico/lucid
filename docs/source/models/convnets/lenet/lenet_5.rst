lenet_5
=======

.. autofunction:: lucid.models.lenet_5

Overview
--------

The `lenet_5` function returns an instance of the LeNet-5 architecture, 
a classic convolutional neural network designed for handwritten digit recognition and 
other small-scale image classification tasks. 

It is suitable for grayscale input images with a size of :math:`32 \times 32`, 
such as those in the original LeNet dataset.

Function Signature
------------------

.. code-block:: python

   def lenet_5(**kwargs) -> LeNet

Returns
-------

- **LeNet**: 
  An instance of the `LeNet` class configured for the LeNet-5 architecture.

Architecture
------------

The LeNet-5 architecture consists of:

1. **Feature Extractor**:
   Two convolutional layers:
   - Conv1: 1 input channel, 6 output channels, :math:`5 \times 5` kernel, stride 1.
   - Conv2: 6 input channels, 16 output channels, :math:`5 \times 5` kernel, stride 1.
   
   Two average pooling layers with a :math:`2 \times 2` kernel and stride 2.

2. **Classifier**:
   Three fully connected layers:
   - FC1: Input features: :math:`16 \times 5 \times 5 = 400`, output features: 120.
   - FC2: Input features: 120, output features: 84.
   - FC3: Input features: 84, output features: 10 (number of classes).

***Total Parameters***: 61,706

Example Usage
-------------

Below is an example of using the `lenet_5` function to create a LeNet-5 model:

.. code-block:: python

   import lucid.models as models

   # Create the LeNet-5 model
   model = models.lenet_5()

   # Create a sample input tensor (e.g., 32x32 grayscale image)
   input_tensor = lucid.Tensor([...])

   # Perform a forward pass
   output = model(input_tensor)
   print(output)
