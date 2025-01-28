.. Lucid documentation master file, created by
   sphinx-quickstart on Wed Nov 13 13:10:20 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   :caption: Tensor
   :hidden:

   tensor/Tensor_.rst
   tensor/tensor.rst

   Tensor Operations <tensor/operations/index.rst>
   Tensor Utilites <tensor/utilities/index.rst>

.. toctree::
   :maxdepth: 2
   :caption: Linear Algebra
   :hidden:

   linalg/linalg.rst

   Linalg Operations <linalg/operations/index.rst>

.. toctree::
   :maxdepth: 2
   :caption: Random
   :hidden:

   random/random.rst

   RNG Functions <random/functions/index.rst>

.. toctree::
   :maxdepth: 3
   :caption: Neural Networks
   :hidden:

   nn/nn.rst
   nn/Parameter.rst
   nn/Buffer.rst
   nn/Module.rst

   Neural Functions <nn/functions/index.rst>
   Weight Initializations <nn/init/index.rst>
   Modules <nn/modules/index.rst>
   Fused Modules <nn/fused/index.rst>
   Containers <nn/containers/index.rst>

.. toctree::
    :maxdepth: 2
    :caption: Optimization
    :hidden:

    optim/optim.rst
    optim/Optimizer.rst

    Optimizers <optim/optimizers/index.rst>

.. toctree::
    :maxdepth: 2
    :caption: Data
    :hidden:

    data/data.rst
    data/Dataset.rst
    data/ConcatDataset.rst
    data/DataLoader.rst

.. toctree::
    :maxdepth: 2
    :caption: Datasets
    :hidden:

    datasets/datasets.rst

    Image Datasets <datasets/image/index.rst>

.. toctree::
    :maxdepth: 3
    :caption: Models
    :hidden:

    models/models.rst
    models/summarize.rst

    ConvNets <models/conv/index.rst>

.. toctree::
    :maxdepth: 2
    :caption: Transformation
    :hidden:

    transforms/transforms.rst
    transforms/Compose.rst
    transforms/ToTensor.rst

    Image Transforms <transforms/image/index.rst>

.. toctree::
   :maxdepth: 1
   :caption: Others
   :hidden:

   others/types.rst
   others/no_grad.rst
   others/grad_enabled.rst
   others/newaxis.rst
   others/register_model.rst


.. module:: lucid
   :synopsis: An educational deep learning framework built using NumPy.

Lucid
=====

**Lucid** is an educational deep learning framework developed to help users understand 
the underlying mechanics of machine learning models and tensor operations. 

It is designed to provide a simple yet powerful environment to experiment with neural networks, 
optimization, and backpropagation using only **NumPy**. 

Lucid is ideal for those who want to learn about the inner workings of deep learning 
algorithms and operations without the complexity of high-level frameworks.

Overview
--------

Lucid provides core functionality for building and training machine learning models. 
By utilizing NumPy arrays as the fundamental data structure (referred to as **Tensors**), 
Lucid allows for the construction of layers, models, and operations commonly found in neural networks. 

It offers automatic differentiation (autodiff) for computing gradients and performing backpropagation, 
enabling efficient optimization of model parameters.

Key Features
------------

- **Tensors**: Tensors are the main data structure in Lucid, 
  similar to arrays in NumPy but with additional features such as automatic gradient tracking.

- **Autodiff**: Lucid computes gradients automatically using reverse-mode differentiation, 
  making it possible to train models through backpropagation.

- **Modularity**: Lucid is designed with modularity in mind, allowing users to build and 
  customize layers, models, and operations with ease.

- **Gradient Tracking**: Support for tracking gradients through Tensors, 
  enabling automatic backpropagation during training.

- **Educational Focus**: Lucid is a minimalistic library designed to be intuitive and provide 
  a deeper understanding of the mechanics of deep learning.

Core Components
---------------

.. rubric:: `Tensors`

Tensors are the primary data structure in Lucid, similar to NumPy arrays but with additional capabilities, 
such as the ability to track gradients. 

Operations performed on tensors are automatically tracked, allowing for efficient backpropagation.

- **Tensor Operations**: Basic operations like addition, subtraction, multiplication, 
  and division are supported, with automatic gradient computation for supported operations.

- **Gradient Tracking**: When constructing Tensors, 
  users can specify if they require gradients for backpropagation.

- **Shape Management**: Lucid supports reshaping, transposing, and other tensor manipulation 
  operations to allow for flexible model design.

.. rubric:: `Neural Networks (nn.Module)`

Lucid provides a framework for defining and training neural networks. 
Models are built by subclassing the **nn.Module** class, which allows users to define layers, 
forward passes, and backward passes (gradient computations) for the model.

- **Layer Definitions**: Layers can be constructed using basic operations, like matrix multiplication, 
  activation functions, and loss functions.

- **Forward and Backward Passes**: Users define the computation graph in the `forward` method, 
  and Lucid handles backpropagation automatically by tracking operations performed on tensors.

.. rubric:: `Linear Algebra Operations (lucid.linalg)`

Lucid includes basic linear algebra operations, such as matrix multiplication, 
inverse, determinant calculation, and more.

- **Matrix Operations**: These operations are essential for building and manipulating neural networks, 
  particularly for tasks like transforming data in the forward pass.

.. rubric:: `Optimization`

Lucid supports optimization routines like Stochastic Gradient Descent (SGD), 
which allow for the training of models by minimizing a loss function.

- **Autodiff and Backpropagation**: Lucid's autodiff capabilities make it easy to compute 
  gradients and optimize model parameters using backpropagation.

Example Usage
-------------

The following example demonstrates how to define and train a simple neural network using Lucid.

.. code-block:: python
   :caption: Example of a Simple Model

   import lucid
   import lucid.nn as nn
   import lucid.nn.functional as F

   # Define a simple model class
   class SimpleModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.dense1 = nn.Linear(3, 5)
           self.dense2 = nn.Linear(5, 1)
       
       def forward(self, x):
           x = self.dense1(x)
           x = F.relu(x)
           x = self.dense2(x)
           return x

   # Create an instance of the model
   model = SimpleModel()

   # Create a sample input tensor
   input_tensor = lucid.Tensor([[1.0, 2.0, 3.0]])

   # Forward pass
   output = model(input_tensor)

   print(output)

In the above example, we define a simple model with two dense layers, 
apply a ReLU activation, and perform a forward pass using an input tensor. 

This showcases how easy it is to define models and run computations with Lucid.

Notes
-----

.. note::
   - Lucid is built for learning and experimenting with deep learning concepts, 
     allowing users to see how operations like backpropagation, optimization, 
     and activation functions are implemented at a low level.

   - Lucid is lightweight, with no external dependencies beyond NumPy, 
     making it easy to install and use without complex setups or specialized hardware.

Limitations
-----------

.. warning::
   - Lucid does not aim to provide the high-level functionalities of production-ready frameworks. 
     Instead, it focuses on educational value and understanding how deep 
     learning models are built from scratch.

   - Performance optimizations that are available in specialized libraries may not be as 
     efficient in Lucid, as it is not optimized for production workloads.

Conclusion
----------

Lucid provides a minimalistic, educational environment to learn about deep learning using only NumPy. 
It gives users the tools to experiment with neural networks, automatic differentiation, 
optimization, and other essential components of deep learning, all while providing insight into how 
these operations are implemented at the core level.

For further information and usage details, refer to the documentation of specific modules like 
`lucid.nn` and `lucid.linalg`

.. seealso::
   `lucid.nn`, `lucid.linalg`
