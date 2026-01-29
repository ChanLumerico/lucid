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
   Tensor Utilities <tensor/utilities/index.rst>

.. toctree::
   :maxdepth: 1
   :caption: Autograd
   :hidden:

   autograd/autograd.rst
   
   Autograd APIs <autograd/index.rst>

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
    :maxdepth: 2
    :caption: Einstein Operations
    :hidden:

    einops/einops.rst

    Einops Functions <einops/functions/index.rst>

.. toctree::
   :maxdepth: 3
   :caption: Neural Networks
   :hidden:

   nn/nn.rst
   nn/Parameter.rst
   nn/Buffer.rst
   nn/Module.rst
   
   Module Hooks <nn/ModuleHooks.rst>

   Neural Functions <nn/functions/index.rst>
   Weight Initializations <nn/init/index.rst>
   Modules <nn/modules/index.rst>
   Fused Modules <nn/fused/index.rst>
   Containers <nn/containers/index.rst>
   Utilities <nn/utilities/index.rst>

.. toctree::
    :maxdepth: 2
    :caption: Optimization
    :hidden:

    optim/optim.rst
    optim/Optimizer.rst
    optim/lr_scheduler.rst

    Optimizers <optim/optimizers/index.rst>
    LR Schedulers <optim/lr_scheduler/index.rst>

.. toctree::
    :maxdepth: 2
    :caption: Data
    :hidden:

    data/data.rst
    data/Dataset.rst
    data/Subset.rst
    data/TensorDataset.rst
    data/ConcatDataset.rst
    data/DataLoader.rst

    Utilities <data/utilities/index.rst>

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

    Image Classification <models/imgclf/index.rst>
    Image Generation <models/imggen/index.rst>
    Object Detection <models/objdet/index.rst>
    Sequence-to-Sequence <models/seq2seq/index.rst>

.. toctree::
    :maxdepth: 1
    :caption: Weights
    :hidden:
    
    weights/weights.rst

    Pre-Trained Weights <weights/list.rst>

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
   :caption: Visualization
   :hidden:

   visual/visual.rst
   
   Mermaid Charts <visual/Mermaid.rst>

.. toctree::
   :maxdepth: 1
   :caption: Porting
   :hidden:

   porting/save.rst
   porting/load.rst

.. toctree::
   :maxdepth: 1
   :caption: Others
   :hidden:

   others/Numeric.rst
   others/no_grad.rst
   others/grad_enabled.rst
   others/count_flops.rst
   others/newaxis.rst
   others/register_model.rst


.. module:: lucid
   :synopsis: An educational deep learning framework built from scratch.

Lucid² 💎
=========

.. image:: https://img.shields.io/pypi/v/lucid-dl?color=red
   :alt: PyPI - Version

.. image:: https://img.shields.io/pypi/dm/lucid-dl
   :alt: PyPI - Downloads

.. image:: https://static.pepy.tech/personalized-badge/lucid-dl?period=total&units=NONE&left_color=GRAY&right_color=yellow&left_text=total%20downloads
   :alt: PyPI - Total Downloads

.. image:: https://img.shields.io/github/languages/code-size/ChanLumerico/lucid
   :alt: GitHub code size in bytes

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :alt: Code Style

.. image:: https://img.shields.io/badge/dynamic/json?label=Lines%20of%20Code&color=purple&url=https%3A%2F%2Fraw.githubusercontent.com%2FChanLumerico%2Flucid%2Fmain%2Floc%2Floc_badge.json&query=%24.linesOfCode&cacheSeconds=3600
   :alt: Lines of Code

**Lucid** is a minimalist deep learning framework built entirely from scratch in Python.
It provides a pedagogically rich environment to explore the foundations of modern deep
learning systems—including autodiff, neural network modules, and GPU acceleration—while
remaining lightweight, readable, and free of complex dependencies.

Whether you’re a student, educator, or an advanced researcher seeking to demystify deep
learning internals, Lucid delivers a transparent and highly introspectable API that
faithfully replicates key behaviors of major frameworks like PyTorch, yet in a form simple
enough to study line by line.

How to Install
--------------

Basic Installation
~~~~~~~~~~~~~~~~~~
Install via PyPI:

.. code-block:: bash

   pip install lucid-dl

Alternatively, install the latest development version from GitHub:

.. code-block:: bash

   pip install git+https://github.com/ChanLumerico/lucid.git

Enable GPU (Metal / MLX Acceleration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you are using a Mac with Apple Silicon (M1, M2, M3), Lucid supports GPU execution via the MLX library.

To enable Metal acceleration:

1. **Install MLX:**

   .. code-block:: bash

      pip install mlx

2. **Confirm** you have a compatible device (Apple Silicon).
3. **Run any computation** with `device="gpu"`.

Verification
~~~~~~~~~~~~
Check whether GPU acceleration is functioning:

.. code-block:: python

   import lucid
   x = lucid.ones((1024, 1024), device="gpu")
   print(x.device)  # Should print: 'gpu'

Tensor: The Core Abstraction
-----------------------------
At the heart of Lucid is the `Tensor` class—a generalization of NumPy arrays that supports advanced
operations such as gradient tracking, device placement, and computation graph construction.

**Each Tensor encapsulates**:

- A data array (`ndarray` or `mlx.array`)
- A gradient buffer (`grad`)
- The operation that produced it
- A list of parent tensors from which it was derived
- A flag indicating whether it participates in the computation graph (`requires_grad`)

Construction and Configuration Example:

.. code-block:: python

   from lucid import Tensor

   x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, device="gpu")

- Setting `requires_grad=True` adds the tensor to the autodiff graph.
- Specifying `device="gpu"` allocates the tensor using the Metal backend.

Switching Between Devices
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tensors can be moved between CPU and GPU at any time using the `.to()` method:

.. code-block:: python

   x = x.to("gpu")  # Now uses MLX arrays for accelerated computation
   y = x.to("cpu")  # Moves data back to NumPy

Inspect the device of a tensor with:

.. code-block:: python

   print(x.device)  # Either 'cpu' or 'gpu'

Automatic Differentiation (Autodiff)
------------------------------------
Lucid implements **reverse-mode automatic differentiation**, which is especially efficient for computing
gradients of scalar-valued loss functions.

It builds a dynamic graph during the forward pass, capturing every operation involving tensors that require
gradients. Each node in the graph stores a custom backward function that computes local gradients and propagates
them upstream using the chain rule.

**Computation Graph Internals**:

- Each `Tensor` acts as a node in a Directed Acyclic Graph (DAG).
- Operations create edges between inputs and outputs.
- Each tensor’s `_backward_op` defines how to compute gradients with respect to its parent tensors.

**The backward method**:

1. Topologically sorts the computation graph.
2. Initializes the output gradient (typically `1.0`).
3. Executes all backward operations in reverse order.

Example:

.. code-block:: python

   import lucid

   x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
   y = x * 2 + 1
   z = y.sum()
   z.backward()
   print(x.grad)  # Output: [2.0, 2.0, 2.0]

This chain-rule application computes the gradient :math:`\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x} = [2, 2, 2]`.

Hooks & Shape Alignment
-------------------------
Lucid supports:

- **Hooks** for inspecting or modifying gradients.
- **Shape broadcasting and matching** to handle nonconforming tensor shapes.

Metal Acceleration (MLX Backend)
--------------------------------
Lucid supports **Metal acceleration** on Apple Silicon devices using the
`MLX <https://github.com/ml-explore/mlx>`_ library. This integration enables tensor operations,
neural network layers, and gradient computations to run efficiently on the GPU by leveraging Apple’s
unified memory and neural engine.

**Key Features**:

- Tensors with `device="gpu"` are allocated as `mlx.core.array`.
- Core mathematical operations, matrix multiplications, and backward passes leverage MLX APIs.
- The API remains unchanged; simply use `.to("gpu")` or pass `device="gpu"` to tensor constructors.

Basic Acceleration Example:

.. code-block:: python

   import lucid

   x = lucid.randn(1024, 1024, device="gpu", requires_grad=True)
   y = x @ x.T
   z = y.sum()
   z.backward()
   print(x.grad.device)  # 'gpu'

GPU-Based Model Example:

.. code-block:: python

   import lucid.nn as nn
   import lucid.nn.functional as F

   class TinyNet(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc = nn.Linear(100, 10)

       def forward(self, x):
           return F.relu(self.fc(x))

   model = TinyNet().to("gpu")
   data = lucid.randn(32, 100, device="gpu", requires_grad=True)
   output = model(data)
   loss = output.sum()
   loss.backward()

.. warning::
   When training models on GPU using MLX, you **must explicitly evaluate** the loss tensor after each forward
   pass to prevent the MLX computation graph from growing uncontrollably. MLX defers evaluation until necessary;
   if evaluation is not forced (e.g. by calling `.eval()`), the graph may grow too deep,
   leading to performance issues or memory errors.

Recommended GPU Training Pattern:

.. code-block:: python

   loss = model(input).sum()
   loss.eval()  # Force evaluation on GPU
   loss.backward()

Neural Networks with `lucid.nn`
----------------------------------
Lucid provides a modular, PyTorch-style interface for building neural networks via the `nn.Module` class.
Users define model classes by subclassing `nn.Module` and assigning parameters and layers as attributes.
Each module automatically registers its parameters, supports device migration via `.to()`, and integrates with
Lucid’s autodiff system.

Custom Module Definition Example:

.. code-block:: python

   import lucid.nn as nn

   class MLP(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc1 = nn.Linear(784, 128)
           self.fc2 = nn.Linear(128, 10)

       def forward(self, x):
           x = self.fc1(x)
           x = nn.functional.relu(x)
           x = self.fc2(x)
           return x

Parameter Registration:

.. code-block:: python

   model = MLP()
   print(model.parameters())

Moving to GPU:

.. code-block:: python

   model = model.to("gpu")

Training & Evaluation
---------------------
Lucid supports training neural networks using standard loops, customized optimizers, and
tracking gradients across batches of data.

Full Training Loop Example:

.. code-block:: python

   import lucid
   from lucid.nn.functional import mse_loss

   model = MLP().to("gpu")
   optimizer = lucid.optim.SGD(model.parameters(), lr=0.01)

   for epoch in range(100):
       preds = model(x_train)
       loss = mse_loss(preds, y_train)
       loss.eval()  # Force evaluation

       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

       print(f"Epoch {epoch}, Loss: {loss.item()}")

Evaluation without Gradients:

.. code-block:: python

   with lucid.no_grad():
       out = model(x_test)

Loading Pretrained Weights
--------------------------

Lucid supports loading pretrained weights for models using the `lucid.weights` module, 
which provides access to standard pretrained initializations.

.. code-block:: python

   from lucid.models import lenet_5
   from lucid.weights import LeNet_5_Weights

   # Load LeNet-5 with pretrained weights
   model = lenet_5(weights=LeNet_5_Weights.DEFAULT)

You can also initialize models without weights by passing `weights=None`.

Educational by Design
----------------------
Lucid isn’t a black box—it’s built to be explored. Every class, function, and line of code is crafted
to be readable and hackable.

- Build intuition for backpropagation.
- Modify internal operations to experiment with custom autograd.
- Benchmark CPU vs GPU behavior with your own models.
- Debug layer by layer, shape by shape, and gradient by gradient.

Conclusion
----------
Lucid serves as a powerful educational resource and a minimalist experimental sandbox.
By exposing the internals of tensors, gradients, and models—and integrating GPU acceleration—Lucid
invites users to *see, touch, and understand* how deep learning truly works.

Others
------
**Dependencies:** `NumPy`, `MLX`, `openml`, `pandas`

**Inspired By:**

.. image:: https://skillicons.dev/icons?i=pytorch
   :alt: PyTorch

.. image:: https://skillicons.dev/icons?i=tensorflow
   :alt: TensorFlow

.. image:: https://skillicons.dev/icons?i=stackoverflow
   :alt: StackOverflow
