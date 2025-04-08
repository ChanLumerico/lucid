# Lucid 💎²

![PyPI - Version](https://img.shields.io/pypi/v/lucid-dl?color=red)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lucid-dl)
![PyPI - Total Downloads](https://img.shields.io/badge/total%20downloads-21.3k-yellow)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/ChanLumerico/lucid)
![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)
![Lines of Code](https://img.shields.io/endpoint?url=https%3A%2F%2Floc-counter.onrender.com%2F%3Frepo%3DChanLumerico%2Flucid%26branch%3Dmain%26ignored%3Ddocs%26stat%3DlinesOfCode&label=Lines%20of%20Code&color=purple)

**Lucid** is a minimalist deep learning framework built entirely from scratch in Python. It offers a pedagogically rich environment to explore the foundations of modern deep learning systems, including autodiff, neural network modules, and GPU acceleration — all while staying lightweight, readable, and free of complex dependencies.

Whether you're a student, educator, or an advanced researcher seeking to demystify deep learning internals, Lucid provides a transparent and highly introspectable API that faithfully replicates key behaviors of major frameworks like PyTorch, yet in a form simple enough to study line by line.

[📑 Lucid Documentation](https://chanlumerico.github.io/lucid/build/html/index.html)

>📢 *Lucid 1.23* currently serves as a pre-release of ***Lucid 2.0***.

## 🔧 How to Install

Lucid is designed to be light, portable, and friendly to all users — no matter your setup.

### ▶️ Basic Installation
Lucid is available directly on PyPI:
```bash
pip install lucid-dl
```

Alternatively, you can install the latest development version from GitHub:
```bash
pip install git+https://github.com/ChanLumerico/lucid.git
```
This installs all the core components needed to use Lucid in CPU mode using NumPy.

### ⚡ Enable GPU (Metal / MLX Acceleration)
If you're using a Mac with Apple Silicon (M1, M2, M3), Lucid supports GPU execution via the MLX library.

To enable Metal acceleration:
1. Install MLX:
```bash
pip install mlx
```
2. Confirm you have a compatible device (Apple Silicon).
3. Run any computation with `device="gpu"`.

### ✅ Verification
Here's how to check whether GPU acceleration is functioning:
```python
import lucid
x = lucid.ones((1024, 1024), device="gpu")
print(x.device)  # Should print: 'gpu'
```


## 📐 Tensor: The Core Abstraction

At the heart of Lucid is the `Tensor` class — a generalization of NumPy arrays that supports advanced operations such as gradient tracking, device placement, and computation graph construction.

Each Tensor encapsulates:
- A data array (`ndarray` or `mlx.array`)
- Gradient (`grad`) buffer
- The operation that produced it
- A list of parent tensors from which it was derived
- Whether it participates in the computation graph (`requires_grad`)

### 🔁 Construction and Configuration
```python
from lucid import Tensor

x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, device="gpu")
```

- `requires_grad=True` adds this tensor to the autodiff graph.
- `device="gpu"` allocates the tensor on the Metal backend.

### 🔌 Switching Between Devices
Tensors can be moved between CPU and GPU at any time using `.to()`:
```python
x = x.to("gpu")  # Now uses MLX arrays for accelerated computation
y = x.to("cpu")  # Moves data back to NumPy
```

You can inspect which device a tensor resides on via:
```python
print(x.device)  # Either 'cpu' or 'gpu'
```


## 📉 Automatic Differentiation (Autodiff)

Lucid implements **reverse-mode automatic differentiation**, which is commonly used in deep learning due to its efficiency for computing gradients of scalar-valued loss functions.

It builds a dynamic graph during the forward pass, capturing every operation involving Tensors that require gradients. Each node stores a custom backward function which, when called, computes local gradients and propagates them upstream using the chain rule.

### 📘 Computation Graph Internals
The computation graph is a Directed Acyclic Graph (DAG) in which:
- Each `Tensor` acts as a node.
- Each operation creates edges between inputs and outputs.
- A `_backward_op` method is associated with each Tensor that defines how to compute gradients w.r.t. parents.

The `.backward()` method:
1. Topologically sorts the graph.
2. Initializes the output gradient (usually with 1.0).
3. Executes all backward operations in reverse order.

### 🧠 Example
```python
import lucid

x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2 + 1
z = y.sum()
z.backward()
print(x.grad)  # Output: [2.0, 2.0, 2.0]
```

This chain-rule application computes the gradient $\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y}\cdot\frac{\partial y}{\partial x} = [2, 2, 2]$.

### 🔄 Hooks & Shape Alignment
Lucid supports:
- **Hooks** for gradient inspection or modification.
- **Shape broadcasting and matching** for non-conforming tensor shapes.


## 🚀 Metal Acceleration (MLX Backend)

Lucid supports **Metal acceleration** on Apple Silicon devices using [MLX](https://github.com/ml-explore/mlx). This integration allows tensor operations, neural network layers, and gradient computations to run efficiently on the GPU, leveraging Apple’s unified memory and neural engine.

### 📋 Key Features
- Tensors with `device="gpu"` are allocated as `mlx.core.array`.
- Core mathematical operations, matrix multiplications, and backward passes use MLX APIs.
- No change in API: switching to GPU is as simple as `.to("gpu")` or passing `device="gpu"` to tensor constructors.

### 💡 Example 1: Basic Acceleration
```python
import lucid

x = lucid.randn(1024, 1024, device="gpu", requires_grad=True)
y = x @ x.T
z = y.sum()
z.backward()
print(x.grad.device)  # 'gpu'
```

### 💡 Example 2: GPU-Based Model
```python
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
```

When training models on GPU using MLX, **you must explicitly evaluate the loss tensor** after each forward pass to prevent the MLX computation graph from growing uncontrollably.

MLX defers evaluation until needed. If you don’t force evaluation (e.g. calling `.eval()`), the internal graph may become too deep and lead to performance degradation or memory errors.

### Recommended GPU Training Pattern:
```python
loss = model(input).sum()
loss.eval()  # force evaluation on GPU
loss.backward()
```
This ensures that all prior GPU computations are flushed and evaluated **before** backward pass begins.


## 🧱 Neural Networks with `lucid.nn`

Lucid provides a modular PyTorch-style interface to build neural networks via `nn.Module`. Users define model classes by subclassing `nn.Module` and defining parameters and layers as attributes.

Each module automatically registers its parameters, supports device migration (`.to()`), and integrates with Lucid’s autodiff system.

### 🧰 Custom Module Definition
```python
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
```

### 🧩 Parameter Registration
All parameters are registered automatically and can be accessed:
```python
model = MLP()
print(model.parameters())
```

### 🧭 Moving to GPU
```python
model = model.to("gpu")
```
This ensures all internal parameters are transferred to GPU memory.


## 🏋️‍♂️ Training & Evaluation

Lucid supports training neural networks using standard loops, customized optimizers, and tracking gradients over batches of data.

### ✅ Full Training Loop
```python
import lucid
from lucid.nn.functional import mse_loss

model = MLP().to("gpu")
optimizer = lucid.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    preds = model(x_train)
    loss = mse_loss(preds, y_train)
    loss.eval()  # force evaluation

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")
```

### 🧪 Evaluation without Gradients
```python
with lucid.no_grad():
    out = model(x_test)
```
Prevents gradient tracking and reduces memory usage.

## 🧬 Educational by Design

Lucid is not a black box. It’s built to be explored. Every class, every function, and every line is designed to be readable and hackable.

- Use it to build intuition for backpropagation.
- Modify internal operations to test custom autograd.
- Benchmark CPU vs GPU behavior on your own model.
- Debug layer by layer, shape by shape, gradient by gradient.

Whether you're building neural nets from scratch, inspecting gradient flow, or designing a new architecture — Lucid is your transparent playground.

## 🧠 Conclusion
Lucid serves as a powerful educational resource and a minimalist experimental sandbox. By exposing the internals of tensors, gradients, and models — and integrating GPU acceleration — it invites users to **see, touch, and understand** how deep learning truly works.

## 📜 Others

**Dependencies**: `NumPy`, `MLX`, `openml`, `pandas`

**Inspired By**:

![](https://skillicons.dev/icons?i=pytorch)
![](https://skillicons.dev/icons?i=tensorflow)
![](https://skillicons.dev/icons?i=stackoverflow)
