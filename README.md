# Lucid

**Lucid** is a lightweight, educational deep learning library, meticulously crafted from scratch using only NumPy. Designed for developers, students, and enthusiasts, Lucid serves as a foundational tool to understand the mechanics behind modern deep learning frameworks. It offers a comprehensive yet simple implementation of core deep learning concepts, making it an ideal resource for learning, experimenting, and prototyping.

## Key Features

- **Built with NumPy**: No external deep learning dependencies. Lucid is constructed entirely with NumPy, emphasizing a clear understanding of operations and gradients at every level.
- **Core Tensor Operations**: Support for essential tensor operations, including broadcasting, advanced indexing, and linear algebra.
- **Automatic Differentiation**: A custom autodiff engine that computes gradients for scalar, vector, matrix, and tensor operations with support for complex computation graphs.
- **Educational Purpose**: Written with clarity and simplicity, Lucid prioritizes readability and pedagogy over performance, making it ideal for studying deep learning internals.
- **Flexibility for Experimentation**: Experiment with gradient computations, custom operations, and optimization techniques in an intuitive environment.

## Getting Started

### Installation

Clone the repository to get started:

```bash
git clone https://github.com/ChanLumerico/lucid.git
cd lucid
```

Since Lucid is built on NumPy, ensure you have it installed:

```bash
pip install numpy
```

### Example Usage

#### Create a Tensor
```python
from lucid import Tensor

# Create tensors with gradients
a = Tensor([1, 2, 3], requires_grad=True)
b = Tensor([4, 5, 6], requires_grad=True)

# Perform operations
c = a + b
c.backward()

print(a.grad)  # Gradients of 'a'
print(b.grad)  # Gradients of 'b'
```

#### Perform Linear Algebra Operations
```python
from lucid import matmul

# Matrix multiplication
a = Tensor([[1, 2], [3, 4]], requires_grad=True)
b = Tensor([[5, 6], [7, 8]], requires_grad=True)

c = matmul(a, b)
c.backward()

print(a.grad)  # Gradients with respect to 'a'
print(b.grad)  # Gradients with respect to 'b'
```

## Why Lucid?

Lucid is not just a library; it's an exploration of how deep learning works at its core. While frameworks like TensorFlow and PyTorch abstract much of the complexity, Lucid reveals the fundamental building blocks of deep learning. Hereâs why youâll love it:

- **Educational**: Learn how gradients are computed, how tensors interact, and how models are built.
- **Lightweight**: Avoid the overhead of larger frameworks.
- **Readable**: Designed for clarity and understanding.
- **Experimental**: A perfect playground for testing out new ideas.

## Documentation

Explore the [full documentation](docs/index.html) for more detailed guides, API references, and advanced tutorials.

## Contribution

We welcome contributions from anyone passionate about deep learning or educational tools. Feel free to open issues, suggest features, or submit pull requests to help make Lucid even better.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Lucid is inspired by the simplicity of NumPy and the power of modern deep learning frameworks. It is crafted for those curious about the building blocks of AI.

