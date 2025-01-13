
# **Lucid's Autodiff System**

## **Abstract**

This report provides an in-depth exploration of Lucid's reverse-mode automatic differentiation (autodiff) system. Lucid is an educational deep learning framework built with NumPy, designed to provide a clear understanding of gradient computation and backpropagation. This document elaborates on the design, implementation, and functionality of Lucid's autodiff system with annotated code excerpts.

## **Introduction**

Automatic differentiation (autodiff) is a cornerstone of modern deep learning frameworks. Lucid employs reverse-mode autodiff to compute gradients efficiently, enabling the optimization of neural network parameters. Unlike symbolic differentiation or numerical differentiation, autodiff combines efficiency and accuracy by breaking down computations into smaller, differentiable operations.

## **Core Principles**

Lucid's autodiff system is grounded in the following principles:

1. **Computation Graph**:
   - A directed acyclic graph (DAG) is constructed during the forward pass.
   - Each operation records its parents and defines a backward operation for gradient computation.

2. **Reverse-Mode Differentiation**:
   - Gradients are propagated from the output back to the inputs using the chain rule:

     $
     \text{gradient of parent} = \frac{\partial \text{output}}{\partial \text{parent}}
     $

3. **Modularity**:
   - Gradients are tracked automatically for tensors with `requires_grad=True`.
   - Hooks provide users with the ability to modify gradients or add custom logic during 
     backpropagation.


## **Implementation Overview**

### **Tensor Class: The Core Data Structure**

The `Tensor` class encapsulates data, gradients, and computation graph information. It forms the foundation of Lucid's autodiff system.

```python
class Tensor(_TensorOps):
    def __init__(
        self,
        data: _ArrayOrScalar,
        requires_grad: bool = False,
        keep_grad: bool = False,
        dtype: type = _base_dtype,
    ) -> None: ...
```

### **Building the Computation Graph**

During the forward pass, each tensor operation defines its parents and a `_backward_op` function to calculate gradients during backpropagation.

#### **Example: Slicing Operation**

```python
def __getitem__(self, idx: SupportsIndex) -> Self:
    sliced_data = self.data[idx]
    new_tensor = Tensor(sliced_data, self.requires_grad, self.keep_grad, self.dtype)

    def _backward_op() -> None:
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        new_grad = lucid._match_grad_shape(self.data[idx], new_tensor.grad)
        lucid._set_tensor_grad(self, new_grad, at=idx)

    if self.requires_grad:
        new_tensor._backward_op = _backward_op
        new_tensor._prev = [self]

    return new_tensor
```

### **Backward Pass: Topological Sorting**

Gradients are computed in reverse topological order using the `backward()` method.

```python
def backward(self, keep_grad: bool = False) -> None:
    if self.grad is None:
        self.grad = np.ones_like(self.data)

    visited = set()
    topo_order: list[Tensor] = []

    def _build_topo(tensor: Tensor) -> None:
        if tensor not in visited:
            visited.add(tensor)
            for parent in tensor._prev:
                _build_topo(parent)
            topo_order.append(tensor)

    _build_topo(self)
    topo_order.reverse()

    for tensor in topo_order:
        tensor._backward_op()
        for hook in tensor._backward_hooks:
            hook(tensor, tensor.grad)

        if not (tensor.is_leaf or keep_grad or self.keep_grad):
            self.grad = None
```

## **Autodiff in Action**

### **Matrix Multiplication Example**

The `matmul` function demonstrates the integration of forward and backward operations.

#### **Forward Pass**

```python
def matmul(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data @ b.data, requires_grad=a.requires_grad or b.requires_grad)

    def _backward_op():
        if out.grad is not None:
            if a.requires_grad:
                a.grad = lucid._match_grad_shape(a.data, out.grad @ b.data.T)
            if b.requires_grad:
                b.grad = lucid._match_grad_shape(b.data, a.data.T @ out.grad)

    if out.requires_grad:
        out._backward_op = _backward_op
        out._prev = [a, b]

    return out
```

#### **Backward Propagation**

- Gradient w.r.t $a$: $ \nabla_a = \nabla_{\text{out}} \cdot b^\top $
- Gradient w.r.t $b$: $ \nabla_b = a^\top \cdot \nabla_{\text{out}} $

### **Usage Example**
```python
import lucid

a = lucid.Tensor([[1, 2], [3, 4]], requires_grad=True)
b = lucid.Tensor([[5, 6], [7, 8]], requires_grad=True)

# Forward pass
out = a @ b

# Backward pass
out.backward()

# Gradients
print("Gradient of a:", a.grad)
print("Gradient of b:", b.grad)
```

---

## **Supporting Features**

### **Gradient Tracking**

Gradients are tracked automatically for tensors with `requires_grad=True`.

```python
>>> t = Tensor([[1.0, 2.0]], requires_grad=True)
>>> print(t.requires_grad)
True
```

### **Shape Matching**

Gradients are reshaped or broadcasted to align with their parent tensors.

```python
def _match_grad_shape(data: _NumPyArray, grad: _NumPyArray) -> _NumPyArray:
    # Shape alignment logic
    ...
    return matched_grad
```

### **Hooks**

Custom hooks allow users to modify gradients during backpropagation.

```python
def custom_hook(tensor, grad):
    print("Custom gradient:", grad)

>>> t.register_hook(custom_hook)
```

## **Conclusion**

Lucid’s autodiff system is an elegant and efficient implementation of reverse-mode differentiation, suitable for educational purposes and lightweight experimentation. 

It provides:
- Clear computation graph construction.
- Efficient gradient propagation.
- Flexibility through hooks and shape alignment.

This system serves as a foundational tool for learning and experimenting with deep learning concepts.

