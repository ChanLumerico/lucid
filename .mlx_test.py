import numpy as np
import mlx.core as mx


class Tensor:
    def __init__(self, data, requires_grad=False, device="cpu") -> None:
        if isinstance(data, np.ndarray):
            self.data = mx.array(data) if device == "gpu" else data
        elif isinstance(data, mx.array):
            self.data = data
        
        self.requires_grad = requires_grad
        self.device = device
        self.grad = None

        self._prev = []
        self._backward_op = lambda: None
    
    def to(self, device):
        if self.device == device:
            return self
        
        if device == "cpu":
            self.data = np.array(self.data)
        elif device == "gpu":
            self.data = mx.array(self.data)
        
        self.device = device
        return self
    
    # Test temporary operation
    def add(self, other):
        if self.device != other.device:
            raise ValueError(f"Device mismatch: {self.device} vs {other.device}")
        
        if self.device == "gpu":
            result_data = mx.add(self.data, other.data)
        else:
            result_data = self.data + other.data
        
        result = Tensor(result_data, device=self.device)

        def compute_grad():
            grad = result.grad
            if result.device == "gpu":
                grad = mx.array(grad)
            
            self.grad = grad
            other.grad = grad
        
        result._backward_op = compute_grad
        return result


a = Tensor(np.diag([1, 2, 3]), device="cpu")
b = Tensor(np.ones((3, 3)), device="cpu")

a.to("gpu")
b.to("gpu")

c = a.add(b)
c.grad = np.ones_like(c.data) if c.device == "cpu" else mx.ones_like(c.data)

print(a.grad, b.grad)

c._backward_op()

print(a.grad, b.grad)
print(type(a.grad), type(b.grad))