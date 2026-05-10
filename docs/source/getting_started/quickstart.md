# Quick Start

A five-minute walkthrough of the core Lucid workflow.

## 1. Tensors

```python
import lucid

# CPU tensor (Accelerate backend)
x = lucid.randn(4, 8)
print(x.shape)   # (4, 8)
print(x.device)  # cpu

# Move to Apple GPU (Metal / MLX backend)
x_gpu = x.to("metal")
lucid.eval(x_gpu)        # flush the lazy Metal graph
```

## 2. Autograd

```python
import lucid

a = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = a ** 2
c = b.sum()
c.backward()

print(a.grad)   # [[2., 4.], [6., 8.]]
```

## 3. Building a model

```python
import lucid
import lucid.nn as nn

class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(256, 10)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc2(self.act(self.fc1(x)))

model = MLP()
print(sum(p.numel() for p in model.parameters()), "parameters")
```

## 4. Training loop

```python
import lucid
import lucid.nn as nn
import lucid.optim as optim

model = MLP().to("metal")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    x = lucid.randn(64, 784, device="metal")
    y = lucid.randint(0, 10, shape=(64,), device="metal")

    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

    lucid.eval(loss)
    print(f"epoch {epoch}  loss={loss.item():.4f}")
```

## 5. Save & load

```python
import lucid

lucid.save(model.state_dict(), "mlp.lucid")
state = lucid.load("mlp.lucid")
model.load_state_dict(state)
```

## Next steps

- [Tensor API](../api/tensor.rst) — full tensor reference
- [Autograd guide](../guides/autograd.md) — custom Function, anomaly detection
- [Metal GPU guide](../guides/metal_device.rst) — device management, streams, profiling
