# Autograd Guide

## Reverse-mode differentiation

Every `Tensor` with `requires_grad=True` participates in graph recording:

```python
import lucid

x = lucid.randn(3, requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)   # 2 * x
```

## Disabling gradient tracking

Use `lucid.no_grad()` for inference or initialisation:

```python
with lucid.no_grad():
    out = model(x)   # no graph recorded
```

## Custom Function nodes

Subclass `lucid.autograd.Function` to define custom forward/backward:

```python
import lucid
from lucid.autograd import Function

class Clamp(Function):
    @staticmethod
    def forward(ctx, x: lucid.Tensor, lo: float, hi: float) -> lucid.Tensor:
        ctx.save_for_backward(x)
        ctx.lo, ctx.hi = lo, hi
        return x.clamp(lo, hi)

    @staticmethod
    def backward(ctx, grad_out: lucid.Tensor) -> tuple[lucid.Tensor, None, None]:
        (x,) = ctx.saved_tensors
        mask = (x >= ctx.lo) & (x <= ctx.hi)
        return grad_out * mask.float(), None, None

y = Clamp.apply(x, 0.0, 1.0)
y.sum().backward()
```

## Gradient checkpointing

Trade memory for recomputation in long sequential models:

```python
from lucid.autograd import checkpoint

out = checkpoint(model.encoder, x)   # forward saved, recomputed on backward
```

## Anomaly detection

Pinpoint NaN / Inf gradients during debugging:

```python
with lucid.autograd.detect_anomaly():
    loss = model(x).sum()
    loss.backward()   # raises RuntimeError with stack trace on NaN grad
```
