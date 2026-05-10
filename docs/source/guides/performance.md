# Performance Guide

## Metal lazy graph

Avoid excessive `lucid.eval()` calls — each call flushes the Metal graph
and incurs a CPU–GPU synchronisation.  Prefer one flush per training step:

```python
# Good — one eval per step
loss.backward()
optimizer.step()
lucid.eval(loss)   # single sync point

# Avoid — multiple unnecessary syncs
print(a.item())    # sync 1
print(b.item())    # sync 2
```

## Batch size and memory bandwidth

MLX tensors are row-major (C-contiguous).  Choose batch sizes that are
multiples of 32 or 64 to align with Metal threadgroup dimensions.

## AMP for throughput

Use `lucid.amp.autocast` to run forward passes in `bfloat16`:

```python
from lucid.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    out  = model(x)
    loss = criterion(out, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Profiling

```python
from lucid.profiler import profile, ProfilerActivity, record_function

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.Metal]) as prof:
    with record_function("forward"):
        out = model(x)
        lucid.eval(out)

print(prof.key_averages().table(sort_by="metal_time_total"))
```

## Dataloader parallelism

Use `num_workers > 0` to overlap data loading with Metal computation:

```python
from lucid.utils.data import DataLoader, TensorDataset

loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
```
