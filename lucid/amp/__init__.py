r"""Automatic Mixed Precision (AMP) utilities for Lucid.

Mixed precision training keeps the master copy of parameters and the
loss in float32 while running the bulk of forward / backward compute in
a lower-precision format (float16 or bfloat16). On Apple Silicon this
typically yields a wall-clock speed-up of roughly two times on
compute-bound layers and roughly halves the activation memory
footprint, enabling larger batch sizes or longer context windows.

This subpackage exposes two cooperating tools:

- :class:`autocast` — a context manager that casts eligible ops to a
  lower-precision dtype while leaving reductions and accumulators in
  float32 for numerical safety.
- :class:`GradScaler` — a dynamic loss-scaling helper that prevents
  gradient underflow when training in float16. Tracks an exponential
  scale factor, halving it on detected ``inf`` / ``nan`` and growing
  it back up on successful steps.

Examples
--------
>>> import lucid
>>> from lucid.amp import autocast, GradScaler
>>> scaler = GradScaler()
>>> with autocast(device_type="metal", dtype=lucid.float16):
...     loss = ...  # forward pass
>>> scaler.scale(loss).backward()
>>> scaler.step(optimizer)
>>> scaler.update()
"""

from lucid.amp.autocast import autocast
from lucid.amp.grad_scaler import GradScaler

__all__ = ["autocast", "GradScaler"]
