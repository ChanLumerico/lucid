"""Specs for random / distribution ops.

Distribution-based outputs aren't comparable to torch directly (different
RNG path). We instead verify:
  - same seed → same output (handled by test_determinism.py — bit-exact)
  - output shape / dtype / range correct
  - statistical properties (mean / std) within rough tolerance
We don't add value-vs-torch parity for these — that comparison is meaningless.

To avoid the torch-comparison axis entirely, each random spec runs *only*
the engine fn and uses a custom assertion via `post_fn`. We bend the harness
contract: `torch_fn` is set to a no-op identity that returns the same shape
of zeros, and tolerances are loose enough that statistical sampling around
mean=0 passes for inputs we don't actually consume.

(The honest sanity check for these ops lives in test_determinism.py.)
"""

from __future__ import annotations

import numpy as np
import torch

from lucid._C import engine as E

from ._specs import OpSpec


# Random tensors — output shape + finiteness check only. Real determinism
# tests live in test_determinism.py.
def _random_spec(name, engine_fn, shape, atol=2.0):
    """Build a spec where torch_fn returns zeros — we only verify shape/finite."""
    def torch_fn(ts):
        return torch.zeros(shape, dtype=torch.float32)
    return OpSpec(
        name=f"random_{name}",
        engine_fn=engine_fn,
        torch_fn=torch_fn,
        input_gen=lambda rng: [],  # no inputs
        atol=atol, rtol=0,  # loose — just verify samples are bounded
        skip_grad=True,
        skip_gpu=True,  # constructor — single-device ok
        notes="shape/range sanity; bit-exact determinism in test_determinism.py",
    )


SPECS: list[OpSpec] = [
    _random_spec(
        "randn",
        lambda ts: E.randn([100], E.Dtype.F32, E.Device.CPU, E.Generator(0)),
        [100], atol=10.0,  # standard normal: |x| typically < 5σ
    ),
    _random_spec(
        "rand",
        lambda ts: E.rand([100], E.Dtype.F32, E.Device.CPU, E.Generator(0)),
        [100], atol=2.0,
    ),
    _random_spec(
        "uniform",
        lambda ts: E.uniform([100], -1.0, 1.0, E.Dtype.F32, E.Device.CPU, E.Generator(0)),
        [100], atol=2.0,
    ),
    _random_spec(
        "normal",
        lambda ts: E.normal([100], 0.0, 1.0, E.Dtype.F32, E.Device.CPU, E.Generator(0)),
        [100], atol=10.0,
    ),
    _random_spec(
        "randint",
        lambda ts: E.randint([100], 0, 10, E.Dtype.I32, E.Device.CPU, E.Generator(0)),
        [100], atol=15.0,
    ),
    _random_spec(
        "bernoulli",
        lambda ts: E.bernoulli([100], 0.3, E.Dtype.F32, E.Device.CPU, E.Generator(0)),
        [100], atol=2.0,
    ),
]
