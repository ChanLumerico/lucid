"""Specs for gfunc — tensor constructors (zeros / ones / eye / arange / ...).

Constructors take no input tensors; we override `engine_fn` / `torch_fn`
to ignore `ts` and produce the same output. The harness still feeds them
a dummy input via `input_gen`.
"""

from __future__ import annotations

import numpy as np
import torch

from lucid._C import engine as E

from ._specs import OpSpec


def _no_input(rng):
    return [rng.standard_normal((1,)).astype("float32")]  # ignored


SPECS: list[OpSpec] = [
    OpSpec(
        name="gfunc_zeros",
        engine_fn=lambda ts: E.zeros([3, 4], E.Dtype.F32, E.Device.CPU),
        torch_fn=lambda ts: torch.zeros((3, 4), dtype=torch.float32),
        input_gen=_no_input,
        skip_grad=True,
        skip_gpu=True,  # constructor — device-aware in engine_fn directly
    ),
    OpSpec(
        name="gfunc_ones",
        engine_fn=lambda ts: E.ones([3, 4], E.Dtype.F32, E.Device.CPU),
        torch_fn=lambda ts: torch.ones((3, 4), dtype=torch.float32),
        input_gen=_no_input,
        skip_grad=True,
        skip_gpu=True,
    ),
    OpSpec(
        name="gfunc_eye",
        engine_fn=lambda ts: E.eye(4, 4, 0, E.Dtype.F32, E.Device.CPU),
        torch_fn=lambda ts: torch.eye(4, dtype=torch.float32),
        input_gen=_no_input,
        skip_grad=True,
        skip_gpu=True,
    ),
    OpSpec(
        name="gfunc_arange",
        engine_fn=lambda ts: E.arange(0, 10, 1, E.Dtype.F32, E.Device.CPU),
        torch_fn=lambda ts: torch.arange(0, 10, 1, dtype=torch.float32),
        input_gen=_no_input,
        skip_grad=True,
        skip_gpu=True,
    ),
    OpSpec(
        name="gfunc_linspace",
        engine_fn=lambda ts: E.linspace(0.0, 1.0, 11, E.Dtype.F32, E.Device.CPU),
        torch_fn=lambda ts: torch.linspace(0.0, 1.0, 11, dtype=torch.float32),
        input_gen=_no_input,
        skip_grad=True,
        skip_gpu=True,
    ),
    OpSpec(
        name="gfunc_full",
        engine_fn=lambda ts: E.full([3, 4], 7.0, E.Dtype.F32, E.Device.CPU),
        torch_fn=lambda ts: torch.full((3, 4), 7.0, dtype=torch.float32),
        input_gen=_no_input,
        skip_grad=True,
        skip_gpu=True,
    ),
    OpSpec(
        name="gfunc_zeros_like",
        engine_fn=lambda ts: E.zeros_like(ts[0]),
        torch_fn=lambda ts: torch.zeros_like(ts[0]),
        input_shapes=[(3, 4)],
        skip_grad=True,
    ),
    OpSpec(
        name="gfunc_ones_like",
        engine_fn=lambda ts: E.ones_like(ts[0]),
        torch_fn=lambda ts: torch.ones_like(ts[0]),
        input_shapes=[(3, 4)],
        skip_grad=True,
    ),
    OpSpec(
        name="gfunc_full_like",
        engine_fn=lambda ts: E.full_like(ts[0], 2.5),
        torch_fn=lambda ts: torch.full_like(ts[0], 2.5),
        input_shapes=[(3, 4)],
        skip_grad=True,
    ),
    OpSpec(
        name="meshgrid_xy",
        engine_fn=lambda ts: E.meshgrid([ts[0], ts[1]], True)[0],
        torch_fn=lambda ts: torch.meshgrid([ts[0], ts[1]], indexing="xy")[0],
        input_shapes=[(4,), (5,)],
        skip_grad=True,
    ),
]
