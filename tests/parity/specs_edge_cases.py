"""Edge-case specs: empty / single-element / large shapes, F64 dtype paths.

These complement the per-family specs by stressing boundary conditions that
shape-by-shape spec writers tend to miss. Add new edge cases here, not in
the per-family files, so they form a focused regression sweep.
"""

from __future__ import annotations

import numpy as np
import torch

from lucid._C import engine as E

from ._specs import OpSpec


# --------------------------------------------------------------------------- #
# Empty / single-element shapes — element-wise & reduction sanity.
# --------------------------------------------------------------------------- #

EMPTY_SINGLE = [
    OpSpec(
        name="edge_add_single_elem",
        engine_fn=lambda ts: E.add(ts[0], ts[1]),
        torch_fn=lambda ts: ts[0] + ts[1],
        input_shapes=[(1,), (1,)],
    ),
    OpSpec(
        name="edge_mul_scalar_to_3d",
        engine_fn=lambda ts: E.mul(ts[0], ts[1]),
        torch_fn=lambda ts: ts[0] * ts[1],
        input_shapes=[(2, 3, 4), (1,)],
    ),
    OpSpec(
        name="edge_relu_single",
        engine_fn=lambda ts: E.relu(ts[0]),
        torch_fn=lambda ts: torch.relu(ts[0]),
        input_shapes=[(1,)],
    ),
    OpSpec(
        name="edge_sum_single",
        engine_fn=lambda ts: E.sum(ts[0], [], False),
        torch_fn=lambda ts: ts[0].sum(),
        input_shapes=[(1,)],
    ),
    # Empty (numel=0): exp/relu of an empty vector.
    OpSpec(
        name="edge_relu_empty",
        engine_fn=lambda ts: E.relu(ts[0]),
        torch_fn=lambda ts: torch.relu(ts[0]),
        input_shapes=[(0,)],
        skip_grad=True,
        notes="empty input has no autograd contribution",
    ),
    OpSpec(
        name="edge_add_empty",
        engine_fn=lambda ts: E.add(ts[0], ts[1]),
        torch_fn=lambda ts: ts[0] + ts[1],
        input_shapes=[(0, 4), (0, 4)],
        skip_grad=True,
    ),
]


# --------------------------------------------------------------------------- #
# Large tensors — exercise BLAS / MLX large-batch paths.
# --------------------------------------------------------------------------- #

LARGE = [
    OpSpec(
        name="edge_matmul_large_512",
        engine_fn=lambda ts: E.matmul(ts[0], ts[1]),
        torch_fn=lambda ts: ts[0] @ ts[1],
        input_shapes=[(512, 256), (256, 128)],
        atol=1e-2, rtol=1e-2,
        notes="larger sgemm — accumulator drift stretches f32 tolerance",
    ),
    OpSpec(
        name="edge_softmax_long_axis",
        engine_fn=lambda ts: E.softmax(ts[0], -1),
        torch_fn=lambda ts: torch.softmax(ts[0], dim=-1),
        input_shapes=[(2, 4096)],
        atol=1e-4, rtol=1e-4,
    ),
    OpSpec(
        name="edge_reduce_sum_large",
        engine_fn=lambda ts: E.sum(ts[0], [], False),
        torch_fn=lambda ts: ts[0].sum(),
        input_shapes=[(8192,)],
        atol=1e-2, rtol=1e-2,  # f32 accumulation drift over 8K elements
    ),
]


# --------------------------------------------------------------------------- #
# Float64 paths — most ops were exercised in F32 only.
#
# F64 lives on CPU only; MLX-Metal rejects float64. Every F64 spec carries
# `skip_gpu=True` so the parity harness only checks the CPU path.
# --------------------------------------------------------------------------- #

F64 = [
    OpSpec(
        name="edge_add_f64",
        engine_fn=lambda ts: E.add(ts[0], ts[1]),
        torch_fn=lambda ts: ts[0] + ts[1],
        input_shapes=[(4, 5), (4, 5)],
        dtype="float64",
        atol=1e-12, rtol=1e-12,
        skip_gpu=True,
    ),
    OpSpec(
        name="edge_matmul_f64",
        engine_fn=lambda ts: E.matmul(ts[0], ts[1]),
        torch_fn=lambda ts: ts[0] @ ts[1],
        input_shapes=[(3, 4), (4, 5)],
        dtype="float64",
        atol=1e-10, rtol=1e-10,
        skip_gpu=True,
    ),
    OpSpec(
        name="edge_exp_f64",
        engine_fn=lambda ts: E.exp(ts[0]),
        torch_fn=lambda ts: torch.exp(ts[0]),
        input_shapes=[(4, 5)],
        dtype="float64",
        atol=1e-12, rtol=1e-12,
        skip_gpu=True,
    ),
    OpSpec(
        name="edge_softmax_f64",
        engine_fn=lambda ts: E.softmax(ts[0], -1),
        torch_fn=lambda ts: torch.softmax(ts[0], dim=-1),
        input_shapes=[(4, 5)],
        dtype="float64",
        atol=1e-12, rtol=1e-12,
        skip_gpu=True,
    ),
    OpSpec(
        name="edge_inv_f64",
        engine_fn=lambda ts: E.linalg.inv(ts[0]),
        torch_fn=lambda ts: torch.linalg.inv(ts[0]),
        input_gen=lambda rng: [
            (rng.standard_normal((4, 4)) + 4 * np.eye(4)).astype("float64")
        ],
        dtype="float64",
        atol=1e-10, rtol=1e-10,
        skip_grad=True,
        skip_gpu=True,
    ),
    OpSpec(
        name="edge_sum_f64",
        engine_fn=lambda ts: E.sum(ts[0], [-1], False),
        torch_fn=lambda ts: ts[0].sum(dim=-1),
        input_shapes=[(4, 5)],
        dtype="float64",
        atol=1e-12, rtol=1e-12,
        skip_gpu=True,
    ),
]


# --------------------------------------------------------------------------- #
# High-dim shapes — make sure 5+ dim flows aren't silently broken.
# --------------------------------------------------------------------------- #

HIGH_DIM = [
    OpSpec(
        name="edge_permute_5d",
        engine_fn=lambda ts: E.permute(ts[0], [4, 0, 2, 1, 3]),
        torch_fn=lambda ts: ts[0].permute(4, 0, 2, 1, 3),
        input_shapes=[(2, 3, 4, 5, 6)],
    ),
    OpSpec(
        name="edge_sum_5d_multi_axis",
        engine_fn=lambda ts: E.sum(ts[0], [1, 3], False),
        torch_fn=lambda ts: ts[0].sum(dim=(1, 3)),
        input_shapes=[(2, 3, 4, 5, 6)],
        atol=1e-4, rtol=1e-4,
    ),
    OpSpec(
        name="edge_broadcast_2d_to_5d",
        engine_fn=lambda ts: E.broadcast_to(ts[0], [2, 3, 4, 5, 6]),
        torch_fn=lambda ts: ts[0].broadcast_to((2, 3, 4, 5, 6)),
        input_shapes=[(4, 1, 6)],
    ),
]


SPECS: list[OpSpec] = EMPTY_SINGLE + LARGE + F64 + HIGH_DIM
