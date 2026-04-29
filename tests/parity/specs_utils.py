"""Specs for utility ops: view/layout/concat/select/sort/etc."""

from __future__ import annotations

import numpy as np
import torch

from lucid._C import engine as E

from ._specs import OpSpec


# Pair-input helper for concat/stack.
def _pair(shape):
    return [shape, shape]


SPECS: list[OpSpec] = [
    # View
    OpSpec(
        name="reshape_flat",
        engine_fn=lambda ts: E.reshape(ts[0], [6, 4]),
        torch_fn=lambda ts: ts[0].reshape(6, 4),
        input_shapes=[(2, 3, 4)],
    ),
    OpSpec(
        name="squeeze_dim0",
        engine_fn=lambda ts: E.squeeze(ts[0], 0),
        torch_fn=lambda ts: ts[0].squeeze(0),
        input_shapes=[(1, 3, 4)],
    ),
    OpSpec(
        name="unsqueeze_dim1",
        engine_fn=lambda ts: E.unsqueeze(ts[0], 1),
        torch_fn=lambda ts: ts[0].unsqueeze(1),
        input_shapes=[(3, 4)],
    ),

    # Layout
    OpSpec(
        name="broadcast_to",
        engine_fn=lambda ts: E.broadcast_to(ts[0], [3, 4, 5]),
        torch_fn=lambda ts: ts[0].broadcast_to((3, 4, 5)),
        input_shapes=[(1, 4, 5)],
    ),
    OpSpec(
        name="expand",
        engine_fn=lambda ts: E.expand(ts[0], [4, 5]),
        torch_fn=lambda ts: ts[0].expand(4, 5),
        input_shapes=[(1, 5)],
    ),
    OpSpec(
        name="flatten_all",
        engine_fn=lambda ts: E.flatten(ts[0], 0, -1),
        torch_fn=lambda ts: ts[0].flatten(0, -1),
        input_shapes=[(2, 3, 4)],
    ),

    # Concat / stack
    OpSpec(
        name="concat_axis0",
        engine_fn=lambda ts: E.concatenate([ts[0], ts[1]], 0),
        torch_fn=lambda ts: torch.cat([ts[0], ts[1]], dim=0),
        input_shapes=_pair((2, 3)),
    ),
    OpSpec(
        name="stack_axis1",
        engine_fn=lambda ts: E.stack([ts[0], ts[1]], 1),
        torch_fn=lambda ts: torch.stack([ts[0], ts[1]], dim=1),
        input_shapes=_pair((2, 3)),
    ),

    # Repeat / tile
    OpSpec(
        name="repeat_axis0_3x",
        engine_fn=lambda ts: E.repeat(ts[0], 3, 0),
        torch_fn=lambda ts: ts[0].repeat_interleave(3, dim=0),
        input_shapes=[(2, 3)],
    ),
    OpSpec(
        name="tile",
        engine_fn=lambda ts: E.tile(ts[0], [2, 3]),
        torch_fn=lambda ts: ts[0].tile((2, 3)),
        input_shapes=[(2, 3)],
    ),

    # Pad — constant value 0
    OpSpec(
        name="pad_constant",
        engine_fn=lambda ts: E.pad(ts[0], [(2, 2), (1, 1)], 0.0),
        torch_fn=lambda ts: torch.nn.functional.pad(ts[0], (1, 1, 2, 2), mode="constant", value=0.0),
        input_shapes=[(3, 4)],
    ),

    # Tri
    OpSpec(
        name="tril",
        engine_fn=lambda ts: E.tril(ts[0], 0),
        torch_fn=lambda ts: torch.tril(ts[0]),
        input_shapes=[(4, 4)],
    ),
    OpSpec(
        name="triu_off",
        engine_fn=lambda ts: E.triu(ts[0], 1),
        torch_fn=lambda ts: torch.triu(ts[0], diagonal=1),
        input_shapes=[(4, 4)],
    ),

    # Where (non-differentiable for the boolean mask itself)
    OpSpec(
        name="where",
        engine_fn=lambda ts: E.where(E.greater(ts[0], ts[1]), ts[0], ts[1]),
        torch_fn=lambda ts: torch.where(ts[0] > ts[1], ts[0], ts[1]),
        input_shapes=[(4, 5), (4, 5)],
        skip_grad=True,
        notes="Boolean mask path is non-differentiable.",
    ),

    # Diagonal — forward + backward (GPU backward via mlx::core::scatter_add)
    OpSpec(
        name="diagonal_offset0",
        engine_fn=lambda ts: E.diagonal(ts[0], 0, 0, 1),
        torch_fn=lambda ts: torch.diagonal(ts[0], offset=0, dim1=0, dim2=1),
        input_shapes=[(4, 4)],
    ),
    OpSpec(
        name="diagonal_offset1",
        engine_fn=lambda ts: E.diagonal(ts[0], 1, 0, 1),
        torch_fn=lambda ts: torch.diagonal(ts[0], offset=1, dim1=0, dim2=1),
        input_shapes=[(4, 5)],
    ),
    OpSpec(
        name="diagonal_offset_neg",
        engine_fn=lambda ts: E.diagonal(ts[0], -1, 0, 1),
        torch_fn=lambda ts: torch.diagonal(ts[0], offset=-1, dim1=0, dim2=1),
        input_shapes=[(5, 4)],
    ),
    OpSpec(
        name="diagonal_batched",
        engine_fn=lambda ts: E.diagonal(ts[0], 0, 1, 2),
        torch_fn=lambda ts: torch.diagonal(ts[0], offset=0, dim1=1, dim2=2),
        input_shapes=[(2, 4, 4)],
    ),

    # Roll
    OpSpec(
        name="roll",
        engine_fn=lambda ts: E.roll(ts[0], [1], [0]),
        torch_fn=lambda ts: torch.roll(ts[0], shifts=1, dims=0),
        input_shapes=[(4, 5)],
    ),

    # Argmax / Argmin (non-differentiable)
    OpSpec(
        name="argmax",
        engine_fn=lambda ts: E.argmax(ts[0], -1, False),
        torch_fn=lambda ts: torch.argmax(ts[0], dim=-1),
        input_shapes=[(4, 5)],
        skip_grad=True,
    ),

    # Sort / argsort / topk (non-differentiable indexing)
    OpSpec(
        name="sort_axis-1",
        engine_fn=lambda ts: E.sort(ts[0], -1),
        torch_fn=lambda ts: torch.sort(ts[0], dim=-1).values,
        input_shapes=[(4, 5)],
        skip_grad=True,
    ),
    OpSpec(
        name="argsort_axis-1",
        engine_fn=lambda ts: E.argsort(ts[0], -1),
        torch_fn=lambda ts: torch.argsort(ts[0], dim=-1),
        input_shapes=[(4, 5)],
        skip_grad=True,
    ),
    OpSpec(
        name="topk_values",
        engine_fn=lambda ts: E.topk(ts[0], 3, -1)[0],
        torch_fn=lambda ts: torch.topk(ts[0], 3, dim=-1).values,
        input_shapes=[(4, 6)],
        skip_grad=True,
    ),
    OpSpec(
        name="topk_indices",
        engine_fn=lambda ts: E.topk(ts[0], 3, -1)[1],
        torch_fn=lambda ts: torch.topk(ts[0], 3, dim=-1).indices.to(torch.int32),
        input_shapes=[(4, 6)],
        skip_grad=True,
    ),

    # Gather / scatter
    OpSpec(
        name="gather_axis-1",
        engine_fn=lambda ts: E.gather(ts[0], ts[1], -1),
        torch_fn=lambda ts: torch.gather(ts[0], -1, ts[1].long()),
        input_gen=lambda rng: [
            rng.standard_normal((4, 5)).astype("float32"),
            rng.integers(0, 5, size=(4, 3)).astype("int64"),
        ],
        skip_grad=True,
    ),

    # Split / chunk / unbind
    OpSpec(
        name="chunk_3",
        engine_fn=lambda ts: E.chunk(ts[0], 3, 1)[0],
        torch_fn=lambda ts: ts[0].chunk(3, dim=1)[0],
        input_shapes=[(4, 9)],
    ),
    OpSpec(
        name="split_at",
        engine_fn=lambda ts: E.split_at(ts[0], [3], 1)[0],
        torch_fn=lambda ts: ts[0][:, :3],
        input_shapes=[(4, 9)],
    ),

    # Stack variants
    OpSpec(
        name="hstack",
        engine_fn=lambda ts: E.hstack([ts[0], ts[1]]),
        torch_fn=lambda ts: torch.hstack([ts[0], ts[1]]),
        input_shapes=[(4, 3), (4, 5)],
    ),
    OpSpec(
        name="vstack",
        engine_fn=lambda ts: E.vstack([ts[0], ts[1]]),
        torch_fn=lambda ts: torch.vstack([ts[0], ts[1]]),
        input_shapes=[(2, 4), (3, 4)],
    ),

    # Layout micro-ops
    OpSpec(
        name="ravel",
        engine_fn=lambda ts: E.ravel(ts[0]),
        torch_fn=lambda ts: ts[0].ravel(),
        input_shapes=[(2, 3, 4)],
    ),
    OpSpec(
        name="contiguous",
        engine_fn=lambda ts: E.contiguous(ts[0]),
        torch_fn=lambda ts: ts[0].contiguous(),
        input_shapes=[(4, 5)],
    ),
    OpSpec(
        name="expand_dims",
        engine_fn=lambda ts: E.expand_dims(ts[0], 1),
        torch_fn=lambda ts: ts[0].unsqueeze(1),
        input_shapes=[(4, 5)],
    ),
    OpSpec(
        name="diag_extract",
        engine_fn=lambda ts: E.diag(ts[0], 0),
        torch_fn=lambda ts: torch.diag(ts[0], 0),
        input_shapes=[(4, 4)],
        skip_grad=True,
    ),
    OpSpec(
        name="clip",
        engine_fn=lambda ts: E.clip(ts[0], -0.5, 0.5),
        torch_fn=lambda ts: torch.clip(ts[0], -0.5, 0.5),
        input_shapes=[(4, 5)],
        skip_grad=True,
        notes="boundary clamp grad ambiguous near edge",
    ),
    OpSpec(
        name="masked_fill",
        engine_fn=lambda ts: E.masked_fill(ts[0], ts[1], 0.0),
        torch_fn=lambda ts: ts[0].masked_fill(ts[1].bool(), 0.0),
        input_gen=lambda rng: [
            rng.standard_normal((4, 5)).astype("float32"),
            (rng.standard_normal((4, 5)) > 0).astype("bool"),
        ],
        skip_grad=True,
    ),

    # Bitwise (integer / non-differentiable)
    OpSpec(
        name="bitwise_and_i32",
        engine_fn=lambda ts: E.bitwise_and(ts[0], ts[1]),
        torch_fn=lambda ts: torch.bitwise_and(ts[0], ts[1]),
        input_gen=lambda rng: [
            rng.integers(0, 16, size=(4, 5)).astype("int32"),
            rng.integers(0, 16, size=(4, 5)).astype("int32"),
        ],
        skip_grad=True,
    ),
    OpSpec(
        name="bitwise_or_i32",
        engine_fn=lambda ts: E.bitwise_or(ts[0], ts[1]),
        torch_fn=lambda ts: torch.bitwise_or(ts[0], ts[1]),
        input_gen=lambda rng: [
            rng.integers(0, 16, size=(4, 5)).astype("int32"),
            rng.integers(0, 16, size=(4, 5)).astype("int32"),
        ],
        skip_grad=True,
    ),
    OpSpec(
        name="invert_bool",
        engine_fn=lambda ts: E.invert(ts[0]),
        torch_fn=lambda ts: ~ts[0],
        input_gen=lambda rng: [(rng.integers(0, 2, size=(4, 5))).astype("bool")],
        skip_grad=True,
    ),

    # Comparisons (already in bfunc, add greater_equal/less_equal)
    OpSpec(
        name="greater_equal",
        engine_fn=lambda ts: E.greater_equal(ts[0], ts[1]),
        torch_fn=lambda ts: ts[0] >= ts[1],
        input_shapes=[(4, 5), (4, 5)],
        skip_grad=True,
    ),
    OpSpec(
        name="less_equal",
        engine_fn=lambda ts: E.less_equal(ts[0], ts[1]),
        torch_fn=lambda ts: ts[0] <= ts[1],
        input_shapes=[(4, 5), (4, 5)],
        skip_grad=True,
    ),

    # Floor division — works for negatives now (engine casts to f32 on GPU
    # then floors, matching numpy/PyTorch floor-toward-negative-infinity).
    OpSpec(
        name="floordiv_i64",
        engine_fn=lambda ts: E.floordiv(ts[0], ts[1]),
        torch_fn=lambda ts: torch.div(ts[0], ts[1], rounding_mode="floor"),
        input_gen=lambda rng: [
            rng.integers(-20, 20, size=(4, 5)).astype("int64"),
            (rng.integers(1, 8, size=(4, 5))).astype("int64"),
        ],
        skip_grad=True,
    ),

    # ---- C.2: bitwise_xor (was missing, trivial) ----
    OpSpec(
        name="bitwise_xor_i32",
        engine_fn=lambda ts: E.bitwise_xor(ts[0], ts[1]),
        torch_fn=lambda ts: torch.bitwise_xor(ts[0], ts[1]),
        input_gen=lambda rng: [
            rng.integers(0, 16, size=(4, 5)).astype("int32"),
            rng.integers(0, 16, size=(4, 5)).astype("int32"),
        ],
        skip_grad=True,
    ),

    # ---- C.4: split (was missing) ----
    OpSpec(
        name="split_uniform",
        engine_fn=lambda ts: E.split(ts[0], 3, 0)[1],
        torch_fn=lambda ts: ts[0].split(3, dim=0)[1],
        input_shapes=[(9, 4)],
        skip_grad=True,
    ),

    # ---- C.4: squeeze_all (was missing) ----
    OpSpec(
        name="squeeze_all",
        engine_fn=lambda ts: E.squeeze_all(ts[0]),
        torch_fn=lambda ts: ts[0].squeeze(),
        input_shapes=[(1, 4, 1, 5, 1)],
    ),

    # ---- C.4: unbind (was missing) ----
    OpSpec(
        name="unbind_axis0",
        engine_fn=lambda ts: E.unbind(ts[0], 0)[2],
        torch_fn=lambda ts: ts[0].unbind(0)[2],
        input_shapes=[(4, 5)],
    ),
]
