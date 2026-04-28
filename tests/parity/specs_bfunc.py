"""Specs for binary ops: arithmetic, comparisons, bitwise, contractions."""

from __future__ import annotations

import numpy as np
import torch

from lucid._C import engine as E

from ._specs import OpSpec


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _spd(rng, n, batch=()):
    """Symmetric positive-definite matrix (for cholesky/solve etc)."""
    A = rng.standard_normal(size=(*batch, n, n)).astype("float32")
    return A @ np.swapaxes(A, -1, -2) + n * np.eye(n, dtype="float32")


# --------------------------------------------------------------------------- #
# Element-wise arithmetic (scalar shapes incl. broadcasting)
# --------------------------------------------------------------------------- #

ARITH = []
for shape_a, shape_b in [
    ((4, 5), (4, 5)),
    ((4, 5), (5,)),       # right broadcast
    ((1, 5), (4, 5)),     # left broadcast
    ((2, 3, 4), (4,)),    # 3-D broadcast
]:
    for op_name, eng, ref in [
        ("add", E.add, lambda a, b: a + b),
        ("sub", E.sub, lambda a, b: a - b),
        ("mul", E.mul, lambda a, b: a * b),
        ("div", E.div, lambda a, b: a / b),
    ]:
        ARITH.append(OpSpec(
            name=f"{op_name}_{tuple(shape_a)}_{tuple(shape_b)}".replace(" ", ""),
            engine_fn=lambda ts, _e=eng: _e(ts[0], ts[1]),
            torch_fn=lambda ts, _r=ref: _r(ts[0], ts[1]),
            input_shapes=[shape_a, shape_b],
        ))

# Power: positive base to keep grads finite under f32.
def _pow_inputs(rng):
    a = rng.uniform(0.5, 2.0, size=(4, 5)).astype("float32")
    b = rng.uniform(0.5, 2.0, size=(4, 5)).astype("float32")
    return [a, b]

ARITH.append(OpSpec(
    name="pow_4x5_4x5",
    engine_fn=lambda ts: E.pow(ts[0], ts[1]),
    torch_fn=lambda ts: ts[0] ** ts[1],
    input_gen=_pow_inputs,
    atol=1e-3, rtol=1e-3,
))

ARITH.append(OpSpec(
    name="maximum_4x5",
    engine_fn=lambda ts: E.maximum(ts[0], ts[1]),
    torch_fn=lambda ts: torch.maximum(ts[0], ts[1]),
    input_shapes=[(4, 5), (4, 5)],
    skip_grad=True,  # tie-breaking ambiguity
))
ARITH.append(OpSpec(
    name="minimum_4x5",
    engine_fn=lambda ts: E.minimum(ts[0], ts[1]),
    torch_fn=lambda ts: torch.minimum(ts[0], ts[1]),
    input_shapes=[(4, 5), (4, 5)],
    skip_grad=True,
))


# --------------------------------------------------------------------------- #
# Matmul / dot family
# --------------------------------------------------------------------------- #

MATMUL = [
    OpSpec(
        name="matmul_2D",
        engine_fn=lambda ts: E.matmul(ts[0], ts[1]),
        torch_fn=lambda ts: ts[0] @ ts[1],
        input_shapes=[(3, 4), (4, 5)],
        atol=1e-3, rtol=1e-3,
    ),
    OpSpec(
        name="matmul_batched_3D",
        engine_fn=lambda ts: E.matmul(ts[0], ts[1]),
        torch_fn=lambda ts: ts[0] @ ts[1],
        input_shapes=[(2, 3, 4), (2, 4, 5)],
        atol=1e-3, rtol=1e-3,
    ),
    OpSpec(
        name="matmul_broadcast_4D",
        engine_fn=lambda ts: E.matmul(ts[0], ts[1]),
        torch_fn=lambda ts: ts[0] @ ts[1],
        input_shapes=[(2, 1, 3, 4), (1, 5, 4, 6)],
        atol=1e-3, rtol=1e-3,
    ),
    OpSpec(
        name="dot_1D",
        engine_fn=lambda ts: E.dot(ts[0], ts[1]),
        torch_fn=lambda ts: torch.dot(ts[0], ts[1]),
        input_shapes=[(7,), (7,)],
        atol=1e-4, rtol=1e-4,
    ),
    OpSpec(
        name="inner_1D",
        engine_fn=lambda ts: E.inner(ts[0], ts[1]),
        torch_fn=lambda ts: torch.inner(ts[0], ts[1]),
        input_shapes=[(5,), (5,)],
        atol=1e-4, rtol=1e-4,
    ),
    OpSpec(
        name="outer_1D",
        engine_fn=lambda ts: E.outer(ts[0], ts[1]),
        torch_fn=lambda ts: torch.outer(ts[0], ts[1]),
        input_shapes=[(4,), (3,)],
    ),
    OpSpec(
        name="tensordot_axes_2",
        engine_fn=lambda ts: E.tensordot(ts[0], ts[1], [2, 3], [0, 1]),
        torch_fn=lambda ts: torch.tensordot(ts[0], ts[1], dims=([2, 3], [0, 1])),
        input_shapes=[(2, 3, 4, 5), (4, 5, 6)],
        atol=1e-3, rtol=1e-3,
    ),
]


# --------------------------------------------------------------------------- #
# Comparisons + bitwise (non-differentiable)
# --------------------------------------------------------------------------- #

COMPARE = [
    OpSpec(
        name="equal",
        engine_fn=lambda ts: E.equal(ts[0], ts[1]),
        torch_fn=lambda ts: ts[0] == ts[1],
        input_shapes=[(4, 5), (4, 5)],
        skip_grad=True,
    ),
    OpSpec(
        name="not_equal",
        engine_fn=lambda ts: E.not_equal(ts[0], ts[1]),
        torch_fn=lambda ts: ts[0] != ts[1],
        input_shapes=[(4, 5), (4, 5)],
        skip_grad=True,
    ),
    OpSpec(
        name="greater",
        engine_fn=lambda ts: E.greater(ts[0], ts[1]),
        torch_fn=lambda ts: ts[0] > ts[1],
        input_shapes=[(4, 5), (4, 5)],
        skip_grad=True,
    ),
    OpSpec(
        name="less",
        engine_fn=lambda ts: E.less(ts[0], ts[1]),
        torch_fn=lambda ts: ts[0] < ts[1],
        input_shapes=[(4, 5), (4, 5)],
        skip_grad=True,
    ),
]


SPECS: list[OpSpec] = ARITH + MATMUL + COMPARE
