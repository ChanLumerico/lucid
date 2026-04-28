"""Specs for unary ops + reductions + scan + transpose."""

from __future__ import annotations

import numpy as np
import torch

from lucid._C import engine as E

from ._specs import OpSpec


# --------------------------------------------------------------------------- #
# Unary element-wise (sample one shape per family — they're all
# straight-through unary kernels, more shapes adds little signal).
# --------------------------------------------------------------------------- #

def _pos_inputs(rng):
    return [rng.uniform(0.1, 2.0, size=(4, 5)).astype("float32")]


def _bounded_inputs(rng):  # for arcsin / arccos / etc.
    return [rng.uniform(-0.9, 0.9, size=(4, 5)).astype("float32")]


UNARY = [
    OpSpec("exp",      lambda ts: E.exp(ts[0]),      lambda ts: torch.exp(ts[0]),      input_shapes=[(4, 5)]),
    OpSpec("log",      lambda ts: E.log(ts[0]),      lambda ts: torch.log(ts[0]),      input_gen=_pos_inputs, atol=1e-3, rtol=1e-3),
    OpSpec("log2",     lambda ts: E.log2(ts[0]),     lambda ts: torch.log2(ts[0]),     input_gen=_pos_inputs, atol=1e-3, rtol=1e-3),
    OpSpec("sqrt",     lambda ts: E.sqrt(ts[0]),     lambda ts: torch.sqrt(ts[0]),     input_gen=_pos_inputs, atol=1e-3, rtol=1e-3),
    OpSpec("square",   lambda ts: E.square(ts[0]),   lambda ts: torch.square(ts[0]),   input_shapes=[(4, 5)]),
    OpSpec("cube",     lambda ts: E.cube(ts[0]),     lambda ts: ts[0] ** 3,            input_shapes=[(4, 5)]),
    OpSpec("neg",      lambda ts: E.neg(ts[0]),      lambda ts: -ts[0],                input_shapes=[(4, 5)]),
    OpSpec("abs",      lambda ts: E.abs(ts[0]),      lambda ts: torch.abs(ts[0]),      input_shapes=[(4, 5)]),
    OpSpec("reciprocal", lambda ts: E.reciprocal(ts[0]), lambda ts: 1.0/ts[0], input_gen=_pos_inputs, atol=1e-3, rtol=1e-3),
    OpSpec("sin",      lambda ts: E.sin(ts[0]),      lambda ts: torch.sin(ts[0]),      input_shapes=[(4, 5)]),
    OpSpec("cos",      lambda ts: E.cos(ts[0]),      lambda ts: torch.cos(ts[0]),      input_shapes=[(4, 5)]),
    OpSpec("tan",      lambda ts: E.tan(ts[0]),      lambda ts: torch.tan(ts[0]),      input_gen=_bounded_inputs, atol=1e-3, rtol=1e-3),
    OpSpec("sinh",     lambda ts: E.sinh(ts[0]),     lambda ts: torch.sinh(ts[0]),     input_shapes=[(4, 5)]),
    OpSpec("cosh",     lambda ts: E.cosh(ts[0]),     lambda ts: torch.cosh(ts[0]),     input_shapes=[(4, 5)]),
    OpSpec("tanh",     lambda ts: E.tanh(ts[0]),     lambda ts: torch.tanh(ts[0]),     input_shapes=[(4, 5)]),
    OpSpec("arcsin",   lambda ts: E.arcsin(ts[0]),   lambda ts: torch.arcsin(ts[0]),   input_gen=_bounded_inputs, atol=1e-3, rtol=1e-3),
    OpSpec("arccos",   lambda ts: E.arccos(ts[0]),   lambda ts: torch.arccos(ts[0]),   input_gen=_bounded_inputs, atol=1e-3, rtol=1e-3),
    OpSpec("arctan",   lambda ts: E.arctan(ts[0]),   lambda ts: torch.arctan(ts[0]),   input_shapes=[(4, 5)]),
    OpSpec("sigmoid",  lambda ts: E.sigmoid(ts[0]),  lambda ts: torch.sigmoid(ts[0]),  input_shapes=[(4, 5)], atol=1e-4, rtol=1e-4),
    OpSpec("relu",     lambda ts: E.relu(ts[0]),     lambda ts: torch.relu(ts[0]),     input_shapes=[(4, 5)]),
    OpSpec("gelu",     lambda ts: E.gelu(ts[0]),     lambda ts: torch.nn.functional.gelu(ts[0]), input_shapes=[(4, 5)], atol=1e-3, rtol=1e-3),
    OpSpec("silu",     lambda ts: E.silu(ts[0]),     lambda ts: torch.nn.functional.silu(ts[0]), input_shapes=[(4, 5)], atol=1e-4, rtol=1e-4),
    OpSpec("softplus", lambda ts: E.softplus(ts[0]), lambda ts: torch.nn.functional.softplus(ts[0]), input_shapes=[(4, 5)], atol=1e-3, rtol=1e-3),
    OpSpec("softmax",  lambda ts: E.softmax(ts[0], -1),
                        lambda ts: torch.nn.functional.softmax(ts[0], dim=-1),
                        input_shapes=[(4, 5)], atol=1e-4, rtol=1e-4),
]


# --------------------------------------------------------------------------- #
# Reductions
# --------------------------------------------------------------------------- #

def _all_reductions(name, eng, ref, atol=1e-3):
    return [
        OpSpec(
            name=f"{name}_all",
            engine_fn=lambda ts, _e=eng: _e(ts[0], [], False),
            torch_fn=lambda ts, _r=ref: _r(ts[0]),
            input_shapes=[(2, 3, 4)],
            atol=atol, rtol=atol,
        ),
        OpSpec(
            name=f"{name}_axis0",
            engine_fn=lambda ts, _e=eng: _e(ts[0], [0], False),
            torch_fn=lambda ts, _r=ref: _r(ts[0], dim=0),
            input_shapes=[(2, 3, 4)],
            atol=atol, rtol=atol,
        ),
        OpSpec(
            name=f"{name}_axis_neg1_keepdim",
            engine_fn=lambda ts, _e=eng: _e(ts[0], [-1], True),
            torch_fn=lambda ts, _r=ref: _r(ts[0], dim=-1, keepdim=True),
            input_shapes=[(2, 3, 4)],
            atol=atol, rtol=atol,
        ),
    ]


REDUCE = []
REDUCE += _all_reductions("sum",  E.sum,  lambda x, **kw: x.sum(**kw))
REDUCE += _all_reductions("mean", E.mean, lambda x, **kw: x.mean(**kw))
REDUCE += _all_reductions("prod", E.prod, lambda x, **kw: x.prod(**kw) if "dim" in kw else x.prod())

# torch.max(dim=...) returns (values, indices) — wrap.
def _torch_max_axis(x, dim=None, keepdim=False):
    if dim is None:
        return x.max()
    return x.max(dim=dim, keepdim=keepdim).values

REDUCE += _all_reductions("max", E.max, _torch_max_axis)
REDUCE += _all_reductions("min", E.min, lambda x, dim=None, keepdim=False: x.min() if dim is None else x.min(dim=dim, keepdim=keepdim).values)


# --------------------------------------------------------------------------- #
# Scan: cumsum, cumprod
# --------------------------------------------------------------------------- #

SCAN = [
    OpSpec(
        name="cumsum_axis-1",
        engine_fn=lambda ts: E.cumsum(ts[0], -1),
        torch_fn=lambda ts: torch.cumsum(ts[0], dim=-1),
        input_shapes=[(2, 3, 4)],
    ),
    OpSpec(
        name="cumprod_axis0",
        engine_fn=lambda ts: E.cumprod(ts[0], 0),
        torch_fn=lambda ts: torch.cumprod(ts[0], dim=0),
        input_gen=lambda rng: [rng.uniform(0.5, 1.5, size=(3, 4)).astype("float32")],
        atol=1e-3, rtol=1e-3,
    ),
]


# --------------------------------------------------------------------------- #
# Transpose / permute / swapaxes
# --------------------------------------------------------------------------- #

TRANSPOSE = [
    OpSpec(
        name="transpose_reverse_all_3D",
        engine_fn=lambda ts: E.transpose(ts[0]),
        torch_fn=lambda ts: ts[0].permute(*reversed(range(ts[0].dim()))),
        input_shapes=[(2, 3, 4)],
    ),
    OpSpec(
        name="permute_3D",
        engine_fn=lambda ts: E.permute(ts[0], [2, 0, 1]),
        torch_fn=lambda ts: ts[0].permute(2, 0, 1),
        input_shapes=[(2, 3, 4)],
    ),
    OpSpec(
        name="swapaxes_3D",
        engine_fn=lambda ts: E.swapaxes(ts[0], 0, 2),
        torch_fn=lambda ts: ts[0].swapaxes(0, 2),
        input_shapes=[(2, 3, 4)],
    ),
]


# --------------------------------------------------------------------------- #
# Var / Trace
# --------------------------------------------------------------------------- #

EXTRA = [
    OpSpec(
        name="var_axis-1",
        engine_fn=lambda ts: E.var(ts[0], [-1], False),
        torch_fn=lambda ts: torch.var(ts[0], dim=-1, correction=0),
        input_shapes=[(4, 6)],
        atol=1e-4, rtol=1e-4,
    ),
    OpSpec(
        name="trace_2D",
        engine_fn=lambda ts: E.trace(ts[0]),
        torch_fn=lambda ts: torch.trace(ts[0]),
        input_shapes=[(5, 5)],
    ),
]


# --------------------------------------------------------------------------- #
# Activations not yet covered
# --------------------------------------------------------------------------- #

ACTIVATIONS = [
    OpSpec("elu", lambda ts: E.elu(ts[0], 1.0),
            lambda ts: torch.nn.functional.elu(ts[0], alpha=1.0),
            input_shapes=[(4, 5)], atol=1e-4, rtol=1e-4),
    OpSpec("leaky_relu", lambda ts: E.leaky_relu(ts[0], 0.1),
            lambda ts: torch.nn.functional.leaky_relu(ts[0], negative_slope=0.1),
            input_shapes=[(4, 5)]),
    OpSpec("selu", lambda ts: E.selu(ts[0]),
            lambda ts: torch.nn.functional.selu(ts[0]),
            input_shapes=[(4, 5)], atol=1e-4, rtol=1e-4),
    OpSpec("mish", lambda ts: E.mish(ts[0]),
            lambda ts: torch.nn.functional.mish(ts[0]),
            input_shapes=[(4, 5)], atol=1e-4, rtol=1e-4),
    OpSpec("hard_sigmoid", lambda ts: E.hard_sigmoid(ts[0]),
            lambda ts: torch.nn.functional.hardsigmoid(ts[0]),
            input_shapes=[(4, 5)], skip_grad=True,
            notes="boundary-jump grad differs from torch's piecewise definition"),
    OpSpec("hard_swish", lambda ts: E.hard_swish(ts[0]),
            lambda ts: torch.nn.functional.hardswish(ts[0]),
            input_shapes=[(4, 5)], skip_grad=True),
    OpSpec("relu6", lambda ts: E.relu6(ts[0]),
            lambda ts: torch.nn.functional.relu6(ts[0]),
            input_shapes=[(4, 5)], skip_grad=True),
]


# --------------------------------------------------------------------------- #
# Discrete / rounding — non-differentiable
# --------------------------------------------------------------------------- #

DISCRETE = [
    OpSpec("floor", lambda ts: E.floor(ts[0]),
            lambda ts: torch.floor(ts[0]),
            input_shapes=[(4, 5)], skip_grad=True),
    OpSpec("ceil", lambda ts: E.ceil(ts[0]),
            lambda ts: torch.ceil(ts[0]),
            input_shapes=[(4, 5)], skip_grad=True),
    OpSpec("round", lambda ts: E.round(ts[0]),
            lambda ts: torch.round(ts[0]),
            input_shapes=[(4, 5)], skip_grad=True),
    OpSpec("sign", lambda ts: E.sign(ts[0]),
            lambda ts: torch.sign(ts[0]),
            input_shapes=[(4, 5)], skip_grad=True),
]


# --------------------------------------------------------------------------- #
# Reductions — argmin / scan: cumprod (already), argsort
# --------------------------------------------------------------------------- #

REDUCE_EXTRA = [
    OpSpec("argmin",
            lambda ts: E.argmin(ts[0], -1, False),
            lambda ts: torch.argmin(ts[0], dim=-1),
            input_shapes=[(4, 5)], skip_grad=True),
]


SPECS: list[OpSpec] = (UNARY + REDUCE + SCAN + TRANSPOSE + EXTRA
                        + ACTIVATIONS + DISCRETE + REDUCE_EXTRA)
