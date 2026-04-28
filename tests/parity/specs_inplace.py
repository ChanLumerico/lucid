"""Specs for in-place op variants (`_` suffix).

In-place ops mutate input and bump its version. We verify the resulting
buffer matches the equivalent out-of-place op. Backward is intentionally
skipped — in-place mutation breaks autograd by design (version-mismatch
guard fires), and these are typically used outside autograd graphs.
"""

from __future__ import annotations

import numpy as np
import torch

from lucid._C import engine as E

from ._specs import OpSpec


def _pos(rng):
    return [rng.uniform(0.1, 2.0, size=(4, 5)).astype("float32")]


def _bounded(rng):
    return [rng.uniform(-0.9, 0.9, size=(4, 5)).astype("float32")]


def _make_unary(name, eng_inplace, ref, gen=None):
    def engine_fn(ts):
        # in-place returns the same TensorImpl; we run on a copy each time.
        eng_inplace(ts[0])
        return ts[0]
    spec_kwargs = {"input_shapes": [(4, 5)]} if gen is None else {"input_gen": gen}
    return OpSpec(
        name=f"inplace_{name}",
        engine_fn=engine_fn,
        torch_fn=lambda ts: ref(ts[0]),
        atol=1e-3, rtol=1e-3,
        skip_grad=True,
        **spec_kwargs,
    )


UNARY = [
    _make_unary("abs",        lambda t: E.abs_(t),        torch.abs),
    _make_unary("neg",        lambda t: E.neg_(t),        lambda x: -x),
    _make_unary("ceil",       lambda t: E.ceil_(t),       torch.ceil),
    _make_unary("floor",      lambda t: E.floor_(t),      torch.floor),
    _make_unary("round",      lambda t: E.round_(t),      torch.round),
    _make_unary("sign",       lambda t: E.sign_(t),       torch.sign),
    _make_unary("sqrt",       lambda t: E.sqrt_(t),       torch.sqrt, _pos),
    _make_unary("square",     lambda t: E.square_(t),     torch.square),
    _make_unary("cube",       lambda t: E.cube_(t),       lambda x: x ** 3),
    _make_unary("reciprocal", lambda t: E.reciprocal_(t), torch.reciprocal, _pos),
    _make_unary("exp",        lambda t: E.exp_(t),        torch.exp),
    _make_unary("log",        lambda t: E.log_(t),        torch.log, _pos),
    _make_unary("log2",       lambda t: E.log2_(t),       torch.log2, _pos),
    _make_unary("sin",        lambda t: E.sin_(t),        torch.sin),
    _make_unary("cos",        lambda t: E.cos_(t),        torch.cos),
    _make_unary("tan",        lambda t: E.tan_(t),        torch.tan, _bounded),
    _make_unary("sinh",       lambda t: E.sinh_(t),       torch.sinh),
    _make_unary("cosh",       lambda t: E.cosh_(t),       torch.cosh),
    _make_unary("tanh",       lambda t: E.tanh_(t),       torch.tanh),
    _make_unary("arcsin",     lambda t: E.arcsin_(t),     torch.arcsin, _bounded),
    _make_unary("arccos",     lambda t: E.arccos_(t),     torch.arccos, _bounded),
    _make_unary("arctan",     lambda t: E.arctan_(t),     torch.arctan),
]

# Binary in-place
def _binary_inplace(name, eng_inplace, ref):
    def engine_fn(ts):
        eng_inplace(ts[0], ts[1])
        return ts[0]
    return OpSpec(
        name=f"inplace_{name}",
        engine_fn=engine_fn,
        torch_fn=lambda ts: ref(ts[0], ts[1]),
        input_shapes=[(4, 5), (4, 5)],
        atol=1e-4, rtol=1e-4,
        skip_grad=True,
    )

BINARY = [
    _binary_inplace("add", lambda a, b: E.add_(a, b), lambda a, b: a + b),
    _binary_inplace("sub", lambda a, b: E.sub_(a, b), lambda a, b: a - b),
    _binary_inplace("mul", lambda a, b: E.mul_(a, b), lambda a, b: a * b),
    _binary_inplace("div", lambda a, b: E.div_(a, b),
                     lambda a, b: a / b),
    _binary_inplace("maximum", lambda a, b: E.maximum_(a, b), torch.maximum),
    _binary_inplace("minimum", lambda a, b: E.minimum_(a, b), torch.minimum),
]

POW = [
    OpSpec(
        name="inplace_pow",
        engine_fn=(lambda ts: (E.pow_(ts[0], ts[1]), ts[0])[1]),
        torch_fn=lambda ts: ts[0] ** ts[1],
        input_gen=lambda rng: [
            rng.uniform(0.5, 2.0, size=(4, 5)).astype("float32"),
            rng.uniform(0.5, 2.0, size=(4, 5)).astype("float32"),
        ],
        atol=1e-3, rtol=1e-3,
        skip_grad=True,
    ),
]

# clip with min/max scalars
CLIP = [
    OpSpec(
        name="inplace_clip",
        engine_fn=(lambda ts: (E.clip_(ts[0], -0.5, 0.5), ts[0])[1]),
        torch_fn=lambda ts: torch.clip(ts[0], -0.5, 0.5),
        input_shapes=[(4, 5)],
        atol=1e-4, rtol=1e-4,
        skip_grad=True,
    ),
]


SPECS: list[OpSpec] = UNARY + BINARY + POW + CLIP
