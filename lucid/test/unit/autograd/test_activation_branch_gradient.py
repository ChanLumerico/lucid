"""Activation gradients at NaN, signed zero, infinity and the bounds (LCD-428).

A piecewise backward picks its branch by a comparison on the input, and a
NaN fails every comparison, so which side a NaN lands on is decided by the
comparison the backward makes.  The reference answers with its vectorised
kernels on any real input and with a scalar loop on a tensor too short to
fill a vector, and the two disagree at NaN (relu6: 0 against 1, elu: NaN
against 1).  The oracle is the vectorised path — what training hits — so
the reference's gradient is read off a tensor of ``_WIDTH`` elements.
Lucid's relu dropped a NaN's gradient where both paths pass it, leaky_relu
gave 1 at 0 where the reference gives the slope, and selu and elu with
alpha != 1 took the positive side at 0.  Each activation is held to the
reference through ``backward()`` and ``grad(create_graph=True)`` on both
devices.
"""

import math

import pytest

import lucid
import lucid.nn.functional as F

_VALUES = [
    float("nan"),
    0.0,
    -0.0,
    float("inf"),
    float("-inf"),
    1.0,
    -1.0,
    3.0,
    -3.0,
    6.0,
    0.5,
    -0.5,
]

# name -> call(F-like module, x); the reference's functional module and
# Lucid's take the same arguments for every entry.
_ACTIVATIONS = {
    "relu": lambda f, x: f.relu(x),
    "relu6": lambda f, x: f.relu6(x),
    "leaky_relu": lambda f, x: f.leaky_relu(x, 0.1),
    "elu": lambda f, x: f.elu(x),
    "elu_alpha": lambda f, x: f.elu(x, 0.5),
    "selu": lambda f, x: f.selu(x),
    "hardswish": lambda f, x: f.hardswish(x),
    "hardsigmoid": lambda f, x: f.hardsigmoid(x),
    "threshold": lambda f, x: f.threshold(x, 0.5, -2.0),
    "softplus": lambda f, x: f.softplus(x),
    "silu": lambda f, x: f.silu(x),
    "gelu": lambda f, x: f.gelu(x),
    "mish": lambda f, x: f.mish(x),
    "hardtanh": lambda f, x: f.hardtanh(x, -1.0, 1.0),
    "celu": lambda f, x: f.celu(x),
    "hardshrink": lambda f, x: f.hardshrink(x, 0.5),
    "softshrink": lambda f, x: f.softshrink(x, 0.5),
}

# name -> call(library, x) for the tensor-level ops.
_TENSOR_OPS = {
    "clamp": lambda lib, x: lib.clamp(x, -1.0, 1.0),
    "clamp_min": lambda lib, x: lib.clamp(x, min=0.0),
    "clamp_max": lambda lib, x: lib.clamp(x, max=0.0),
    "maximum": lambda lib, x: lib.maximum(x, lib.zeros_like(x)),
    "minimum": lambda lib, x: lib.minimum(x, lib.zeros_like(x)),
}

_PYTHON_SIDE = {
    "hardtanh": "LCD-428 follow-up (api): F.hardtanh is lucid.clamp, whose backward "
    "passes the gradient at the bound; the reference's hardtanh backward gives 0 there",
    "celu": "LCD-428 follow-up (api): F.celu is a relu + minimum composite; the "
    "reference differentiates it as elu, so 0 takes the negative side and +inf gives 1",
}
# The cells each Python-side composite gets wrong today, by input value.
_KNOWN_WRONG = {
    "hardtanh": lambda v: v in (-1.0, 1.0),
    "celu": lambda v: v == 0.0 or v == float("inf"),
}
_TIE_RULE = (
    "LCD-428 follow-up (engine): maximum/minimum send a tie's gradient to the "
    "first operand and none to a NaN; the reference splits a tie in half and "
    "gives a NaN the whole gradient"
)


def _ids(v):
    return str(v)


def _same(got, want):
    if math.isnan(want):
        return math.isnan(got)
    return got == pytest.approx(want, rel=1e-5, abs=1e-6)


# Wide enough that the reference runs its vectorised kernel, not the
# scalar loop it keeps for the tail of a short tensor.
_WIDTH = 32


def _reference_grad(ref, call, value, dtype):
    x = ref.tensor([value] * _WIDTH, dtype=dtype, requires_grad=True)
    call(x).sum().backward()
    return float(x.grad[0].item())


def _lucid_grads(call, value, device, dtype):
    x = lucid.tensor([value] * _WIDTH, dtype=dtype, device=device, requires_grad=True)
    call(x).sum().backward()
    eager = float(x.grad[0].item())
    x2 = lucid.tensor([value] * _WIDTH, dtype=dtype, device=device, requires_grad=True)
    (g,) = lucid.autograd.grad(call(x2).sum(), x2, create_graph=True)
    return eager, float(g[0].item())


def _check(ref, device, dtype_name, value, lucid_call, ref_call):
    if device == "metal" and dtype_name == "float64":
        pytest.skip("Metal has no float64")
    want = _reference_grad(ref, ref_call, value, getattr(ref, dtype_name))
    eager, graph = _lucid_grads(lucid_call, value, device, getattr(lucid, dtype_name))
    assert _same(eager, want), f"backward(): {eager} != reference {want}"
    assert _same(graph, want), f"grad(create_graph=True): {graph} != reference {want}"


def _pin(request, reason):
    request.applymarker(pytest.mark.xfail(strict=True, reason=reason))


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("value", _VALUES, ids=_ids)
@pytest.mark.parametrize("name", list(_ACTIVATIONS))
def test_activation_gradient_matches_reference(ref, device, name, value, dtype_name, request):
    call = _ACTIVATIONS[name]
    if name in _KNOWN_WRONG and _KNOWN_WRONG[name](value):
        _pin(request, _PYTHON_SIDE[name])
    _check(
        ref,
        device,
        dtype_name,
        value,
        lambda x: call(F, x),
        lambda x: call(ref.nn.functional, x),
    )


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("value", _VALUES, ids=_ids)
@pytest.mark.parametrize("name", list(_TENSOR_OPS))
def test_tensor_op_gradient_matches_reference(ref, device, name, value, dtype_name, request):
    call = _TENSOR_OPS[name]
    if name in ("maximum", "minimum") and (math.isnan(value) or value == 0.0):
        _pin(request, _TIE_RULE)
    _check(
        ref,
        device,
        dtype_name,
        value,
        lambda x: call(lucid, x),
        lambda x: call(ref, x),
    )
