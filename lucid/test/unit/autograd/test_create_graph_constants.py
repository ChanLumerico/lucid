"""A create_graph gradient requires grad only when it depends on the input (LCD-422).

The engine stood a zero-derivative node behind every gradient that came back
detached, so ``grad((a * 2).sum(), a, create_graph=True)[0]`` — the constant
2 — reported ``requires_grad=True``, and ``strict`` could not tell a linear
function from a curved one.  The reference keeps a gradient in the graph when
its backward is an op of the input, even one whose own derivative is zero
(relu, hardtanh, max pooling), and leaves a constant detached.  The zero
derivative is now attached by those formulas, not by the engine.

A Python ``Function`` said "no gradient" with an empty storage, and a CPU
gradient of zero elements owns no memory: its input's ``.grad`` stayed
``None`` where the reference gives an empty tensor.
"""

import pytest

import lucid
import lucid.nn.functional as F
from lucid.autograd import Function

_VALUES = [-1.5, 0.5, 2.0, 7.0]


def _values(out):
    return out[0] if isinstance(out, tuple) else out


def _functional(lib):
    return lib.autograd if lib is lucid else lib.autograd.functional


def _zeros(lib, device):
    if lib is lucid:
        return lambda n: lucid.zeros(n, device=device)
    return lambda n: lib.zeros(n)


# name -> (call(lib, F, zeros, x)) — the scalar the gradient is taken of.
_FUNCTIONS = {
    "constant_sum": lambda L, f, z, x: x.sum(),
    "linear": lambda L, f, z, x: (x * 2).sum(),
    "affine": lambda L, f, z, x: (x * 3 + 1).sum(),
    "mean": lambda L, f, z, x: x.mean(),
    "div": lambda L, f, z, x: (x / 2).sum(),
    "neg": lambda L, f, z, x: (-x).sum(),
    "cat": lambda L, f, z, x: L.cat([x, x]).sum(),
    "index": lambda L, f, z, x: x[[0, 2]].sum(),
    "where": lambda L, f, z, x: L.where(x > 0, x, z(4)).sum(),
    "maximum": lambda L, f, z, x: L.maximum(x, z(4)).sum(),
    "sort": lambda L, f, z, x: _values(L.sort(x))[:2].sum(),
    "max": lambda L, f, z, x: x.max(),
    "floor": lambda L, f, z, x: L.floor(x).sum(),
    "quadratic": lambda L, f, z, x: (x * x).sum(),
    "exp": lambda L, f, z, x: L.exp(x).sum(),
    "relu": lambda L, f, z, x: f.relu(x).sum(),
    "relu6": lambda L, f, z, x: f.relu6(x).sum(),
    "leaky_relu": lambda L, f, z, x: f.leaky_relu(x).sum(),
    "hardtanh": lambda L, f, z, x: f.hardtanh(x).sum(),
    "hardsigmoid": lambda L, f, z, x: f.hardsigmoid(x).sum(),
    "hardswish": lambda L, f, z, x: f.hardswish(x).sum(),
    "elu": lambda L, f, z, x: f.elu(x).sum(),
    "abs": lambda L, f, z, x: L.abs(x).sum(),
    "max_pool": lambda L, f, z, x: f.max_pool1d(x.reshape(1, 1, 4), 2).sum(),
    "l1_loss": lambda L, f, z, x: f.l1_loss(x, z(4)),
    "mse_loss": lambda L, f, z, x: f.mse_loss(x, z(4)),
}


@pytest.mark.parametrize("name", list(_FUNCTIONS))
def test_requires_grad_matches_reference(ref, device, name):
    call = _FUNCTIONS[name]
    a = ref.tensor(_VALUES, requires_grad=True)
    (want,) = ref.autograd.grad(
        call(ref, ref.nn.functional, _zeros(ref, device), a), a, create_graph=True
    )
    b = lucid.tensor(_VALUES, device=device, requires_grad=True)
    (got,) = lucid.autograd.grad(call(lucid, F, _zeros(lucid, device), b), b, create_graph=True)
    assert got.requires_grad == want.requires_grad
    if not got.requires_grad:
        return
    # In the graph: a second derivative is a value, never "not reachable".
    (h,) = lucid.autograd.grad(got.sum(), b)
    try:
        (h_ref,) = ref.autograd.grad(want.sum(), a)
    except RuntimeError:  # the reference has no double backward for this op
        return
    assert h.tolist() == pytest.approx(ref.zeros(4).add(h_ref).tolist())


def test_clamp_keeps_a_zero_derivative(device):
    """``hardtanh`` is ``clamp`` in Lucid, and the reference keeps
    hardtanh's gradient in the graph; ``clamp`` keeps it too (the reference's
    clamp detaches it).  Recorded so a change here is a decision."""
    x = lucid.tensor(_VALUES, device=device, requires_grad=True)
    (g,) = lucid.autograd.grad(x.clamp(-1, 1).sum(), x, create_graph=True)
    assert g.requires_grad
    (h,) = lucid.autograd.grad(g.sum(), x)
    assert h.tolist() == [0.0] * 4


@pytest.mark.parametrize(
    "kind,fn",
    [
        ("constant", lambda a: a.sum() * 0 + 3.0),
        ("linear", lambda a: (a * 2).sum()),
        ("quadratic", lambda a: (a * a).sum()),
    ],
    ids=["constant", "linear", "quadratic"],
)
def test_hessian_strict_matches_reference(ref, device, kind, fn):
    def run(lib, x):
        try:
            return _functional(lib).hessian(fn, x, strict=True), None
        except RuntimeError as err:
            return None, err

    want, want_err = run(ref, ref.tensor([1.0, 2.0], dtype=ref.float64))
    got, got_err = run(lucid, lucid.tensor([1.0, 2.0], dtype=lucid.float32, device=device))
    assert (got_err is None) == (want_err is None), (kind, got_err, want_err)
    if want_err is None:
        assert got.flatten().tolist() == pytest.approx(want.flatten().tolist())


@pytest.mark.parametrize(
    "kind,fn",
    [
        ("linear", lambda a: a * 2),
        ("affine", lambda a: a * 3 + 1),
        ("quadratic", lambda a: a * a),
    ],
    ids=["linear", "affine", "quadratic"],
)
def test_jacobian_strict_create_graph_matches_reference(ref, device, kind, fn):
    def run(lib, x):
        try:
            return _functional(lib).jacobian(fn, x, create_graph=True, strict=True), None
        except RuntimeError as err:
            return None, err

    want, want_err = run(ref, ref.tensor([1.0, 2.0], dtype=ref.float64))
    got, got_err = run(lucid, lucid.tensor([1.0, 2.0], dtype=lucid.float32, device=device))
    assert (got_err is None) == (want_err is None), (kind, got_err, want_err)
    if want_err is None:
        assert got.flatten().tolist() == pytest.approx(want.flatten().tolist())
        assert got.requires_grad == want.requires_grad


# ── a Python Function over an empty input ────────────────────────────────────


class _Identity(Function):
    @staticmethod
    def forward(ctx, x):
        return x * 1

    @staticmethod
    def backward(ctx, g):
        return g


class _FirstOnly(Function):
    @staticmethod
    def forward(ctx, x, y):
        return x * 1

    @staticmethod
    def backward(ctx, g):
        return g, None


@pytest.mark.parametrize("shape", [(0,), (0, 0), (0, 3), (2, 0, 4)], ids=str)
def test_function_gives_an_empty_input_an_empty_gradient(device, shape):
    x = lucid.zeros(*shape, device=device, requires_grad=True)
    _Identity.apply(x).sum().backward()
    assert x.grad is not None
    assert tuple(x.grad.shape) == shape
    assert x.grad.device == x.device
    # Accumulates into what is there.
    _Identity.apply(x).sum().backward()
    assert tuple(x.grad.shape) == shape
    (g,) = lucid.autograd.grad(_Identity.apply(x).sum(), x)
    assert tuple(g.shape) == shape


@pytest.mark.parametrize("shape", [(0, 3), (3,)], ids=str)
def test_function_none_still_means_no_gradient(device, shape):
    x = lucid.zeros(*shape, device=device, requires_grad=True)
    y = lucid.zeros(*shape, device=device, requires_grad=True)
    _FirstOnly.apply(x, y).sum().backward()
    assert x.grad is not None and tuple(x.grad.shape) == shape
    assert y.grad is None
