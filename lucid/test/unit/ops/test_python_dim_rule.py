"""The Python side resolves a dim by the engine's rule (LCD-406).

The engine's ``wrap_dim`` gives a 0-d tensor one notional axis.  Python
wrappers that read ``shape[dim]``, built a permutation or checked a dim
list kept copies of the rule with the raw rank, and each copy failed
differently: ``list index out of range`` from argsort and swapaxes, a
silent no-op from squeeze on an axis past the last one.  They now share
:mod:`lucid._ops._dims`, which is held here to the engine itself, and the
ops that had no rule at all — ``roll`` without dims, ``normalize`` over
several dims — are held to the reference.
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid._ops._dims import _wrap_dim, _wrap_dims


@pytest.mark.parametrize("ndim", [0, 1, 2, 3])
def test_the_python_rule_is_the_engines(ndim):
    """Every dim the engine takes, and only those, wraps to its answer."""
    x = _C_engine.zeros([2] * ndim, _C_engine.F32, _C_engine.CPU)
    for dim in range(-ndim - 2, ndim + 2):
        try:
            # ``cumsum`` keeps the shape, so the wrapped dim is the one the
            # engine reports back by refusing or not; ``flip`` would too.
            _C_engine.cumsum(x, dim)
        except IndexError:
            with pytest.raises(IndexError, match=f"axis out of range: got {dim}"):
                _wrap_dim(dim, ndim, "op")
            continue
        assert _wrap_dim(dim, ndim, "op") == dim % max(ndim, 1)


@pytest.mark.parametrize(
    ("dims", "ndim"), [((0, -1), 0), ((1, -1), 2), ((0, 0), 3)], ids=str
)
def test_an_axis_named_twice_is_refused(dims, ndim):
    with pytest.raises(ValueError, match="appears multiple times"):
        _wrap_dims(dims, ndim, "op")


_PAST_THE_LAST_AXIS = {
    "squeeze": lambda lib, t, d: lib.squeeze(t, d),
    "squeeze_list": lambda lib, t, d: lib.squeeze(t, [d]),
    "take_along_dim": lambda lib, t, d: lib.take_along_dim(
        t, lib.zeros(t.shape, dtype=lib.int64), dim=d
    ),
    "split": lambda lib, t, d: lib.split(t, 1, dim=d),
    "chunk": lambda lib, t, d: lib.chunk(t, 1, dim=d),
    "quantile": lambda lib, t, d: lib.quantile(t, 0.5, dim=d),
    "argsort_descending": lambda lib, t, d: lib.argsort(t, dim=d, descending=True),
    "topk_smallest": lambda lib, t, d: lib.topk(t, 1, dim=d, largest=False),
    "swapaxes": lambda lib, t, d: lib.swapaxes(t, d, 0),
}


@pytest.mark.parametrize("shape", [(), (3,), (2, 1, 3)], ids=str)
@pytest.mark.parametrize("op", list(_PAST_THE_LAST_AXIS))
def test_an_axis_past_the_last_is_refused(ref, device, op, shape):
    if op in ("split", "chunk") and not shape:
        pytest.skip("a 0-d tensor has nothing to split, whatever the dim")
    call = _PAST_THE_LAST_AXIS[op]
    bad = max(len(shape), 1)
    for dim in (bad, -bad - 1):
        with pytest.raises(IndexError):
            call(ref, ref.zeros(shape), dim)
        with pytest.raises(IndexError, match=f"axis out of range: got {dim}"):
            call(lucid, lucid.zeros(shape, device=device), dim)


def test_squeeze_refuses_an_axis_named_twice(ref):
    with pytest.raises(RuntimeError, match="appears multiple times"):
        ref.squeeze(ref.zeros(1, 3), (0, -2))
    with pytest.raises(ValueError, match="appears multiple times"):
        lucid.squeeze(lucid.zeros(1, 3), (0, -2))


def _values(shape):
    n = int(np.prod(shape)) if shape else 1
    return (np.arange(n, dtype=np.float32) + 1.0).reshape(shape)


def _compare(ref, device, shape, call):
    values = _values(shape)
    x = lucid.tensor(values, device=device, requires_grad=True)
    xr = ref.tensor(values, requires_grad=True)
    got, want = call(lucid, F, x), call(ref, ref.nn.functional, xr)
    assert tuple(got.shape) == tuple(want.shape)
    np.testing.assert_allclose(got.numpy(), want.detach().numpy(), rtol=1e-5, atol=1e-6)
    weights = _values(tuple(want.shape))
    (got * lucid.tensor(weights, device=device)).sum().backward()
    (want * ref.tensor(weights)).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), xr.grad.numpy(), rtol=1e-5, atol=1e-6)


_ROLLS = {
    "no_dims": lambda lib, F, t: lib.roll(t, 2),
    "no_dims_negative": lambda lib, F, t: lib.roll(t, -1),
    "no_dims_method": lambda lib, F, t: t.roll(1),
    "int_dims": lambda lib, F, t: lib.roll(t, 1, 0),
    "tuple_dims": lambda lib, F, t: lib.roll(t, (1,), (-1,)),
}


@pytest.mark.parametrize("shape", [(), (5,), (2, 3, 4)], ids=str)
@pytest.mark.parametrize("call", list(_ROLLS))
def test_roll_matches_reference(ref, device, call, shape):
    if not shape and call in ("int_dims", "tuple_dims"):
        # The reference refuses a 0-d tensor's dim for ``roll``: only the
        # flat roll reaches it.
        with pytest.raises(IndexError):
            _ROLLS[call](ref, None, ref.tensor(1.5))
        with pytest.raises(IndexError):
            _ROLLS[call](lucid, None, lucid.tensor(1.5, device=device))
        return
    _compare(ref, device, shape, _ROLLS[call])


@pytest.mark.parametrize(
    ("shifts", "dims"), [((1, 2), None), ((1, 2), (0,))], ids=["no-dims", "misaligned"]
)
def test_roll_refuses_shifts_that_do_not_align(ref, shifts, dims):
    with pytest.raises(RuntimeError):
        ref.roll(ref.zeros(2, 3), shifts, dims)
    with pytest.raises(ValueError):
        lucid.roll(lucid.zeros(2, 3), shifts, dims)


_NORMALIZE_DIMS = [1, -1, (1,), (0, -1), [0, 1], None]


@pytest.mark.parametrize("p", [2.0, 1.0])
@pytest.mark.parametrize("dim", _NORMALIZE_DIMS, ids=str)
@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)], ids=str)
def test_normalize_takes_every_dim_spelling(ref, device, shape, dim, p):
    _compare(ref, device, shape, lambda lib, F, t: F.normalize(t, p=p, dim=dim))


@pytest.mark.parametrize("dim", [0, -1, (0,), None], ids=str)
def test_normalize_on_a_0d_tensor(ref, device, dim):
    _compare(ref, device, (), lambda lib, F, t: F.normalize(t, dim=dim))
