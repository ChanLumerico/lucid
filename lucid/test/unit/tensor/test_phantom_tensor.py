"""Phantom tensors: the engine's shape-only tensors (LCD-466).

While ``_C_engine.set_phantom_mode(True)`` is on for a thread, every factory
and op on that thread returns a real ``TensorImpl`` whose storage is a
phantom: the shape, dtype and device the op computes before it dispatches,
autograd state, and no values.  Nothing is allocated and no kernel runs.

The contract, as these tests hold it:

* each op's phantom result has the shape, dtype and device of its eager one,
  and the phantom section allocates nothing;
* a read of a phantom's values (``item``, ``tolist``, NumPy, bytes, ``bool``,
  printing) raises ``LucidError``;
* a phantom target takes any write; a real target refuses a phantom's values;
* the mode belongs to the thread that set it.

The tests drive the engine switch directly, not ``nn.shadow_alloc``.
"""

import contextlib
import threading
from collections.abc import Callable, Iterator

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor


@contextlib.contextmanager
def phantom() -> Iterator[None]:
    previous = _C_engine.phantom_mode()
    _C_engine.set_phantom_mode(True)
    try:
        yield
    finally:
        _C_engine.set_phantom_mode(previous)


def _allocations() -> tuple[int, int]:
    return (
        _C_engine.memory_stats(_C_engine.CPU).alloc_count,
        _C_engine.memory_stats(_C_engine.GPU).alloc_count,
    )


def _tensors(out: object) -> list[Tensor]:
    if isinstance(out, Tensor):
        return [out]
    assert isinstance(out, (tuple, list)), type(out)
    return [t for item in out for t in _tensors(item)]


# Each case builds its inputs and its result on ``d``, so the inputs are
# phantoms too when the case runs under the mode.
_CASES: dict[str, Callable[[str], object]] = {
    # factories
    "zeros": lambda d: lucid.zeros(2, 3, device=d),
    "ones": lambda d: lucid.ones(4, device=d, dtype=lucid.float16),
    "empty": lambda d: lucid.empty(3, 1, device=d),
    "full": lambda d: lucid.full((2, 2), 7.0, device=d),
    "eye": lambda d: lucid.eye(3, 5, device=d),
    "arange": lambda d: lucid.arange(0, 10, 3, device=d),
    "linspace": lambda d: lucid.linspace(0.0, 1.0, 7, device=d),
    "logspace": lambda d: lucid.logspace(0.0, 2.0, 5, device=d),
    "rand": lambda d: lucid.rand(3, 4, device=d),
    "randn": lambda d: lucid.randn(2, 5, device=d),
    "randint": lambda d: lucid.randint(0, 9, (6,), device=d),
    "zeros_like": lambda d: lucid.zeros_like(lucid.ones(2, 7, device=d)),
    "full_like": lambda d: lucid.full_like(lucid.ones(3, device=d), 2.0),
    "tensor": lambda d: lucid.tensor([[1, 2], [3, 4]], device=d),
    # elementwise, with broadcasting and promotion
    "add_broadcast": lambda d: lucid.ones(2, 1, 4, device=d) + lucid.ones(3, 1, device=d),
    "sub_scalar": lambda d: lucid.ones(3, device=d) - 1,
    "mul_promote": lambda d: lucid.ones(3, device=d) * lucid.arange(3, device=d),
    "div": lambda d: lucid.ones(2, 3, device=d) / lucid.full((3,), 2.0, device=d),
    "pow": lambda d: lucid.ones(4, device=d) ** 2,
    "exp": lambda d: lucid.exp(lucid.ones(2, 2, device=d)),
    "log": lambda d: lucid.log(lucid.ones(5, device=d)),
    "sqrt": lambda d: lucid.sqrt(lucid.ones(3, 3, device=d)),
    "tanh": lambda d: lucid.tanh(lucid.ones(4, device=d)),
    "sigmoid": lambda d: lucid.sigmoid(lucid.ones(4, device=d)),
    "relu": lambda d: F.relu(lucid.randn(3, 2, device=d)),
    "gelu": lambda d: F.gelu(lucid.randn(3, 2, device=d)),
    "maximum": lambda d: lucid.maximum(lucid.ones(3, device=d), lucid.zeros(2, 3, device=d)),
    "compare": lambda d: lucid.ones(3, device=d) > lucid.zeros(3, device=d),
    "astype": lambda d: lucid.ones(2, 2, device=d).to(lucid.float16),
    # reductions
    "sum_dim": lambda d: lucid.ones(2, 3, 4, device=d).sum(dim=1),
    "mean_keepdim": lambda d: lucid.ones(2, 3, device=d).mean(dim=0, keepdim=True),
    "sum_all": lambda d: lucid.ones(5, 6, device=d).sum(),
    "amax": lambda d: lucid.amax(lucid.ones(3, 4, device=d), dim=1),
    "argmax": lambda d: lucid.argmax(lucid.ones(3, 4, device=d), dim=0),
    "var": lambda d: lucid.var(lucid.ones(4, 5, device=d), dim=1),
    # shape and layout
    "reshape": lambda d: lucid.ones(2, 6, device=d).reshape(3, -1),
    "transpose": lambda d: lucid.ones(2, 3, device=d).transpose(0, 1),
    "permute": lambda d: lucid.ones(2, 3, 4, device=d).permute(2, 0, 1),
    "unsqueeze": lambda d: lucid.ones(3, device=d).unsqueeze(0),
    "flatten": lambda d: lucid.ones(2, 3, 4, device=d).flatten(1),
    "expand": lambda d: lucid.ones(3, 1, device=d).expand(3, 4),
    "cat": lambda d: lucid.cat([lucid.ones(2, 3, device=d), lucid.ones(4, 3, device=d)]),
    "stack": lambda d: lucid.stack([lucid.ones(3, device=d), lucid.ones(3, device=d)], dim=1),
    "split": lambda d: lucid.ones(7, 2, device=d).split(3),
    "chunk": lambda d: lucid.ones(6, 4, device=d).chunk(3, dim=1),
    "narrow": lambda d: lucid.ones(5, 4, device=d).narrow(0, 1, 3),
    "flip": lambda d: lucid.flip(lucid.ones(2, 3, device=d), dims=[1]),
    "tril": lambda d: lucid.tril(lucid.ones(4, 4, device=d)),
    "pad": lambda d: F.pad(lucid.ones(2, 3, device=d), (1, 2)),
    # linear algebra and nn
    "matmul": lambda d: lucid.ones(2, 3, 4, device=d) @ lucid.ones(4, 5, device=d),
    "softmax": lambda d: F.softmax(lucid.ones(2, 5, device=d), dim=-1),
    "log_softmax": lambda d: F.log_softmax(lucid.ones(2, 5, device=d), dim=1),
    "linear": lambda d: F.linear(lucid.ones(3, 4, device=d), lucid.ones(6, 4, device=d)),
    "conv2d": lambda d: F.conv2d(
        lucid.ones(1, 3, 8, 8, device=d), lucid.ones(4, 3, 3, 3, device=d), padding=1
    ),
    "max_pool2d": lambda d: F.max_pool2d(lucid.ones(1, 2, 8, 8, device=d), 2),
    "layer_norm": lambda d: F.layer_norm(lucid.ones(2, 6, device=d), (6,)),
}


@pytest.mark.parametrize("case", list(_CASES))
def test_a_phantom_has_the_shape_dtype_and_device_of_the_eager_result(case, device):
    build = _CASES[case]
    eager = _tensors(build(device))
    before = _allocations()
    with phantom():
        ghosts = _tensors(build(device))
    assert _allocations() == before, "the phantom section allocated"
    assert len(ghosts) == len(eager)
    for ghost, real in zip(ghosts, eager, strict=True):
        assert ghost._impl.is_phantom
        assert not real._impl.is_phantom
        assert tuple(ghost.shape) == tuple(real.shape)
        assert ghost.dtype == real.dtype
        assert ghost.device == real.device


def test_a_phantom_keeps_requires_grad_and_records_a_graph(device):
    with phantom():
        w = lucid.randn(4, 3, device=device, requires_grad=True)
        y = (lucid.ones(2, 4, device=device) @ w).sum()
    assert w._impl.is_phantom and y._impl.is_phantom
    assert w.requires_grad and w.is_leaf
    assert y.requires_grad and y.grad_fn is not None


def test_a_module_built_under_the_mode_holds_phantom_parameters(device):
    before = _allocations()
    with phantom():
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).to(device)
        out = model(lucid.ones(5, 16, device=device))
    assert _allocations() == before
    params = list(model.parameters())
    assert [tuple(p.shape) for p in params] == [(32, 16), (32,), (4, 32), (4,)]
    assert all(p._impl.is_phantom for p in params)
    assert tuple(out.shape) == (5, 4) and out._impl.is_phantom


_READS: dict[str, Callable[[Tensor], object]] = {
    "item": lambda t: t.item(),
    "tolist": lambda t: t.tolist(),
    "bool": lambda t: bool(t),
    "float": lambda t: float(t),
    "to_string": lambda t: t._impl.to_string(),
    "to_bytes": lambda t: t._impl.to_bytes(),
    "data_ptr": lambda t: t.data_ptr(),
}


@pytest.mark.parametrize("read", list(_READS))
def test_reading_a_phantom_s_values_raises(read, device):
    with phantom():
        t = lucid.ones(1, device=device)
    with pytest.raises(_C_engine.LucidError, match="phantom tensor has no values"):
        _READS[read](t)


def test_reading_a_phantom_as_numpy_raises(device):
    pytest.importorskip("numpy")
    with phantom():
        t = lucid.ones(2, 2, device=device)
    with pytest.raises(_C_engine.LucidError, match="phantom tensor has no values"):
        t.numpy()


def test_a_phantom_prints_without_values():
    # The engine refuses to render the values; the printer says so.
    with phantom():
        t = lucid.ones(2)
    assert "unavailable" in repr(t)


def test_even_an_empty_phantom_has_no_values():
    with phantom():
        t = lucid.zeros(0)
    with pytest.raises(_C_engine.LucidError, match="phantom tensor has no values"):
        t.tolist()


def test_the_mode_belongs_to_the_thread_that_set_it():
    seen: dict[str, object] = {}

    def other() -> None:
        seen["mode"] = _C_engine.phantom_mode()
        seen["values"] = lucid.arange(3).tolist()

    with phantom():
        worker = threading.Thread(target=other)
        worker.start()
        worker.join()
        assert lucid.zeros(3)._impl.is_phantom
    assert seen == {"mode": False, "values": [0, 1, 2]}


def test_a_thread_in_the_mode_leaves_the_others_real():
    ready, release = threading.Event(), threading.Event()
    seen: dict[str, bool] = {}

    def other() -> None:
        with phantom():
            seen["phantom"] = lucid.zeros(2)._impl.is_phantom
            ready.set()
            release.wait(timeout=10)

    worker = threading.Thread(target=other)
    worker.start()
    assert ready.wait(timeout=10)
    try:
        assert lucid.ones(2).tolist() == [1.0, 1.0]
        assert not _C_engine.phantom_mode()
    finally:
        release.set()
        worker.join()
    assert seen == {"phantom": True}


def test_an_engine_error_inside_the_mode_leaves_it_as_it_was():
    with phantom():
        with pytest.raises(_C_engine.LucidError):
            lucid.ones(2).item()
        with pytest.raises(ValueError):
            lucid.ones(2, 3) + lucid.ones(4, 5)
        assert _C_engine.phantom_mode()
    assert not _C_engine.phantom_mode()


def test_the_mode_is_restored_when_the_body_raises():
    with pytest.raises(KeyError), phantom():
        lucid.zeros(3)
        raise KeyError("body")
    assert not _C_engine.phantom_mode()
    assert lucid.zeros(2).tolist() == [0.0, 0.0]


def test_a_real_target_refuses_a_phantom_s_values(device):
    real = lucid.zeros(3, device=device)
    with phantom():
        ghost = lucid.ones(3, device=device)
        with pytest.raises(_C_engine.LucidError, match="phantom tensor has no values"):
            real.add_(1)
    with pytest.raises(_C_engine.LucidError, match="phantom tensor has no values"):
        real.copy_(ghost)
    assert not real._impl.is_phantom
    assert real.tolist() == [0.0, 0.0, 0.0]


def test_a_phantom_target_takes_any_write(device):
    with phantom():
        ghost = lucid.zeros(3, device=device)
        ghost.add_(1)
        ghost.mul_(lucid.ones(3, device=device))
        ghost.zero_()
    real = lucid.ones(3, device=device)
    version = ghost._impl.version
    ghost.copy_(real)
    ghost.fill_(2.0)
    assert ghost._impl.is_phantom
    assert ghost._impl.version > version


def test_after_the_mode_the_kernel_bases_take_a_phantom(device):
    # The unary, binary and reduction kernel bases give a phantom back for a
    # phantom input, mode or no mode, and so do the casts that feed them.
    with phantom():
        ghost = lucid.ones(2, 3, device=device)
    real = lucid.ones(3, device=device)
    outs = (ghost + real, lucid.exp(ghost), ghost.sum(dim=0), ghost + real.to(lucid.float16))
    assert all(out._impl.is_phantom for out in outs)
    assert [tuple(out.shape) for out in outs] == [(2, 3), (2, 3), (3,), (2, 3)]


@pytest.mark.parametrize(
    "op",
    [
        lambda t: t.to(lucid.float16),
        lambda t: t.contiguous(),
        lambda t: t.to("cpu", copy=True),
    ],
)
def test_after_the_mode_any_other_op_refuses_a_phantom(op):
    with phantom():
        ghost = lucid.ones(2, 3).transpose(0, 1)
    with pytest.raises(_C_engine.LucidError):
        op(ghost)


def test_moving_a_phantom_relabels_it(device):
    with phantom():
        ghost = lucid.ones(2, 2)
    target = _C_engine.GPU if device == "metal" else _C_engine.CPU
    moved = ghost._impl.transfer_to_device(target, True)
    assert moved.is_phantom and moved.shape == [2, 2]
    assert moved.device == target and moved.requires_grad


def test_moving_a_real_tensor_under_the_mode_copies_nothing(device):
    real = lucid.ones(4)
    before = _allocations()
    with phantom():
        moved = real.to(device, copy=True)
    assert _allocations() == before
    assert moved._impl.is_phantom and moved.device == lucid.device(device)


def test_a_real_tensor_is_not_a_phantom():
    assert not lucid.zeros(2)._impl.is_phantom
    assert not _C_engine.phantom_mode()


@pytest.mark.xfail(
    strict=True,
    reason="LCD-466 FOUND: nn/BatchNorm.cpp put_running_stat replaces a real running "
    "statistic's storage slot with the phantom update instead of refusing it",
)
def test_a_real_running_statistic_refuses_a_phantom_update():
    bn = nn.BatchNorm1d(3)
    with phantom():
        x = lucid.ones(4, 3)
        with pytest.raises(_C_engine.LucidError):
            bn(x)
    assert not bn.running_mean._impl.is_phantom
