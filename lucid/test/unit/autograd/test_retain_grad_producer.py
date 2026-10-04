"""``retain_grad`` keeps what reaches the tensor's producer slot (CHA-151).

The engine used to retain a gradient on the consumer side: each node that
read the tensor handed over, edge by edge, what it sent back to it — and only
the nodes that kept a handle on their inputs could.  A tensor read by
``cat``, by a custom ``Function`` or by nothing at all (the root of the
backward) kept ``.grad = None``, and ``autograd.grad`` retained nothing.

It is now kept where the reference keeps it: on the producer's output slot,
summed over every consumer, after the tensor's hooks — whoever the consumers
are, at the root, in ``autograd.grad``, after an in-place write (the slot
moves with the tensor), and with its graph under ``create_graph``.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid._dispatch import _unwrap
from lucid.test._helpers.compare import assert_close


def _compare(got: dict[str, object], want: dict[str, object]) -> None:
    assert got.keys() == want.keys()
    for key, w in want.items():
        g = got[key]
        if w is None or isinstance(w, bool):
            assert g == w, key
        else:
            assert_close(g, w, msg=key)


def _x(lib: ModuleType, device: str) -> object:
    return lib.tensor([1.0, -2.0, 0.5], device=device, requires_grad=True)


def _square(lib: ModuleType) -> type:
    class Square(lib.autograd.Function):  # type: ignore[name-defined, misc]
        @staticmethod
        def forward(ctx: object, a: object) -> object:
            ctx.save_for_backward(a)  # type: ignore[attr-defined]
            return a * a  # type: ignore[operator]

        @staticmethod
        def backward(ctx: object, g: object) -> object:
            (a,) = ctx.saved_tensors  # type: ignore[attr-defined]
            return 2 * a * g

    return Square


def _two_outputs(lib: ModuleType) -> type:
    class Two(lib.autograd.Function):  # type: ignore[name-defined, misc]
        @staticmethod
        def forward(ctx: object, a: object) -> object:
            return a * 2, a * 3

        @staticmethod
        def backward(ctx: object, g1: object, g2: object) -> object:
            return g1 * 2 + g2 * 3  # type: ignore[operator]

    return Two


# ── scenarios: run on Lucid and on the reference ──────────────────────────


def _through_cat(lib: ModuleType, device: str) -> dict[str, object]:
    x = _x(lib, device)
    y = x * 2
    y.retain_grad()
    z = lib.cat([y, x])
    (z * z).sum().backward()
    return {"y": y.grad, "x": x.grad}


def _through_a_function(lib: ModuleType, device: str) -> dict[str, object]:
    x = _x(lib, device)
    y = x * 2
    y.retain_grad()
    _square(lib).apply(y).sum().backward()
    return {"y": y.grad, "x": x.grad}


def _at_the_root(lib: ModuleType, device: str) -> dict[str, object]:
    x = _x(lib, device)
    s = (x * 2).sum()
    s.retain_grad()
    s.backward()
    return {"s": s.grad, "x": x.grad}


def _a_function_output(lib: ModuleType, device: str) -> dict[str, object]:
    a = lib.tensor([1.0, -0.5], device=device, requires_grad=True)
    o1, o2 = _two_outputs(lib).apply(a)
    o2.retain_grad()
    ((o1 * o1).sum() + (5 * o2).sum() + (o2 * o2).sum()).backward()
    return {"o2": o2.grad, "a": a.grad}


def _a_module_output(lib: ModuleType, device: str) -> dict[str, object]:
    lin = lib.nn.Linear(2, 2, bias=False)
    with lib.no_grad():
        lin.weight.copy_(lib.tensor([[1.0, 0.5], [-1.0, 2.0]]))
    lin = lin.to(device)
    lin.register_full_backward_hook(lambda mod, gi, go: None)
    x = lib.tensor([[1.0, -2.0]], device=device, requires_grad=True)
    y = lin(x)
    y.retain_grad()
    ((3 * y).sum() + (y * y).sum()).backward()
    return {"y": y.grad, "x": x.grad}


def _consumers_before_and_after(lib: ModuleType, device: str) -> dict[str, object]:
    x = _x(lib, device)
    y = x * 2
    early = y * 3
    y.retain_grad()
    late = y * y
    (early.sum() + lib.cat([late, y]).sum()).backward()
    return {"y": y.grad, "x": x.grad}


def _in_autograd_grad(lib: ModuleType, device: str) -> dict[str, object]:
    x = _x(lib, device)
    y = x * 2
    y.retain_grad()
    w = lib.ones(3, device=device, requires_grad=True)
    v = w * 4
    v.retain_grad()
    (gx,) = lib.autograd.grad((y * y).sum() + v.sum(), x)
    # v is off the path to x: nothing reaches it.
    return {"y": y.grad, "gx": gx, "v": v.grad, "x": x.grad}


def _after_an_in_place_write(lib: ModuleType, device: str) -> dict[str, object]:
    x = _x(lib, device)
    y = x * 2
    y.retain_grad()
    y.mul_(3)
    (y * y).sum().backward()
    return {"y": y.grad, "x": x.grad}


def _after_a_write_through_a_view(lib: ModuleType, device: str) -> dict[str, object]:
    # The write moves y to a new place in the graph; its retained gradient is
    # the one of its new place (main gave [20, 20, 5], the old place's).
    x = _x(lib, device)
    y = x * 2
    y.retain_grad()
    y[0:2].mul_(3)
    (y * 5).sum().backward()
    return {"y": y.grad, "x": x.grad}


def _after_copy(lib: ModuleType, device: str) -> dict[str, object]:
    # copy_ gives y the source's place in the graph; retain_grad follows it.
    x = _x(lib, device)
    y = x * 2
    y.retain_grad()
    z = lib.tensor([9.0, 9.0, 9.0], device=device, requires_grad=True)
    y.copy_(z * 3)
    (y * 5).sum().backward()
    return {"y": y.grad, "z": z.grad}


def _after_setitem(lib: ModuleType, device: str) -> dict[str, object]:
    x = _x(lib, device)
    y = x * 2
    y.retain_grad()
    y[0] = 7.0
    (y * 5).sum().backward()
    return {"y": y.grad, "x": x.grad}


def _create_graph(lib: ModuleType, device: str) -> dict[str, object]:
    x = _x(lib, device)
    y = lib.cat([x * 2, x])
    y.retain_grad()
    (y * y).sum().backward(create_graph=True)
    assert y.grad is not None
    carries_graph = y.grad.grad_fn is not None
    y_grad = y.grad.detach().clone()
    (h,) = lib.autograd.grad(y.grad.sum(), x)
    return {"y": y_grad, "carries_graph": carries_graph, "h": h}


def _two_passes(lib: ModuleType, device: str) -> dict[str, object]:
    x = _x(lib, device)
    y = x * 2
    y.retain_grad()
    loss = lib.cat([y, y]).sum()
    loss.backward(retain_graph=True)
    loss.backward()
    return {"y": y.grad, "x": x.grad}


_SCENARIOS: dict[str, Callable[[ModuleType, str], dict[str, object]]] = {
    "through-cat": _through_cat,
    "through-a-function": _through_a_function,
    "at-the-root": _at_the_root,
    "a-function-output": _a_function_output,
    "a-module-output": _a_module_output,
    "consumers-before-and-after": _consumers_before_and_after,
    "in-autograd-grad": _in_autograd_grad,
    "after-an-in-place-write": _after_an_in_place_write,
    "after-a-write-through-a-view": _after_a_write_through_a_view,
    "after-copy": _after_copy,
    "after-setitem": _after_setitem,
    "create-graph": _create_graph,
    "two-passes": _two_passes,
}


#: ``y[0] = v`` rebinds the Python tensor to a new impl (``_tensor/_indexing.py``
#: ``_rebind``) that does not carry the retain flag, so ``.grad`` stays None —
#: on main as well.  A Metal view writes back into its base the same way
#: (``_tensor/_metal_views.py`` ``write_back``).  The fix is on the Python side
#: (CHA-151-A): these turn into failures, as a reminder, once it lands.
_SETITEM_REBINDS = {"after-setitem": None, "after-a-write-through-a-view": "metal"}


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_matches_the_reference(name: str, ref: ModuleType, device: str) -> None:
    if name in _SETITEM_REBINDS and _SETITEM_REBINDS[name] in (None, device):
        pytest.xfail("setitem rebinds the impl without the retain flag (CHA-151-A)")
    _compare(_SCENARIOS[name](lucid, device), _SCENARIOS[name](ref, "cpu"))


@pytest.mark.parametrize("name", sorted(_SETITEM_REBINDS))
def test_setitem_still_loses_the_retain_flag(name: str, device: str) -> None:
    # Pins the gap above until the Python side carries the flag over.
    if _SETITEM_REBINDS[name] not in (None, device):
        pytest.skip("the engine handles this write on this device")
    assert _SCENARIOS[name](lucid, device)["y"] is None


# ── without the reference ──────────────────────────────────────────────────


def test_the_reported_gaps(device: str) -> None:
    x = lucid.ones(3, device=device, requires_grad=True)
    y = x * 2
    y.retain_grad()
    lucid.cat([y, x]).sum().backward()
    assert y.grad is not None and y.grad.tolist() == [1.0, 1.0, 1.0]

    s = (x * 2).sum()
    s.retain_grad()
    s.backward()
    assert s.grad is not None and s.grad.item() == 1.0


def test_a_retained_gradient_is_a_buffer_of_its_own(device: str) -> None:
    # It is the gradient that flows on into y's producer, which is added into
    # in place: they must not share a buffer.
    x = lucid.ones(3, device=device, requires_grad=True)
    y = x * 1.0
    y.retain_grad()
    z = y.view(3)
    (z.sum() + (2 * y).sum()).backward()
    assert y.grad is not None and y.grad.tolist() == [3.0, 3.0, 3.0]
    assert x.grad is not None and x.grad.tolist() == [3.0, 3.0, 3.0]


def test_a_leaf_only_keeps_the_flag() -> None:
    x = lucid.ones(3, requires_grad=True)
    x.retain_grad()
    assert _unwrap(x).retains_grad
    (x * 2).sum().backward()
    assert x.grad is not None and x.grad.tolist() == [2.0, 2.0, 2.0]
