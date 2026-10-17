"""Under ``no_grad``, ``.to()`` of a tensor that requires grad returns one that does not.

The result is a new tensor made outside any graph, so the reference hands
it back with ``requires_grad=False``; Lucid copied the source's flag onto
it whatever the grad mode (LCD-486).  A ``.to()`` that changes nothing
returns the source itself, flag and all, in both.  A module converted under
``no_grad`` keeps its parameters trainable — ``.data =`` keeps their flag.
"""

import pytest

import lucid
import lucid.nn as nn
from lucid.test._fixtures.devices import metal_available

_CONVERSIONS = {
    "dtype": lambda t, lib: t.to(lib.float64),
    "copy": lambda t, lib: t.to("cpu", copy=True),
    "dtype keyword": lambda t, lib: t.to(dtype=lib.float16),
    "double": lambda t, lib: t.double(),
}


@pytest.mark.parametrize("conversion", list(_CONVERSIONS))
def test_a_conversion_under_no_grad_does_not_require_grad(conversion):
    x = lucid.ones(3, requires_grad=True)
    with lucid.no_grad():
        y = _CONVERSIONS[conversion](x, lucid)
    assert y is not x and y.is_leaf
    assert not y.requires_grad
    assert x.requires_grad


@pytest.mark.parity
@pytest.mark.parametrize("conversion", list(_CONVERSIONS))
def test_the_flag_matches_the_reference(ref, conversion):
    x, rx = lucid.ones(3, requires_grad=True), ref.ones(3, requires_grad=True)
    with lucid.no_grad(), ref.no_grad():
        y, ry = _CONVERSIONS[conversion](x, lucid), _CONVERSIONS[conversion](rx, ref)
    assert (y.requires_grad, y.is_leaf) == (ry.requires_grad, ry.is_leaf)


@pytest.mark.skipif(not metal_available(), reason="Metal not available")
@pytest.mark.parametrize("target", ["metal", "cpu"])
def test_a_device_move_under_no_grad_does_not_require_grad(target):
    source = "cpu" if target == "metal" else "metal"
    x = lucid.ones(3, device=source, requires_grad=True)
    with lucid.no_grad():
        y = x.to(target)
    assert y.device == lucid.device(target) and not y.requires_grad


def test_a_conversion_that_changes_nothing_returns_the_source():
    x = lucid.ones(3, requires_grad=True)
    with lucid.no_grad():
        assert x.to(lucid.float32) is x and x.to("cpu") is x


def test_with_grad_on_the_conversion_stays_in_the_graph():
    x = lucid.ones(3, requires_grad=True)
    y = x.to(lucid.float64)
    assert y.requires_grad and y.grad_fn is not None


@pytest.mark.parametrize(
    "convert",
    [lambda m: m.double(), lambda m: m.half(), lambda m: m.to(lucid.float64)],
    ids=["double", "half", "to"],
)
def test_a_module_converted_under_no_grad_keeps_trainable_parameters(convert):
    module = nn.Linear(2, 2)
    module.bias.requires_grad_(False)
    with lucid.no_grad():
        convert(module)
    assert module.weight.requires_grad and not module.bias.requires_grad
    assert isinstance(module.weight, nn.Parameter) and module.weight.is_leaf
