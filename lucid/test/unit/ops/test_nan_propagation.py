"""Regression tests: relu swallowed NaN on the CPU.

Found 2026-08-02 by pushing ``[nan, inf, -inf, -1, 0, 2]`` through every
elementwise op and comparing against IEEE.

``relu(NaN)`` returned **0.0 on the CPU and NaN on Metal** — the same op
disagreeing across devices.  The CPU path used Accelerate's
``vDSP_vthres``, which returns the *threshold* for a NaN input, while the
Metal path and Accelerate's own ``vDSP_vmax`` propagate it.  Inside Lucid
that made ``relu`` disagree with ``maximum(x, 0)``, with ``relu6`` and
with ``hardtanh``, all of which were already correct.

Severity is about diagnosis rather than magnitude: a NaN that propagates
is traceable to its source, and a NaN that silently becomes zero is not —
training continues on a quietly corrupted activation.

The existing device-parity sweeps missed it because they probe with
well-conditioned uniform data.  Hence the non-finite probe here.
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F

_PROBE = np.array([np.nan, np.inf, -np.inf, -1.0, 0.0, 2.0], dtype=np.float32)
_DEVICES = ["cpu", "metal"]


def _probe(device):
    return lucid.tensor(_PROBE, device=device)


@pytest.mark.parametrize("device", _DEVICES)
def test_relu_propagates_nan(device):
    """The defect, stated directly."""
    got = F.relu(_probe(device)).numpy()
    assert np.isnan(got[0]), f"{device}: relu turned NaN into {got[0]!r}"


@pytest.mark.parametrize("device", _DEVICES)
def test_relu_matches_ieee_maximum(device):
    got = F.relu(_probe(device)).numpy()
    assert np.array_equal(got, np.maximum(_PROBE, 0), equal_nan=True)


def test_relu_agrees_across_devices_on_non_finite_input():
    """What the sweep should have been probing with all along."""
    cpu_out = F.relu(_probe("cpu"))
    metal_out = F.relu(_probe("metal"))
    # Guard the instrument: if the "metal" tensor had quietly stayed on the
    # CPU the comparison would be trivially true and prove nothing.
    assert str(cpu_out.device) == "device('cpu')"
    assert str(metal_out.device) == "device('metal')"
    assert np.array_equal(cpu_out.numpy(), metal_out.numpy(), equal_nan=True)


@pytest.mark.parametrize("device", _DEVICES)
def test_relu_agrees_with_its_other_spellings(device):
    """relu, maximum(x, 0), relu6 and hardtanh must not disagree on NaN."""
    x = _probe(device)
    zero = lucid.zeros_like(x)
    reference = lucid.maximum(x, zero).numpy()
    assert np.array_equal(F.relu(x).numpy(), reference, equal_nan=True)
    # relu6 clamps above, so only the NaN position is comparable.
    assert np.isnan(F.relu6(x).numpy()[0])
    assert np.isnan(F.hardtanh(x).numpy()[0])


@pytest.mark.parametrize("device", _DEVICES)
def test_in_place_relu_propagates_nan_too(device):
    """``relu_`` has a separate engine path and needs its own check."""
    x = _probe(device)
    with lucid.no_grad():
        out = F.relu_(x)
    assert out is x
    assert np.isnan(out.numpy()[0])


@pytest.mark.parametrize("device", _DEVICES)
def test_relu_still_rectifies(device):
    """Guard the instrument: propagating NaN must not have broken the op."""
    got = F.relu(_probe(device)).numpy()
    assert np.array_equal(got[2:], np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float32))
    assert np.isposinf(got[1])


def test_relu_gradient_unchanged():
    x = lucid.tensor(np.array([-1.0, 0.5, 2.0], dtype=np.float32))
    x.requires_grad_(True)
    F.relu(x).sum().backward()
    assert np.array_equal(x.grad.numpy(), [0.0, 1.0, 1.0])


@pytest.mark.parametrize(
    "name,fn",
    [
        ("abs", lucid.abs),
        ("exp", lucid.exp),
        ("tanh", lucid.tanh),
        ("sigmoid", lucid.sigmoid),
        ("square", lucid.square),
        ("clip", lambda t: lucid.clip(t, -1.0, 1.0)),
        ("leaky_relu", F.leaky_relu),
        ("hardtanh", F.hardtanh),
        ("relu6", F.relu6),
    ],
)
@pytest.mark.parametrize("device", _DEVICES)
def test_the_rest_of_the_family_already_propagated(name, fn, device):
    """These were correct before the fix; pinned so the fix cannot regress them."""
    assert np.isnan(fn(_probe(device)).numpy()[0]), name
