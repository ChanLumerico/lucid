"""A compiled training step differentiates the tensor ``loss_fn`` returned.

It used to take the output of the last op traced.  A model that computes
its loss and then goes on — Dreamer's forward returns reconstructions and
metrics computed after the loss — had one of those differentiated in the
loss's place: the step returned 0.0008 against eager's 2120, with
non-finite gradients, and raised nothing.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn

from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")


def _loss_then_more(y: lucid.Tensor) -> lucid.Tensor:
    loss = (y * y).sum()
    _metric = y.abs().mean() * 3.0 + 1.0  # traced after the loss, discarded
    return loss


def _setup() -> tuple[nn.Module, lucid.Tensor]:
    lucid.manual_seed(0)
    model = nn.Linear(4, 3).to(COMPILE_DEVICE)
    return model, lucid.randn(5, 4).to(COMPILE_DEVICE)


def _eager(model: nn.Module, x: lucid.Tensor) -> tuple[float, list[np.ndarray]]:
    for p in model.parameters():
        p.grad = None
    loss = _loss_then_more(model(x))
    loss.backward()
    return float(loss.item()), [p.grad.numpy().copy() for p in model.parameters()]


def test_make_step_uses_the_returned_loss() -> None:
    model, x = _setup()
    want_loss, want = _eager(model, x)
    step = lucid.compile.make_step(model, _loss_then_more)
    step(x).backward()
    for p in model.parameters():
        p.grad = None
    loss = step(x)
    loss.backward()
    assert not step.eager_only
    assert np.isclose(float(loss.item()), want_loss, rtol=1e-5)
    for p, w in zip(model.parameters(), want):
        assert np.allclose(p.grad.numpy(), w, rtol=1e-4, atol=1e-5)


def test_compiled_step_uses_the_returned_loss() -> None:
    model, x = _setup()
    want_loss, want = _eager(model, x)
    for p in model.parameters():
        p.grad = None
    loss = lucid.compile.compiled_step(model, x, _loss_then_more)
    assert np.isclose(float(loss.item()), want_loss, rtol=1e-5)
    for p, w in zip(model.parameters(), want):
        assert np.allclose(p.grad.numpy(), w, rtol=1e-4, atol=1e-5)
