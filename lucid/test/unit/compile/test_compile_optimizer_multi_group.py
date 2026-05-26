"""Compile optimizer — multi-group (X3) acceptance.

Backbone + head model with distinct learning rates per group.
``compile_optimizer`` now wraps a multi-``param_group`` optimizer in
``_MultiGroupCompiledOptimizer`` which runs one MPSGraph executable
per group.  Verifies that 5 SGD steps match eager training within
1e-4 absolute (tighter than F16 because everything is F32).
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import compile_optimizer

from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor


class _BackboneHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(16, 32)
        self.head = nn.Linear(32, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.head(self.backbone(x).relu())


def _matched_models() -> tuple[nn.Module, nn.Module]:
    lucid.manual_seed(0)
    a = _BackboneHead().to(COMPILE_DEVICE)
    b = _BackboneHead().to(COMPILE_DEVICE)
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        pb.copy_(pa.detach().clone())
    return a, b


def _build_optimizer(model: nn.Module) -> optim.SGD:
    """SGD with two groups: backbone @ lr=1e-3, head @ lr=1e-2."""
    return optim.SGD(
        [
            {"params": list(model.backbone.parameters()), "lr": 1e-3},
            {"params": list(model.head.parameters()), "lr": 1e-2},
        ],
        lr=1e-3,  # default; per-group lr overrides
    )


def _trajectory(
    model: nn.Module, opt_factory, x: lucid.Tensor, t: lucid.Tensor, steps: int
) -> list[float]:
    opt = opt_factory(model)
    losses: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(model(x), t)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses


def _trajectory_compiled(
    model: nn.Module, opt_factory, x: lucid.Tensor, t: lucid.Tensor, steps: int
) -> list[float]:
    opt = opt_factory(model)
    copt = compile_optimizer(opt)
    losses: list[float] = []
    for _ in range(steps):
        copt.zero_grad()
        loss = F.mse_loss(model(x), t)
        loss.backward()
        copt.step()
        losses.append(float(loss.item()))
    return losses


def test_multi_group_sgd_parity() -> None:
    """Backbone(lr=1e-3) + Head(lr=1e-2): compile matches eager within 1e-4."""
    lucid.manual_seed(0)
    x = metal_tensor(8, 16)
    t = metal_tensor(8, 4)
    eager_model, comp_model = _matched_models()

    eager = _trajectory(eager_model, _build_optimizer, x, t, steps=5)
    comp = _trajectory_compiled(comp_model, _build_optimizer, x, t, steps=5)

    assert len(eager) == len(comp) == 5
    for k in range(5):
        diff = abs(eager[k] - comp[k])
        assert diff < 1e-4, (
            f"multi-group compile drift at step {k}: "
            f"eager={eager[k]:.6f}, compile={comp[k]:.6f}, diff={diff:.6f}"
        )


def test_multi_group_returns_wrapper() -> None:
    """Sanity: compile_optimizer on multi-group returns the wrapper, not the single-group class."""
    from lucid.compile._optim import _MultiGroupCompiledOptimizer

    lucid.manual_seed(0)
    model = _BackboneHead().to(COMPILE_DEVICE)
    opt = _build_optimizer(model)
    copt = compile_optimizer(opt)
    assert isinstance(copt, _MultiGroupCompiledOptimizer)
    # Lifecycle delegation works.
    assert copt.param_groups is opt.param_groups
