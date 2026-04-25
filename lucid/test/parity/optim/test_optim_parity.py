import numpy as np

import pytest

import torch

import lucid

import lucid.nn as lnn

import lucid.optim as loptim

from lucid.test.parity.core import OptimTrajectoryCase, run_optim_trajectory_case


def _build_linear_models(seed: int) -> tuple[lnn.Module, torch.nn.Module]:
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((3, 5)).astype(np.float64)
    lm = lnn.Linear(5, 3, bias=False)
    lm.weight.data = W.copy()
    tm = torch.nn.Linear(5, 3, bias=False).to(torch.float64)
    with torch.no_grad():
        tm.weight.copy_(torch.as_tensor(W.copy(), dtype=torch.float64))
    return (lm, tm)


def _build_step_inputs(step: int, seed: int):
    rng = np.random.default_rng(seed * 1000 + step)
    x = rng.standard_normal((4, 5)).astype(np.float64)
    y = rng.standard_normal((4, 3)).astype(np.float64)
    return (
        lucid.tensor(x.copy()),
        lucid.tensor(y.copy()),
        torch.as_tensor(x.copy(), dtype=torch.float64),
        torch.as_tensor(y.copy(), dtype=torch.float64),
    )


def _mse(out, y):
    return ((out - y) ** 2).mean()


def _case(name, lucid_ctor, torch_ctor, *, steps=16, seed=0):
    return OptimTrajectoryCase(
        name=name,
        build_models=_build_linear_models,
        build_step_inputs=_build_step_inputs,
        lucid_optim=lucid_ctor,
        torch_optim=torch_ctor,
        lucid_loss=_mse,
        torch_loss=_mse,
        steps=steps,
        tol_class_param="optim_param",
        seed=seed,
    )


CASES: list[OptimTrajectoryCase] = [
    _case(
        "SGD_default",
        lambda p: loptim.SGD(p, lr=0.01),
        lambda p: torch.optim.SGD(p, lr=0.01),
    ),
    _case(
        "SGD_momentum",
        lambda p: loptim.SGD(p, lr=0.01, momentum=0.9),
        lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9),
    ),
    _case(
        "SGD_momentum_weight_decay",
        lambda p: loptim.SGD(p, lr=0.01, momentum=0.9, weight_decay=0.001),
        lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9, weight_decay=0.001),
    ),
    _case(
        "Adam_default",
        lambda p: loptim.Adam(p, lr=0.001),
        lambda p: torch.optim.Adam(p, lr=0.001),
    ),
    _case(
        "Adam_weight_decay",
        lambda p: loptim.Adam(p, lr=0.001, weight_decay=0.001),
        lambda p: torch.optim.Adam(p, lr=0.001, weight_decay=0.001),
    ),
    _case(
        "AdamW_default",
        lambda p: loptim.AdamW(p, lr=0.001),
        lambda p: torch.optim.AdamW(p, lr=0.001),
    ),
    _case(
        "AdamW_weight_decay",
        lambda p: loptim.AdamW(p, lr=0.001, weight_decay=0.01),
        lambda p: torch.optim.AdamW(p, lr=0.001, weight_decay=0.01),
    ),
    _case(
        "RAdam_default",
        lambda p: loptim.RAdam(p, lr=0.001),
        lambda p: torch.optim.RAdam(p, lr=0.001),
    ),
    _case(
        "NAdam_default",
        lambda p: loptim.NAdam(p, lr=0.002),
        lambda p: torch.optim.NAdam(p, lr=0.002),
        steps=20,
    ),
    _case(
        "RMSprop_default",
        lambda p: loptim.RMSprop(p, lr=0.001),
        lambda p: torch.optim.RMSprop(p, lr=0.001),
    ),
    _case(
        "Adagrad_default",
        lambda p: loptim.Adagrad(p, lr=0.01),
        lambda p: torch.optim.Adagrad(p, lr=0.01),
    ),
    _case(
        "Adadelta_default",
        lambda p: loptim.Adadelta(p, lr=1.0),
        lambda p: torch.optim.Adadelta(p, lr=1.0),
    ),
    _case(
        "Adamax_default",
        lambda p: loptim.Adamax(p, lr=2e-3),
        lambda p: torch.optim.Adamax(p, lr=2e-3),
    ),
    _case(
        "Rprop_default",
        lambda p: loptim.Rprop(p, lr=1e-2),
        lambda p: torch.optim.Rprop(p, lr=1e-2),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_optim_parity(case: OptimTrajectoryCase) -> None:
    run_optim_trajectory_case(case)
