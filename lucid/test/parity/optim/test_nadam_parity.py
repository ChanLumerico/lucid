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


def _lucid_mse(out, y):
    return ((out - y) ** 2).mean()


def _torch_mse(out, y):
    return ((out - y) ** 2).mean()


CASES: list[OptimTrajectoryCase] = [
    OptimTrajectoryCase(
        name="nadam_default",
        build_models=_build_linear_models,
        build_step_inputs=_build_step_inputs,
        lucid_optim=lambda params: loptim.NAdam(params, lr=0.002),
        torch_optim=lambda params: torch.optim.NAdam(params, lr=0.002),
        lucid_loss=_lucid_mse,
        torch_loss=_torch_mse,
        steps=20,
        tol_class_param="optim_param",
        seed=0,
    ),
    OptimTrajectoryCase(
        name="nadam_high_momentum_decay",
        build_models=_build_linear_models,
        build_step_inputs=_build_step_inputs,
        lucid_optim=lambda params: loptim.NAdam(params, lr=0.001, momentum_decay=0.008),
        torch_optim=lambda params: torch.optim.NAdam(
            params, lr=0.001, momentum_decay=0.008
        ),
        lucid_loss=_lucid_mse,
        torch_loss=_torch_mse,
        steps=16,
        tol_class_param="optim_param",
        seed=1,
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_nadam_parity(case: OptimTrajectoryCase) -> None:
    run_optim_trajectory_case(case)
