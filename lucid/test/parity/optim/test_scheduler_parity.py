import numpy as np

import pytest

import torch

import lucid

import lucid.nn as lnn

import lucid.optim as loptim

import lucid.optim.lr_scheduler as lsched

from lucid.test.parity.core import SchedulerTrajectoryCase, run_scheduler_case


def _stub_lucid_optim(lr: float):
    p = lnn.Parameter(lucid.zeros(1))
    return loptim.SGD([p], lr=lr)


def _stub_torch_optim(lr: float):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=lr)
    for g in opt.param_groups:
        g.setdefault("initial_lr", g["lr"])
    return opt


def _make(
    name,
    lucid_sched,
    torch_sched,
    *,
    steps=20,
    lr=0.01,
    per_step_inputs=lambda s: None,
    xfail=None,
):
    def _build(initial_lr):
        lopt = _stub_lucid_optim(initial_lr)
        topt = _stub_torch_optim(initial_lr)
        return (lopt, lucid_sched(lopt), topt, torch_sched(topt))

    return SchedulerTrajectoryCase(
        name=name,
        build_schedulers=_build,
        initial_lr=lr,
        steps=steps,
        per_step_inputs=per_step_inputs,
        xfail=xfail,
    )


CASES: list[SchedulerTrajectoryCase] = [
    _make(
        "StepLR_step5_gamma05",
        lambda o: lsched.StepLR(o, step_size=5, gamma=0.5),
        lambda o: torch.optim.lr_scheduler.StepLR(o, step_size=5, gamma=0.5),
    ),
    _make(
        "StepLR_step3_gamma01",
        lambda o: lsched.StepLR(o, step_size=3, gamma=0.1),
        lambda o: torch.optim.lr_scheduler.StepLR(o, step_size=3, gamma=0.1),
        steps=10,
    ),
    _make(
        "MultiStepLR",
        lambda o: lsched.MultiStepLR(o, milestones=[3, 7, 12], gamma=0.5),
        lambda o: torch.optim.lr_scheduler.MultiStepLR(
            o, milestones=[3, 7, 12], gamma=0.5
        ),
    ),
    _make(
        "ExponentialLR",
        lambda o: lsched.ExponentialLR(o, gamma=0.9),
        lambda o: torch.optim.lr_scheduler.ExponentialLR(o, gamma=0.9),
    ),
    _make(
        "CosineAnnealingLR",
        lambda o: lsched.CosineAnnealingLR(o, T_max=10, eta_min=0.0001),
        lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(
            o, T_max=10, eta_min=0.0001
        ),
    ),
    _make(
        "LambdaLR_constant_decay",
        lambda o: lsched.LambdaLR(lambda epoch: 0.95**epoch, o),
        lambda o: torch.optim.lr_scheduler.LambdaLR(o, lambda epoch: 0.95**epoch),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_scheduler_parity(case: SchedulerTrajectoryCase) -> None:
    run_scheduler_case(case)
