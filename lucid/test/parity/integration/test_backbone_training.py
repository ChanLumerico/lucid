from typing import Any

import numpy as np
import pytest
import torch

import lucid
import lucid.nn as lnn
import lucid.optim as loptim
import lucid.optim.lr_scheduler as lsched

from lucid.test.parity.core import (
    OptimTrajectoryCase,
    run_optim_trajectory_case,
    SchedulerTrajectoryCase,
    run_scheduler_case,
)

_NUM_CLASSES = 10
_BATCH = 8
_LR = 5e-3
_STEPS = 64
_SCHED_STEP = 2
_SCHED_GAMMA = 0.5
_SEED = 20260305


class _LucidBackbone(lnn.Module):
    def __init__(self, num_classes: int = _NUM_CLASSES) -> None:
        super().__init__()
        self.identity = lnn.Identity()
        self.stem = lnn.Sequential(
            lnn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=True),
            lnn.BatchNorm2d(
                num_features=8,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=False,
            ),
            lnn.ReLU(),
        )
        self.blocks = lnn.ModuleList(
            [
                lnn.Sequential(
                    lnn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=True),
                    lnn.GroupNorm(num_groups=2, num_channels=8, eps=1e-5, affine=True),
                    lnn.ReLU(),
                    lnn.MaxPool2d(kernel_size=2, stride=2),
                ),
                lnn.Sequential(
                    lnn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=True),
                    lnn.GELU(),
                ),
            ]
        )
        self.layer_norm = lnn.LayerNorm((8, 16, 16), eps=1e-5)
        self.dropout = lnn.Dropout2d(p=0.0)
        self.pool = lnn.AdaptiveAvgPool2d(output_size=(4, 4))
        self.flatten = lnn.Flatten(start_axis=1, end_axis=-1)
        self.classifier = lnn.Sequential(
            lnn.Linear(128, 64, bias=True),
            lnn.LayerNorm(64, eps=1e-5),
            lnn.Dropout(p=0.0),
            lnn.Linear(64, num_classes, bias=True),
        )

    def forward(self, x: Any) -> Any:
        x = self.identity(x)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class _TorchBackbone(torch.nn.Module):
    def __init__(self, num_classes: int = _NUM_CLASSES) -> None:
        super().__init__()
        self.identity = torch.nn.Identity()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(
                num_features=8,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=False,
            ),
            torch.nn.ReLU(),
        )
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        8, 8, kernel_size=3, stride=1, padding=1, bias=True
                    ),
                    torch.nn.GroupNorm(num_groups=2, num_channels=8, eps=1e-5),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2),
                ),
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        8, 8, kernel_size=3, stride=1, padding=1, bias=True
                    ),
                    torch.nn.GELU(approximate="tanh"),
                ),
            ]
        )
        self.layer_norm = torch.nn.LayerNorm((8, 16, 16), eps=1e-5)
        self.dropout = torch.nn.Dropout2d(p=0.0)
        self.pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128, 64, bias=True),
            torch.nn.LayerNorm(64, eps=1e-5),
            torch.nn.Dropout(p=0.0),
            torch.nn.Linear(64, num_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.identity(x)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def _build_models(seed: int) -> tuple[lnn.Module, torch.nn.Module]:
    lucid.random.seed(seed)
    torch.manual_seed(seed + 101)
    lm = _LucidBackbone()
    tm = _TorchBackbone().double()
    lm.train(True)
    tm.train(True)
    return lm, tm


def _build_step_inputs(step: int, seed: int):
    rng = np.random.RandomState(seed + 101 + step)
    x = rng.standard_normal((_BATCH, 3, 32, 32)).astype(np.float64)
    y = rng.standard_normal((_BATCH, _NUM_CLASSES)).astype(np.float64)
    return (
        lucid.tensor(x.copy()),
        lucid.tensor(y.copy()),
        torch.as_tensor(x.copy(), dtype=torch.float64),
        torch.as_tensor(y.copy(), dtype=torch.float64),
    )


def _mse(out, y):
    return ((out - y) ** 2).mean()


def test_backbone_full_training_trajectory():
    case = OptimTrajectoryCase(
        name="backbone_adam_steplr",
        build_models=_build_models,
        build_step_inputs=_build_step_inputs,
        lucid_optim=lambda p: loptim.Adam(p, lr=_LR),
        torch_optim=lambda p: torch.optim.Adam(p, lr=_LR),
        lucid_loss=_mse,
        torch_loss=_mse,
        steps=_STEPS,
        tol_class_param="optim_param",
        seed=_SEED,
    )
    run_optim_trajectory_case(case)


def test_backbone_steplr_per_step_lr_match():
    def _build(initial_lr: float):
        lp = lnn.Parameter(lucid.zeros(1))
        tp = torch.nn.Parameter(torch.zeros(1))
        lopt = loptim.Adam([lp], lr=initial_lr)
        topt = torch.optim.Adam([tp], lr=initial_lr)
        for g in topt.param_groups:
            g.setdefault("initial_lr", g["lr"])
        return (
            lopt,
            lsched.StepLR(lopt, step_size=_SCHED_STEP, gamma=_SCHED_GAMMA),
            topt,
            torch.optim.lr_scheduler.StepLR(
                topt, step_size=_SCHED_STEP, gamma=_SCHED_GAMMA
            ),
        )

    case = SchedulerTrajectoryCase(
        name="backbone_steplr_lr_per_step",
        build_schedulers=_build,
        initial_lr=_LR,
        steps=_STEPS,
    )
    run_scheduler_case(case)
