import numpy as np

import pytest

import torch

import torch.nn.functional as TF

import lucid

import lucid.nn.functional as F

from lucid.test.parity import data

from lucid.test.parity.core import ParityCase, TensorInput, run_parity_case


def _pair(shape, *, seed: int, grad: bool = True):
    return lambda s: [
        TensorInput(data.random_floats(shape, seed=s), requires_grad=grad),
        TensorInput(data.random_floats(shape, seed=s + 1), requires_grad=False),
    ]


def _logits_target(shape, num_classes: int, *, seed: int):
    def _build(s):
        logits = data.logits(
            (shape, num_classes) if isinstance(shape, int) else (*shape, num_classes),
            seed=s,
        )
        tgt = data.cat_indices(
            shape if isinstance(shape, int) else int(np.prod(shape)),
            num_classes,
            seed=s + 1,
        )
        return [
            TensorInput(logits, requires_grad=True),
            TensorInput(tgt, requires_grad=False, dtype_override=lucid.Int64),
        ]

    return _build


def _binary_input_target(shape, *, seed: int, logit_range: bool = False):
    def _build(s):
        probs = (
            data.random_floats(shape, seed=s, low=0.1, high=0.9)
            if not logit_range
            else data.logits(shape, seed=s)
        )
        target = (data.random_floats(shape, seed=s + 1) > 0).astype(np.float64)
        return [
            TensorInput(probs, requires_grad=True),
            TensorInput(target, requires_grad=False),
        ]

    return _build


CASES: list[ParityCase] = []

for red in ("mean", "sum", None):
    red_label = red if red is not None else "none"
    CASES.extend(
        [
            ParityCase(
                name=f"mse_{red_label}",
                build_inputs=_pair((3, 4), seed=3000),
                lucid_fn=lambda p, t, r=red: F.mse_loss(p, t, reduction=r),
                torch_fn=lambda p, t, r=red: TF.mse_loss(p, t, reduction=r or "none"),
                tol_class="loss_f32",
                seed=3000,
            ),
            ParityCase(
                name=f"huber_{red_label}",
                build_inputs=_pair((3, 4), seed=3010),
                lucid_fn=lambda p, t, r=red: F.huber_loss(p, t, delta=1.0, reduction=r),
                torch_fn=lambda p, t, r=red: TF.huber_loss(
                    p, t, reduction=r or "none", delta=1.0
                ),
                tol_class="loss_f32",
                seed=3010,
            ),
        ]
    )
CASES.extend(
    [
        ParityCase(
            name="bce_with_logits_mean",
            build_inputs=_binary_input_target((3, 4), seed=3200, logit_range=True),
            lucid_fn=lambda p, t: F.binary_cross_entropy_with_logits(
                p, t, reduction="mean"
            ),
            torch_fn=lambda p, t: TF.binary_cross_entropy_with_logits(
                p, t, reduction="mean"
            ),
            tol_class="loss_f32",
            seed=3200,
        ),
        ParityCase(
            name="bce_with_logits_sum",
            build_inputs=_binary_input_target((3, 4), seed=3210, logit_range=True),
            lucid_fn=lambda p, t: F.binary_cross_entropy_with_logits(
                p, t, reduction="sum"
            ),
            torch_fn=lambda p, t: TF.binary_cross_entropy_with_logits(
                p, t, reduction="sum"
            ),
            tol_class="loss_f32",
            seed=3210,
        ),
    ]
)

CASES.append(
    ParityCase(
        name="bce_mean",
        build_inputs=_binary_input_target((3, 4), seed=3300),
        lucid_fn=lambda p, t: F.binary_cross_entropy(p, t, reduction="mean"),
        torch_fn=lambda p, t: TF.binary_cross_entropy(p, t, reduction="mean"),
        tol_class="loss_f32",
        seed=3300,
    )
)


def _ce_inputs(batch: int, num_classes: int, *, seed: int):
    def _build(s):
        logits = data.logits((batch, num_classes), seed=s)
        target = data.cat_indices(batch, num_classes, seed=s + 1)
        return [
            TensorInput(logits, requires_grad=True),
            TensorInput(target, requires_grad=False, dtype_override=lucid.Int64),
        ]

    return _build


CASES.extend(
    [
        ParityCase(
            name="cross_entropy_mean",
            build_inputs=_ce_inputs(8, 5, seed=3400),
            lucid_fn=lambda p, t: F.cross_entropy(p, t, reduction="mean"),
            torch_fn=lambda p, t: TF.cross_entropy(p, t, reduction="mean"),
            tol_class="loss_f32",
            seed=3400,
        ),
        ParityCase(
            name="cross_entropy_sum",
            build_inputs=_ce_inputs(8, 5, seed=3410),
            lucid_fn=lambda p, t: F.cross_entropy(p, t, reduction="sum"),
            torch_fn=lambda p, t: TF.cross_entropy(p, t, reduction="sum"),
            tol_class="loss_f32",
            seed=3410,
        ),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_loss_parity(case: ParityCase) -> None:
    run_parity_case(case)
