import pytest

import torch

import lucid

from lucid.test.parity import data

from lucid.test.parity.core import ParityCase, TensorInput, run_parity_case


def _float(shape, *, seed: int, grad: bool = True):
    return lambda s: [
        TensorInput(data.random_floats(shape, seed=s), requires_grad=grad)
    ]


def _pos(shape, *, seed: int, grad: bool = True):
    return lambda s: [TensorInput(data.pos_floats(shape, seed=s), requires_grad=grad)]


CASES: list[ParityCase] = []

SHAPE_3D = (2, 3, 4)

for label, axis, t_axis in [
    ("all", None, None),
    ("first", 0, 0),
    ("last", -1, -1),
    ("tuple", (0, 2), (0, 2)),
]:
    for keepdims in (False, True):
        kd_label = "keep" if keepdims else "reduce"
        CASES.append(
            ParityCase(
                name=f"sum_{label}_{kd_label}",
                build_inputs=_float(SHAPE_3D, seed=3000),
                lucid_fn=lambda a, ax=axis, kd=keepdims: (
                    a.sum(axis=ax, keepdims=kd) if ax is not None else a.sum()
                ),
                torch_fn=lambda a, ax=t_axis, kd=keepdims: (
                    a.sum(dim=ax, keepdim=kd) if ax is not None else a.sum()
                ),
                tol_class="reduction_f64",
                seed=3000,
            )
        )
for label, axis, t_axis in [
    ("all", None, None),
    ("first", 0, 0),
    ("last", -1, -1),
    ("tuple", (1, 2), (1, 2)),
]:
    for keepdims in (False, True):
        kd_label = "keep" if keepdims else "reduce"
        CASES.append(
            ParityCase(
                name=f"mean_{label}_{kd_label}",
                build_inputs=_float(SHAPE_3D, seed=3100),
                lucid_fn=lambda a, ax=axis, kd=keepdims: (
                    a.mean(axis=ax, keepdims=kd) if ax is not None else a.mean()
                ),
                torch_fn=lambda a, ax=t_axis, kd=keepdims: (
                    a.mean(dim=ax, keepdim=kd) if ax is not None else a.mean()
                ),
                tol_class="reduction_f64",
                seed=3100,
            )
        )
for label, axis, t_axis in [("first", 0, 0), ("last", -1, -1)]:
    for keepdims in (False, True):
        kd_label = "keep" if keepdims else "reduce"
        CASES.append(
            ParityCase(
                name=f"var_{label}_{kd_label}",
                build_inputs=_float(SHAPE_3D, seed=3200),
                lucid_fn=lambda a, ax=axis, kd=keepdims: a.var(axis=ax, keepdims=kd),
                torch_fn=lambda a, ax=t_axis, kd=keepdims: a.var(
                    dim=ax, keepdim=kd, unbiased=False
                ),
                tol_class="reduction_f64",
                seed=3200,
            )
        )
CASES.extend(
    [
        ParityCase(
            name="cumsum_first",
            build_inputs=_float((4, 3), seed=3300),
            lucid_fn=lambda a: lucid.cumsum(a, axis=0),
            torch_fn=lambda a: torch.cumsum(a, dim=0),
            tol_class="reduction_f64",
            seed=3300,
        ),
        ParityCase(
            name="cumsum_last_3d",
            build_inputs=_float((2, 3, 4), seed=3310),
            lucid_fn=lambda a: lucid.cumsum(a, axis=-1),
            torch_fn=lambda a: torch.cumsum(a, dim=-1),
            tol_class="reduction_f64",
            seed=3310,
        ),
        ParityCase(
            name="cumprod_last",
            build_inputs=_pos((2, 3, 4), seed=3320),
            lucid_fn=lambda a: lucid.cumprod(a, axis=-1),
            torch_fn=lambda a: torch.cumprod(a, dim=-1),
            tol_class="reduction_f64",
            seed=3320,
        ),
    ]
)

CASES.extend(
    [
        ParityCase(
            name="sum_1d",
            build_inputs=_float((7,), seed=3400),
            lucid_fn=lambda a: a.sum(),
            torch_fn=lambda a: a.sum(),
            tol_class="reduction_f64",
            seed=3400,
        ),
        ParityCase(
            name="mean_1d",
            build_inputs=_float((7,), seed=3410),
            lucid_fn=lambda a: a.mean(),
            torch_fn=lambda a: a.mean(),
            tol_class="reduction_f64",
            seed=3410,
        ),
    ]
)

CASES.append(
    ParityCase(
        name="norm_l2_axis_last",
        build_inputs=_float((3, 4), seed=3500),
        lucid_fn=lambda a: lucid.linalg.norm(a, ord=2, axis=-1),
        torch_fn=lambda a: torch.linalg.norm(a, ord=2, dim=-1),
        tol_class="reduction_f64",
        seed=3500,
    )
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_reductions_parity(case: ParityCase) -> None:
    run_parity_case(case)
