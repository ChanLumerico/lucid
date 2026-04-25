import numpy as np

import pytest

import torch

import lucid

from lucid.test.parity import data

from lucid.test.parity.core import ParityCase, ScalarInput, TensorInput, run_parity_case


def _float(
    shape, *, seed: int, grad: bool = True, low: float = -2.0, high: float = 2.0
):
    return lambda s: [
        TensorInput(
            data.random_floats(shape, seed=s, low=low, high=high), requires_grad=grad
        )
    ]


def _pos(shape, *, seed: int, grad: bool = True, low: float = 0.25, high: float = 2.0):
    return lambda s: [
        TensorInput(
            data.pos_floats(shape, seed=s, low=low, high=high), requires_grad=grad
        )
    ]


def _nonzero(shape, *, seed: int, grad: bool = True):
    return lambda s: [
        TensorInput(data.nonzero_floats(shape, seed=s), requires_grad=grad)
    ]


def _unit_bounded(shape, *, seed: int, grad: bool = True, margin: float = 0.1):
    return lambda s: [
        TensorInput(data.unit_bounded(shape, seed=s, margin=margin), requires_grad=grad)
    ]


CASES: list[ParityCase] = []

SIMPLE = [
    ("neg", lambda a: -a, lambda a: -a, _float((3, 4), seed=1000), "elementwise_f64"),
    (
        "exp",
        lucid.exp,
        torch.exp,
        _float((3, 4), seed=1010, low=-1.5, high=1.5),
        "elementwise_f64",
    ),
    ("log", lucid.log, torch.log, _pos((3, 4), seed=1020), "elementwise_f64"),
    (
        "sqrt",
        lucid.sqrt,
        torch.sqrt,
        _pos((3, 4), seed=1030, low=0.5, high=2.0),
        "elementwise_f64",
    ),
    ("sin", lucid.sin, torch.sin, _float((3, 4), seed=1040), "elementwise_f64"),
    ("cos", lucid.cos, torch.cos, _float((3, 4), seed=1050), "elementwise_f64"),
    (
        "tan",
        lucid.tan,
        torch.tan,
        _float((3, 4), seed=1060, low=-1.2, high=1.2),
        "elementwise_f64",
    ),
    ("sinh", lucid.sinh, torch.sinh, _float((3, 4), seed=1080), "elementwise_f64"),
    ("cosh", lucid.cosh, torch.cosh, _float((3, 4), seed=1090), "elementwise_f64"),
    ("tanh", lucid.tanh, torch.tanh, _float((3, 4), seed=1100), "elementwise_f64"),
    (
        "arcsin",
        lucid.arcsin,
        torch.arcsin,
        _unit_bounded((3, 4), seed=1110),
        "elementwise_f64",
    ),
    (
        "arccos",
        lucid.arccos,
        torch.arccos,
        _unit_bounded((3, 4), seed=1120),
        "elementwise_f64",
    ),
    (
        "arctan",
        lucid.arctan,
        torch.arctan,
        _float((3, 4), seed=1130),
        "elementwise_f64",
    ),
    (
        "reciprocal",
        lucid.reciprocal,
        torch.reciprocal,
        _nonzero((3, 4), seed=1140),
        "elementwise_f64",
    ),
    (
        "square",
        lucid.square,
        torch.square,
        _float((3, 4), seed=1150),
        "elementwise_f64",
    ),
    ("cube", lucid.cube, lambda a: a**3, _float((3, 4), seed=1160), "elementwise_f64"),
    ("abs", lucid.abs, torch.abs, _nonzero((3, 4), seed=1170), "elementwise_f64"),
]

for name, lfn, tfn, build, tol in SIMPLE:
    CASES.append(
        ParityCase(
            name=name,
            build_inputs=build,
            lucid_fn=lfn,
            torch_fn=tfn,
            tol_class=tol,
            seed=1000,
        )
    )
CASES.append(
    ParityCase(
        name="sign",
        build_inputs=_nonzero((3, 4), seed=1180, grad=False),
        lucid_fn=lucid.sign,
        torch_fn=torch.sign,
        tol_class="elementwise_f64",
        check_backward=False,
        seed=1180,
    )
)

CASES.append(
    ParityCase(
        name="clip_symmetric",
        build_inputs=_float((3, 4), seed=1200, low=-3.0, high=3.0),
        lucid_fn=lambda a: lucid.clip(a, min_value=-1.0, max_value=1.0),
        torch_fn=lambda a: torch.clamp(a, min=-1.0, max=1.0),
        tol_class="elementwise_f64",
        seed=1200,
    )
)

for name, lfn, tfn in [
    ("round", lucid.round, torch.round),
    ("floor", lucid.floor, torch.floor),
    ("ceil", lucid.ceil, torch.ceil),
]:
    CASES.append(
        ParityCase(
            name=name,
            build_inputs=_float((3, 4), seed=1300, grad=False),
            lucid_fn=lfn,
            torch_fn=tfn,
            tol_class="elementwise_f64",
            check_backward=False,
            seed=1300,
        )
    )
CASES.extend(
    [
        ParityCase(
            name="pow_scalar_int_exp",
            build_inputs=_float((3, 4), seed=1400),
            lucid_fn=lambda a: a**3,
            torch_fn=lambda a: a**3,
            tol_class="elementwise_f64",
            seed=1400,
        ),
        ParityCase(
            name="pow_scalar_frac_exp",
            build_inputs=_pos((3, 4), seed=1410),
            lucid_fn=lambda a: a**0.5,
            torch_fn=lambda a: a**0.5,
            tol_class="elementwise_f64",
            seed=1410,
        ),
        ParityCase(
            name="rpow_scalar_base",
            build_inputs=_float((3, 4), seed=1420, low=-0.5, high=1.0),
            lucid_fn=lambda a: 2.0**a,
            torch_fn=lambda a: 2.0**a,
            tol_class="elementwise_f64",
            seed=1420,
        ),
    ]
)

CASES.extend(
    [
        ParityCase(
            name="T_2d",
            build_inputs=_float((3, 4), seed=1500),
            lucid_fn=lambda a: a.T,
            torch_fn=lambda a: a.T,
            tol_class="elementwise_f64",
            seed=1500,
        ),
        ParityCase(
            name="mT_batched",
            build_inputs=_float((2, 3, 4), seed=1510),
            lucid_fn=lambda a: a.mT,
            torch_fn=lambda a: a.mT,
            tol_class="elementwise_f64",
            seed=1510,
        ),
        ParityCase(
            name="transpose_3d",
            build_inputs=_float((2, 3, 4), seed=1520),
            lucid_fn=lambda a: a.transpose((1, 0, 2)),
            torch_fn=lambda a: a.permute(1, 0, 2),
            tol_class="elementwise_f64",
            seed=1520,
        ),
        ParityCase(
            name="swapaxes_3d",
            build_inputs=_float((2, 3, 4), seed=1530),
            lucid_fn=lambda a: lucid.swapaxes(a, 0, 2),
            torch_fn=lambda a: torch.swapaxes(a, 0, 2),
            tol_class="elementwise_f64",
            seed=1530,
        ),
    ]
)

CASES.extend(
    [
        ParityCase(
            name="sum_all",
            build_inputs=_float((3, 4), seed=1600),
            lucid_fn=lambda a: a.sum(),
            torch_fn=lambda a: a.sum(),
            tol_class="reduction_f64",
            seed=1600,
        ),
        ParityCase(
            name="sum_axis0",
            build_inputs=_float((3, 4), seed=1610),
            lucid_fn=lambda a: a.sum(axis=0),
            torch_fn=lambda a: a.sum(dim=0),
            tol_class="reduction_f64",
            seed=1610,
        ),
        ParityCase(
            name="mean_all",
            build_inputs=_float((3, 4), seed=1620),
            lucid_fn=lambda a: a.mean(),
            torch_fn=lambda a: a.mean(),
            tol_class="reduction_f64",
            seed=1620,
        ),
        ParityCase(
            name="mean_axis_keepdims",
            build_inputs=_float((3, 4), seed=1630),
            lucid_fn=lambda a: a.mean(axis=1, keepdims=True),
            torch_fn=lambda a: a.mean(dim=1, keepdim=True),
            tol_class="reduction_f64",
            seed=1630,
        ),
        ParityCase(
            name="var_all",
            build_inputs=_float((3, 4), seed=1640),
            lucid_fn=lambda a: a.var(),
            torch_fn=lambda a: a.var(unbiased=False),
            tol_class="reduction_f64",
            seed=1640,
        ),
        ParityCase(
            name="trace_2d",
            build_inputs=_float((5, 5), seed=1650),
            lucid_fn=lucid.trace,
            torch_fn=torch.trace,
            tol_class="reduction_f64",
            seed=1650,
        ),
        ParityCase(
            name="trace_highdim",
            build_inputs=_float((3, 3, 4), seed=1655),
            lucid_fn=lucid.trace,
            torch_fn=lambda a: torch.diagonal(a, dim1=0, dim2=1).sum(dim=-1),
            tol_class="reduction_f64",
            seed=1655,
        ),
        ParityCase(
            name="min_all",
            build_inputs=_float((3, 4), seed=1660),
            lucid_fn=lambda a: lucid.min(a),
            torch_fn=lambda a: a.min(),
            tol_class="reduction_f64",
            seed=1660,
        ),
        ParityCase(
            name="max_axis",
            build_inputs=_float((3, 4), seed=1670),
            lucid_fn=lambda a: lucid.max(a, axis=1),
            torch_fn=lambda a: a.max(dim=1).values,
            tol_class="reduction_f64",
            seed=1670,
        ),
        ParityCase(
            name="cumsum_axis",
            build_inputs=_float((3, 4), seed=1680),
            lucid_fn=lambda a: lucid.cumsum(a, axis=1),
            torch_fn=lambda a: torch.cumsum(a, dim=1),
            tol_class="reduction_f64",
            seed=1680,
        ),
        ParityCase(
            name="cumprod_axis",
            build_inputs=_pos((3, 4), seed=1690),
            lucid_fn=lambda a: lucid.cumprod(a, axis=1),
            torch_fn=lambda a: torch.cumprod(a, dim=1),
            tol_class="reduction_f64",
            seed=1690,
        ),
    ]
)

CASES.extend(
    [
        ParityCase(
            name="exp_0d",
            build_inputs=_float((), seed=1800, low=-1.0, high=1.0),
            lucid_fn=lucid.exp,
            torch_fn=torch.exp,
            tol_class="elementwise_f64",
            seed=1800,
        ),
        ParityCase(
            name="sum_0d",
            build_inputs=_float((), seed=1810),
            lucid_fn=lambda a: a.sum(),
            torch_fn=lambda a: a.sum(),
            tol_class="elementwise_f64",
            seed=1810,
        ),
    ]
)

CASES.extend(
    [
        ParityCase(
            name=f"{fname}_batched",
            build_inputs=_pos((2, 3, 4), seed=1900 + off),
            lucid_fn=lfn,
            torch_fn=tfn,
            tol_class="elementwise_f64",
            seed=1900 + off,
        )
        for (off, (fname, lfn, tfn)) in enumerate(
            [
                ("exp", lucid.exp, torch.exp),
                ("log", lucid.log, torch.log),
                ("sqrt", lucid.sqrt, torch.sqrt),
                ("square", lucid.square, torch.square),
                ("tanh", lucid.tanh, torch.tanh),
            ]
        )
    ]
)

CASES.extend(
    [
        ParityCase(
            name="sum_axis_neg1",
            build_inputs=_float((2, 3, 4), seed=2000),
            lucid_fn=lambda a: a.sum(axis=-1),
            torch_fn=lambda a: a.sum(dim=-1),
            tol_class="reduction_f64",
            seed=2000,
        ),
        ParityCase(
            name="sum_axis_tuple",
            build_inputs=_float((2, 3, 4), seed=2010),
            lucid_fn=lambda a: a.sum(axis=(0, 2)),
            torch_fn=lambda a: a.sum(dim=(0, 2)),
            tol_class="reduction_f64",
            seed=2010,
        ),
        ParityCase(
            name="mean_axis_neg1_keepdims",
            build_inputs=_float((2, 3, 4), seed=2020),
            lucid_fn=lambda a: a.mean(axis=-1, keepdims=True),
            torch_fn=lambda a: a.mean(dim=-1, keepdim=True),
            tol_class="reduction_f64",
            seed=2020,
        ),
        ParityCase(
            name="var_axis_keepdims",
            build_inputs=_float((2, 3, 4), seed=2030),
            lucid_fn=lambda a: a.var(axis=1, keepdims=True),
            torch_fn=lambda a: a.var(dim=1, keepdim=True, unbiased=False),
            tol_class="reduction_f64",
            seed=2030,
        ),
        ParityCase(
            name="cumsum_axis0",
            build_inputs=_float((4, 3), seed=2040),
            lucid_fn=lambda a: lucid.cumsum(a, axis=0),
            torch_fn=lambda a: torch.cumsum(a, dim=0),
            tol_class="reduction_f64",
            seed=2040,
        ),
        ParityCase(
            name="cumsum_0d_axis",
            build_inputs=_float((5,), seed=2050),
            lucid_fn=lambda a: lucid.cumsum(a, axis=0),
            torch_fn=lambda a: torch.cumsum(a, dim=0),
            tol_class="reduction_f64",
            seed=2050,
        ),
        ParityCase(
            name="min_axis",
            build_inputs=_float((3, 4), seed=2060),
            lucid_fn=lambda a: lucid.min(a, axis=1),
            torch_fn=lambda a: a.min(dim=1).values,
            tol_class="reduction_f64",
            seed=2060,
        ),
        ParityCase(
            name="max_all",
            build_inputs=_float((3, 4), seed=2070),
            lucid_fn=lambda a: lucid.max(a),
            torch_fn=lambda a: a.max(),
            tol_class="reduction_f64",
            seed=2070,
        ),
    ]
)

CASES.extend(
    [
        ParityCase(
            name="neg_3d",
            build_inputs=_float((2, 3, 4), seed=2100),
            lucid_fn=lambda a: -a,
            torch_fn=lambda a: -a,
            tol_class="elementwise_f64",
            seed=2100,
        ),
        ParityCase(
            name="abs_3d",
            build_inputs=_nonzero((2, 3, 4), seed=2110),
            lucid_fn=lucid.abs,
            torch_fn=torch.abs,
            tol_class="elementwise_f64",
            seed=2110,
        ),
        ParityCase(
            name="clip_positive_only",
            build_inputs=_float((3, 4), seed=2120, low=-3.0, high=3.0),
            lucid_fn=lambda a: lucid.clip(a, min_value=0.0),
            torch_fn=lambda a: torch.clamp(a, min=0.0),
            tol_class="elementwise_f64",
            seed=2120,
        ),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_ufunc_parity(case: ParityCase) -> None:
    run_parity_case(case)
