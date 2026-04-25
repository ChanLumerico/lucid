import numpy as np

import pytest

import torch

import lucid

from lucid.test.parity import data

from lucid.test.parity.core import ParityCase, TensorInput, run_parity_case


def _float(shape, *, seed: int, grad: bool = True):
    return lambda s: [
        TensorInput(data.random_floats(shape, seed=s), requires_grad=grad)
    ]


CASES: list[ParityCase] = []


def _masked_fill_inputs(shape, mask_shape, *, seed: int, grad: bool = True):
    def _build(s):
        return [
            TensorInput(data.random_floats(shape, seed=s), requires_grad=grad),
            TensorInput(data.bool_array(mask_shape, seed=s + 10), requires_grad=False),
        ]

    return _build


CASES.extend(
    [
        ParityCase(
            name="masked_fill_2d",
            build_inputs=_masked_fill_inputs((3, 4), (3, 4), seed=5000),
            lucid_fn=lambda a, m: lucid.masked_fill(a, m, value=-1.0),
            torch_fn=lambda a, m: a.masked_fill(m, -1.0),
            tol_class="elementwise_f64",
            seed=5000,
        ),
        ParityCase(
            name="masked_fill_broadcast_row",
            build_inputs=_masked_fill_inputs((3, 4), (1, 4), seed=5010),
            lucid_fn=lambda a, m: lucid.masked_fill(a, m, value=0.0),
            torch_fn=lambda a, m: a.masked_fill(m, 0.0),
            tol_class="elementwise_f64",
            seed=5010,
        ),
    ]
)


def _gather_inputs(shape, idx_shape, axis_size, *, seed: int, grad: bool = True):
    def _build(s):
        return [
            TensorInput(data.random_floats(shape, seed=s), requires_grad=grad),
            TensorInput(
                data.int_array(idx_shape, seed=s + 20, low=0, high=axis_size),
                requires_grad=False,
                dtype_override=lucid.Int64,
            ),
        ]

    return _build


CASES.extend(
    [
        ParityCase(
            name="gather_axis1",
            build_inputs=_gather_inputs((4, 5), (4, 3), axis_size=5, seed=5100),
            lucid_fn=lambda a, idx: lucid.gather(a, axis=1, index=idx),
            torch_fn=lambda a, idx: torch.gather(a, dim=1, index=idx),
            tol_class="elementwise_f64",
            gradcheck=True,
            seed=5100,
        ),
        ParityCase(
            name="gather_axis0",
            build_inputs=_gather_inputs((4, 5), (3, 5), axis_size=4, seed=5110),
            lucid_fn=lambda a, idx: lucid.gather(a, axis=0, index=idx),
            torch_fn=lambda a, idx: torch.gather(a, dim=0, index=idx),
            tol_class="elementwise_f64",
            seed=5110,
        ),
        ParityCase(
            name="gather_3d_axis_last",
            build_inputs=_gather_inputs((2, 3, 4), (2, 3, 5), axis_size=4, seed=5120),
            lucid_fn=lambda a, idx: lucid.gather(a, axis=-1, index=idx),
            torch_fn=lambda a, idx: torch.gather(a, dim=-1, index=idx),
            tol_class="elementwise_f64",
            seed=5120,
        ),
    ]
)


def _where_inputs(shape, *, seed: int, grad: bool = True):
    def _build(s):
        return [
            TensorInput(data.bool_array(shape, seed=s), requires_grad=False),
            TensorInput(data.random_floats(shape, seed=s + 1), requires_grad=grad),
            TensorInput(data.random_floats(shape, seed=s + 2), requires_grad=grad),
        ]

    return _build


CASES.append(
    ParityCase(
        name="where_2d",
        build_inputs=_where_inputs((3, 4), seed=5200),
        lucid_fn=lucid.where,
        torch_fn=torch.where,
        tol_class="elementwise_f64",
        seed=5200,
    )
)

CASES.extend(
    [
        ParityCase(
            name="sort_values",
            build_inputs=_float((3, 5), seed=5300, grad=False),
            lucid_fn=lambda a: lucid.sort(a, axis=-1)[0],
            torch_fn=lambda a: torch.sort(a, dim=-1).values,
            tol_class="elementwise_f64",
            check_backward=False,
            seed=5300,
        ),
        ParityCase(
            name="argsort",
            build_inputs=_float((3, 5), seed=5310, grad=False),
            lucid_fn=lambda a: lucid.argsort(a, axis=-1),
            torch_fn=lambda a: torch.argsort(a, dim=-1),
            tol_class="integer_exact",
            check_backward=False,
            seed=5310,
        ),
        ParityCase(
            name="argmin_axis",
            build_inputs=_float((3, 4), seed=5320, grad=False),
            lucid_fn=lambda a: lucid.argmin(a, axis=1),
            torch_fn=lambda a: torch.argmin(a, dim=1),
            tol_class="integer_exact",
            check_backward=False,
            seed=5320,
        ),
        ParityCase(
            name="argmax_axis_last",
            build_inputs=_float((3, 4), seed=5330, grad=False),
            lucid_fn=lambda a: lucid.argmax(a, axis=-1),
            torch_fn=lambda a: torch.argmax(a, dim=-1),
            tol_class="integer_exact",
            check_backward=False,
            seed=5330,
        ),
    ]
)

CASES.extend(
    [
        ParityCase(
            name="unbind_axis0_first",
            build_inputs=_float((3, 4), seed=5400),
            lucid_fn=lambda a: lucid.unbind(a, axis=0)[0],
            torch_fn=lambda a: torch.unbind(a, dim=0)[0],
            tol_class="elementwise_f64",
            seed=5400,
        ),
        ParityCase(
            name="unbind_axis1_last",
            build_inputs=_float((3, 4), seed=5410),
            lucid_fn=lambda a: lucid.unbind(a, axis=1)[-1],
            torch_fn=lambda a: torch.unbind(a, dim=1)[-1],
            tol_class="elementwise_f64",
            seed=5410,
        ),
    ]
)

CASES.extend(
    [
        ParityCase(
            name="getitem_int_row",
            build_inputs=_float((5, 4), seed=5500),
            lucid_fn=lambda a: a[2],
            torch_fn=lambda a: a[2],
            tol_class="elementwise_f64",
            seed=5500,
        ),
        ParityCase(
            name="getitem_slice",
            build_inputs=_float((5, 4), seed=5510),
            lucid_fn=lambda a: a[1:4, :2],
            torch_fn=lambda a: a[1:4, :2],
            tol_class="elementwise_f64",
            seed=5510,
        ),
        ParityCase(
            name="getitem_int_scalar",
            build_inputs=_float((5, 4), seed=5520),
            lucid_fn=lambda a: a[2, 3],
            torch_fn=lambda a: a[2, 3],
            tol_class="elementwise_f64",
            seed=5520,
        ),
        ParityCase(
            name="getitem_negative_index",
            build_inputs=_float((5, 4), seed=5530),
            lucid_fn=lambda a: a[-1],
            torch_fn=lambda a: a[-1],
            tol_class="elementwise_f64",
            seed=5530,
        ),
    ]
)

CASES.append(
    ParityCase(
        name="nonzero",
        build_inputs=lambda s: [
            TensorInput(
                data.random_floats((3, 4), seed=s, low=-1.0, high=1.0),
                requires_grad=False,
            )
        ],
        lucid_fn=lambda a: lucid.nonzero(a > 0),
        torch_fn=lambda a: torch.nonzero(a > 0),
        tol_class="integer_exact",
        check_backward=False,
        seed=5600,
    )
)

CASES.append(
    ParityCase(
        name="diagonal_2d",
        build_inputs=_float((5, 5), seed=5700),
        lucid_fn=lambda a: lucid.diagonal(a),
        torch_fn=lambda a: torch.diagonal(a),
        tol_class="elementwise_f64",
        seed=5700,
    )
)

CASES.append(
    ParityCase(
        name="topk_values",
        build_inputs=_float((3, 5), seed=5800, grad=False),
        lucid_fn=lambda a: lucid.topk(a, k=2, axis=-1)[0],
        torch_fn=lambda a: torch.topk(a, k=2, dim=-1).values,
        tol_class="elementwise_f64",
        check_backward=False,
        seed=5800,
    )
)

CASES.append(
    ParityCase(
        name="unique",
        build_inputs=lambda s: [
            TensorInput(
                data.int_array((6,), seed=s, low=0, high=4),
                requires_grad=False,
                dtype_override=lucid.Int64,
            )
        ],
        lucid_fn=lambda a: lucid.unique(a, sorted=True),
        torch_fn=lambda a: torch.unique(a, sorted=True),
        tol_class="integer_exact",
        check_backward=False,
        seed=5900,
    )
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_indexing_parity(case: ParityCase) -> None:
    run_parity_case(case)
