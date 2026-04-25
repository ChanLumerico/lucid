import numpy as np

import pytest

import torch

import lucid

from lucid.test.parity import data

from lucid.test.parity.core import ParityCase, ScalarInput, TensorInput, run_parity_case


def _float_pair(shape_a, shape_b, *, seed: int, grad: tuple[bool, bool] = (True, True)):
    return lambda s: [
        TensorInput(data.random_floats(shape_a, seed=s), requires_grad=grad[0]),
        TensorInput(data.random_floats(shape_b, seed=s + 1), requires_grad=grad[1]),
    ]


def _pos_pair(shape_a, shape_b, *, seed: int, grad: tuple[bool, bool] = (True, True)):
    return lambda s: [
        TensorInput(data.pos_floats(shape_a, seed=s), requires_grad=grad[0]),
        TensorInput(data.pos_floats(shape_b, seed=s + 1), requires_grad=grad[1]),
    ]


def _int_pair(shape_a, shape_b, *, seed: int, low: int = 0, high: int = 16):
    return lambda s: [
        TensorInput(
            data.int_array(shape_a, seed=s, low=low, high=high),
            requires_grad=False,
            dtype_override=lucid.Int64,
        ),
        TensorInput(
            data.int_array(shape_b, seed=s + 1, low=low, high=high),
            requires_grad=False,
            dtype_override=lucid.Int64,
        ),
    ]


CASES: list[ParityCase] = []

for fn_name, lop, top in [
    ("add", lambda a, b: a + b, lambda a, b: a + b),
    ("sub", lambda a, b: a - b, lambda a, b: a - b),
    ("mul", lambda a, b: a * b, lambda a, b: a * b),
    ("div", lambda a, b: a / b, lambda a, b: a / b),
]:
    CASES.extend(
        [
            ParityCase(
                name=f"{fn_name}_2d",
                build_inputs=_float_pair((3, 4), (3, 4), seed=100),
                lucid_fn=lop,
                torch_fn=top,
                tol_class="elementwise_f64",
                seed=100,
            ),
            ParityCase(
                name=f"{fn_name}_broadcast_row",
                build_inputs=_float_pair((3, 4), (1, 4), seed=110),
                lucid_fn=lop,
                torch_fn=top,
                tol_class="elementwise_f64",
                seed=110,
            ),
            ParityCase(
                name=f"{fn_name}_broadcast_col",
                build_inputs=_float_pair((3, 4), (3, 1), seed=120),
                lucid_fn=lop,
                torch_fn=top,
                tol_class="elementwise_f64",
                seed=120,
            ),
            ParityCase(
                name=f"{fn_name}_scalar_rhs",
                build_inputs=lambda s: [
                    TensorInput(data.random_floats((3, 4), seed=s), requires_grad=True),
                    ScalarInput(1.7),
                ],
                lucid_fn=lop,
                torch_fn=top,
                tol_class="elementwise_f64",
                seed=130,
            ),
            ParityCase(
                name=f"{fn_name}_0d",
                build_inputs=_float_pair((), (), seed=140),
                lucid_fn=lop,
                torch_fn=top,
                tol_class="elementwise_f64",
                seed=140,
            ),
        ]
    )
CASES.extend(
    [
        ParityCase(
            name="div_safe_denom",
            build_inputs=lambda s: [
                TensorInput(data.random_floats((3, 4), seed=s), requires_grad=True),
                TensorInput(data.pos_floats((3, 4), seed=s + 1), requires_grad=True),
            ],
            lucid_fn=lambda a, b: a / b,
            torch_fn=lambda a, b: a / b,
            tol_class="elementwise_f64",
            seed=150,
        )
    ]
)

CASES.append(
    ParityCase(
        name="floordiv_2d",
        build_inputs=lambda s: [
            TensorInput(
                data.random_floats((3, 4), seed=s, low=1.0, high=5.0),
                requires_grad=False,
            ),
            TensorInput(data.pos_floats((3, 4), seed=s + 1), requires_grad=False),
        ],
        lucid_fn=lambda a, b: a // b,
        torch_fn=lambda a, b: torch.floor_divide(a, b),
        tol_class="elementwise_f64",
        check_backward=False,
        seed=160,
    )
)

for fn_name, lop, top in [
    ("eq", lambda a, b: a == b, lambda a, b: a == b),
    ("ne", lambda a, b: a != b, lambda a, b: a != b),
    ("gt", lambda a, b: a > b, lambda a, b: a > b),
    ("ge", lambda a, b: a >= b, lambda a, b: a >= b),
    ("lt", lambda a, b: a < b, lambda a, b: a < b),
    ("le", lambda a, b: a <= b, lambda a, b: a <= b),
]:
    CASES.append(
        ParityCase(
            name=f"{fn_name}_2d",
            build_inputs=lambda s, sa=(3, 4), sb=(3, 4): [
                TensorInput(data.random_floats(sa, seed=s), requires_grad=False),
                TensorInput(data.random_floats(sb, seed=s + 1), requires_grad=False),
            ],
            lucid_fn=lop,
            torch_fn=top,
            tol_class="boolean_exact",
            check_backward=False,
            seed=200,
        )
    )
for fn_name, lop, top in [
    ("minimum", lucid.minimum, torch.minimum),
    ("maximum", lucid.maximum, torch.maximum),
]:
    CASES.extend(
        [
            ParityCase(
                name=f"{fn_name}_2d",
                build_inputs=_float_pair((3, 4), (3, 4), seed=300),
                lucid_fn=lop,
                torch_fn=top,
                tol_class="elementwise_f64",
                seed=300,
            ),
            ParityCase(
                name=f"{fn_name}_broadcast",
                build_inputs=_float_pair((3, 4), (1, 4), seed=310),
                lucid_fn=lop,
                torch_fn=top,
                tol_class="elementwise_f64",
                seed=310,
            ),
        ]
    )
CASES.extend(
    [
        ParityCase(
            name="power_2d_pos_base",
            build_inputs=lambda s: [
                TensorInput(data.pos_floats((3, 4), seed=s), requires_grad=True),
                TensorInput(
                    data.random_floats((3, 4), seed=s + 1, low=-1.5, high=1.5),
                    requires_grad=True,
                ),
            ],
            lucid_fn=lucid.power,
            torch_fn=torch.pow,
            tol_class="elementwise_f64",
            seed=400,
        ),
        ParityCase(
            name="power_scalar_exponent",
            build_inputs=lambda s: [
                TensorInput(data.pos_floats((3, 4), seed=s), requires_grad=True),
                ScalarInput(2.3),
            ],
            lucid_fn=lucid.power,
            torch_fn=torch.pow,
            tol_class="elementwise_f64",
            seed=410,
        ),
    ]
)

CASES.extend(
    [
        ParityCase(
            name="matmul_2d_2d",
            build_inputs=_float_pair((3, 5), (5, 4), seed=500),
            lucid_fn=lambda a, b: a @ b,
            torch_fn=lambda a, b: a @ b,
            tol_class="matmul_f64",
            seed=500,
        ),
        ParityCase(
            name="matmul_batched",
            build_inputs=_float_pair((2, 3, 5), (2, 5, 4), seed=510),
            lucid_fn=lambda a, b: a @ b,
            torch_fn=lambda a, b: a @ b,
            tol_class="matmul_f64",
            seed=510,
        ),
        ParityCase(
            name="matmul_1d_2d",
            build_inputs=_float_pair((5,), (5, 4), seed=520),
            lucid_fn=lambda a, b: a @ b,
            torch_fn=lambda a, b: a @ b,
            tol_class="matmul_f64",
            seed=520,
        ),
        ParityCase(
            name="matmul_2d_1d",
            build_inputs=_float_pair((3, 5), (5,), seed=530),
            lucid_fn=lambda a, b: a @ b,
            torch_fn=lambda a, b: a @ b,
            tol_class="matmul_f64",
            seed=530,
        ),
        ParityCase(
            name="dot_1d",
            build_inputs=_float_pair((7,), (7,), seed=540),
            lucid_fn=lucid.dot,
            torch_fn=torch.dot,
            tol_class="matmul_f64",
            seed=540,
        ),
        ParityCase(
            name="inner_1d",
            build_inputs=_float_pair((7,), (7,), seed=550),
            lucid_fn=lucid.inner,
            torch_fn=torch.inner,
            tol_class="matmul_f64",
            seed=550,
        ),
        ParityCase(
            name="outer_1d",
            build_inputs=_float_pair((5,), (4,), seed=560),
            lucid_fn=lucid.outer,
            torch_fn=torch.outer,
            tol_class="matmul_f64",
            seed=560,
        ),
        ParityCase(
            name="tensordot_axes1",
            build_inputs=_float_pair((3, 4, 5), (5, 6), seed=570),
            lucid_fn=lambda a, b: lucid.tensordot(a, b, axes=1),
            torch_fn=lambda a, b: torch.tensordot(a, b, dims=1),
            tol_class="matmul_f64",
            seed=570,
        ),
        ParityCase(
            name="tensordot_axes2",
            build_inputs=_float_pair((3, 4, 5), (4, 5, 6), seed=580),
            lucid_fn=lambda a, b: lucid.tensordot(a, b, axes=2),
            torch_fn=lambda a, b: torch.tensordot(a, b, dims=2),
            tol_class="matmul_f64",
            seed=580,
        ),
    ]
)

for fn_name, lop, top in [
    ("bitwise_and", lambda a, b: a & b, lambda a, b: a & b),
    ("bitwise_or", lambda a, b: a | b, lambda a, b: a | b),
]:
    CASES.append(
        ParityCase(
            name=f"{fn_name}_int",
            build_inputs=_int_pair((3, 4), (3, 4), seed=700),
            lucid_fn=lop,
            torch_fn=top,
            tol_class="integer_exact",
            check_backward=False,
            seed=700,
        )
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_bfunc_parity(case: ParityCase) -> None:
    run_parity_case(case)
