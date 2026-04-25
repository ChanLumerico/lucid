import numpy as np

import pytest

import torch

import lucid

import lucid.linalg as la

from lucid.test.parity import data

from lucid.test.parity.core import ParityCase, TensorInput, run_parity_case


def _square(n, *, seed: int, grad: bool = True, well_conditioned: bool = True):
    def _build(s):
        rng = np.random.default_rng(s)
        if well_conditioned:
            arr = rng.standard_normal((n, n)).astype(np.float64)
            arr = arr @ arr.T + n * np.eye(n)
        else:
            arr = rng.standard_normal((n, n)).astype(np.float64)
        return [TensorInput(arr, requires_grad=grad)]

    return _build


def _general_2d(rows, cols, *, seed: int, grad: bool = True):
    return lambda s: [
        TensorInput(data.random_floats((rows, cols), seed=s), requires_grad=grad)
    ]


CASES: list[ParityCase] = []

CASES.extend(
    [
        ParityCase(
            name="norm_l2_axis_last",
            build_inputs=_general_2d(3, 4, seed=7000),
            lucid_fn=lambda a: la.norm(a, ord=2, axis=-1),
            torch_fn=lambda a: torch.linalg.norm(a, ord=2, dim=-1),
            tol_class="linalg_f64",
            seed=7000,
        ),
        ParityCase(
            name="norm_l1_axis",
            build_inputs=_general_2d(3, 4, seed=7010),
            lucid_fn=lambda a: la.norm(a, ord=1, axis=-1),
            torch_fn=lambda a: torch.linalg.norm(a, ord=1, dim=-1),
            tol_class="linalg_f64",
            seed=7010,
        ),
    ]
)

CASES.append(
    ParityCase(
        name="inv_spd_3x3",
        build_inputs=_square(3, seed=7100),
        lucid_fn=la.inv,
        torch_fn=torch.linalg.inv,
        tol_class="linalg_f64",
        seed=7100,
    )
)

CASES.extend(
    [
        ParityCase(
            name="det_3x3",
            build_inputs=_square(3, seed=7200, grad=False),
            lucid_fn=la.det,
            torch_fn=torch.linalg.det,
            tol_class="linalg_f64",
            check_backward=False,
            seed=7200,
        ),
        ParityCase(
            name="matrix_power_2",
            build_inputs=_square(3, seed=7210, grad=False),
            lucid_fn=lambda a: la.matrix_power(a, 2),
            torch_fn=lambda a: torch.linalg.matrix_power(a, 2),
            tol_class="linalg_f64",
            check_backward=False,
            seed=7210,
        ),
    ]
)


def _solve_inputs(n, *, seed: int):
    def _build(s):
        rng = np.random.default_rng(s)
        A = rng.standard_normal((n, n)).astype(np.float64)
        A = A @ A.T + n * np.eye(n)
        b = rng.standard_normal((n,)).astype(np.float64)
        return [
            TensorInput(A, requires_grad=False),
            TensorInput(b, requires_grad=False),
        ]

    return _build


CASES.append(
    ParityCase(
        name="solve_3x3_b1",
        build_inputs=_solve_inputs(3, seed=7300),
        lucid_fn=la.solve,
        torch_fn=torch.linalg.solve,
        tol_class="linalg_f64",
        check_backward=False,
        seed=7300,
    )
)

CASES.extend(
    [
        ParityCase(
            name="cholesky_3x3",
            build_inputs=_square(3, seed=7400, grad=False),
            lucid_fn=la.cholesky,
            torch_fn=torch.linalg.cholesky,
            tol_class="linalg_f64",
            check_backward=False,
            seed=7400,
        )
    ]
)

CASES.append(
    ParityCase(
        name="pinv_3x4",
        build_inputs=_general_2d(3, 4, seed=7500, grad=False),
        lucid_fn=la.pinv,
        torch_fn=torch.linalg.pinv,
        tol_class="linalg_f64",
        check_backward=False,
        seed=7500,
    )
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_linalg_parity(case: ParityCase) -> None:
    run_parity_case(case)
