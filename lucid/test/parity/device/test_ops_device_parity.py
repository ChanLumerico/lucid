import numpy as np

import pytest

import lucid

from lucid.test.parity import data

from lucid.test.parity.core import DeviceParityCase, TensorInput, run_device_parity_case


def _f32(shape, *, seed: int, grad: bool = True, low: float = -2.0, high: float = 2.0):
    return lambda s: [
        TensorInput(
            data.random_floats(shape, seed=s, low=low, high=high, dtype=np.float32),
            requires_grad=grad,
        )
    ]


def _f32_pair(shape_a, shape_b, *, seed: int, grad: tuple[bool, bool] = (True, True)):
    return lambda s: [
        TensorInput(
            data.random_floats(shape_a, seed=s, dtype=np.float32), requires_grad=grad[0]
        ),
        TensorInput(
            data.random_floats(shape_b, seed=s + 1, dtype=np.float32),
            requires_grad=grad[1],
        ),
    ]


def _f32_pos(shape, *, seed: int, grad: bool = True):
    return lambda s: [
        TensorInput(
            data.pos_floats(shape, seed=s, dtype=np.float32), requires_grad=grad
        )
    ]


CASES: list[DeviceParityCase] = []

CASES.extend(
    [
        DeviceParityCase(
            name="matmul_1d_2d",
            build_inputs=_f32_pair((5,), (5, 4), seed=9000),
            lucid_fn=lambda a, b: a @ b,
            tol_class="matmul_f32",
            seed=9000,
        ),
        DeviceParityCase(
            name="matmul_2d_1d",
            build_inputs=_f32_pair((3, 5), (5,), seed=9010),
            lucid_fn=lambda a, b: a @ b,
            tol_class="matmul_f32",
            seed=9010,
        ),
        DeviceParityCase(
            name="matmul_2d_2d",
            build_inputs=_f32_pair((3, 5), (5, 4), seed=9020),
            lucid_fn=lambda a, b: a @ b,
            tol_class="matmul_f32",
            seed=9020,
        ),
        DeviceParityCase(
            name="T_backward",
            build_inputs=_f32((3, 4), seed=9030),
            lucid_fn=lambda a: a.T,
            tol_class="elementwise_f32",
            seed=9030,
        ),
        DeviceParityCase(
            name="stack_axis1",
            build_inputs=_f32_pair((3, 4), (3, 4), seed=9040),
            lucid_fn=lambda a, b: lucid.stack((a, b), axis=1),
            tol_class="elementwise_f32",
            seed=9040,
        ),
        DeviceParityCase(
            name="masked_fill_broadcast",
            build_inputs=lambda s: [
                TensorInput(
                    data.random_floats((3, 4), seed=s, dtype=np.float32),
                    requires_grad=True,
                ),
                TensorInput(data.bool_array((1, 4), seed=s + 1), requires_grad=False),
            ],
            lucid_fn=lambda a, m: lucid.masked_fill(a, m, value=0.0),
            tol_class="elementwise_f32",
            seed=9050,
        ),
        DeviceParityCase(
            name="trace_2d",
            build_inputs=_f32((5, 5), seed=9060),
            lucid_fn=lucid.trace,
            tol_class="reduction_f32",
            seed=9060,
        ),
        DeviceParityCase(
            name="trace_highdim",
            build_inputs=_f32((3, 3, 4), seed=9070),
            lucid_fn=lucid.trace,
            tol_class="reduction_f32",
            seed=9070,
        ),
    ]
)

for op_name, lop in [
    ("add", lambda a, b: a + b),
    ("sub", lambda a, b: a - b),
    ("mul", lambda a, b: a * b),
]:
    CASES.append(
        DeviceParityCase(
            name=f"{op_name}_2d",
            build_inputs=_f32_pair((3, 4), (3, 4), seed=9100),
            lucid_fn=lop,
            tol_class="elementwise_f32",
            seed=9100,
        )
    )
CASES.extend(
    [
        DeviceParityCase(
            name="exp_2d",
            build_inputs=_f32((3, 4), seed=9200, low=-1.0, high=1.0),
            lucid_fn=lucid.exp,
            tol_class="elementwise_f32",
            seed=9200,
        ),
        DeviceParityCase(
            name="log_pos",
            build_inputs=_f32_pos((3, 4), seed=9210),
            lucid_fn=lucid.log,
            tol_class="elementwise_f32",
            seed=9210,
        ),
        DeviceParityCase(
            name="tanh_2d",
            build_inputs=_f32((3, 4), seed=9220),
            lucid_fn=lucid.tanh,
            tol_class="elementwise_f32",
            seed=9220,
        ),
        DeviceParityCase(
            name="sum_axis",
            build_inputs=_f32((2, 3, 4), seed=9230),
            lucid_fn=lambda a: a.sum(axis=1),
            tol_class="reduction_f32",
            seed=9230,
        ),
        DeviceParityCase(
            name="mean_keepdims",
            build_inputs=_f32((2, 3, 4), seed=9240),
            lucid_fn=lambda a: a.mean(axis=-1, keepdims=True),
            tol_class="reduction_f32",
            seed=9240,
        ),
        DeviceParityCase(
            name="var_axis",
            build_inputs=_f32((3, 4), seed=9250),
            lucid_fn=lambda a: a.var(axis=1),
            tol_class="reduction_f32",
            seed=9250,
        ),
        DeviceParityCase(
            name="transpose_3d",
            build_inputs=_f32((2, 3, 4), seed=9260),
            lucid_fn=lambda a: a.transpose((1, 0, 2)),
            tol_class="elementwise_f32",
            seed=9260,
        ),
        DeviceParityCase(
            name="concat_axis1",
            build_inputs=_f32_pair((3, 4), (3, 4), seed=9270),
            lucid_fn=lambda a, b: lucid.concatenate((a, b), axis=1),
            tol_class="elementwise_f32",
            seed=9270,
        ),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_device_parity(case: DeviceParityCase) -> None:
    run_device_parity_case(case)
