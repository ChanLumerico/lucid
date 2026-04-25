import numpy as np

import pytest

import lucid

import lucid.nn.functional as F

from lucid.test.parity import data

from lucid.test.parity.core import DeviceParityCase, TensorInput, run_device_parity_case


def _f32(shape, *, seed: int, grad: bool = True, low: float = -2.0, high: float = 2.0):
    return lambda s: [
        TensorInput(
            data.random_floats(shape, seed=s, low=low, high=high, dtype=np.float32),
            requires_grad=grad,
        )
    ]


def _f32_pos(shape, *, seed: int, grad: bool = True):
    return lambda s: [
        TensorInput(
            data.pos_floats(shape, seed=s, dtype=np.float32), requires_grad=grad
        )
    ]


CASES: list[DeviceParityCase] = []

for name, fn, build in [
    ("relu", F.relu, _f32_pos((3, 4), seed=10000)),
    ("sigmoid", F.sigmoid, _f32((3, 4), seed=10010)),
    ("tanh_act", F.tanh, _f32((3, 4), seed=10020)),
    ("silu", F.silu, _f32((3, 4), seed=10030)),
    ("gelu", F.gelu, _f32((3, 4), seed=10040)),
]:
    CASES.append(
        DeviceParityCase(
            name=f"act_{name}",
            build_inputs=build,
            lucid_fn=fn,
            tol_class="elementwise_f32",
            seed=10000,
        )
    )
CASES.extend(
    [
        DeviceParityCase(
            name="softmax_last",
            build_inputs=_f32((3, 5), seed=10100),
            lucid_fn=lambda a: F.softmax(a, axis=-1),
            tol_class="softmax_f32",
            seed=10100,
        )
    ]
)


def _linear_f32(batch, in_f, out_f, *, seed: int):
    return lambda s: [
        TensorInput(
            data.random_floats((batch, in_f), seed=s, dtype=np.float32),
            requires_grad=True,
        ),
        TensorInput(
            data.random_floats((out_f, in_f), seed=s + 1, dtype=np.float32),
            requires_grad=True,
        ),
        TensorInput(
            data.random_floats((out_f,), seed=s + 2, dtype=np.float32),
            requires_grad=True,
        ),
    ]


CASES.extend(
    [
        DeviceParityCase(
            name="linear",
            build_inputs=_linear_f32(4, 5, 3, seed=10200),
            lucid_fn=F.linear,
            tol_class="matmul_f32",
            seed=10200,
        ),
        DeviceParityCase(
            name="conv2d_3x3",
            build_inputs=lambda s: [
                TensorInput(
                    data.random_floats((2, 3, 8, 8), seed=s, dtype=np.float32),
                    requires_grad=True,
                ),
                TensorInput(
                    data.random_floats((4, 3, 3, 3), seed=s + 1, dtype=np.float32),
                    requires_grad=True,
                ),
                TensorInput(
                    data.random_floats((4,), seed=s + 2, dtype=np.float32),
                    requires_grad=True,
                ),
            ],
            lucid_fn=lambda x, w, b: F.conv2d(x, w, b, stride=1, padding=1),
            tol_class="matmul_f32",
            seed=10300,
        ),
    ]
)

CASES.extend(
    [
        DeviceParityCase(
            name="max_pool2d",
            build_inputs=_f32((2, 3, 8, 8), seed=10400),
            lucid_fn=lambda x: F.max_pool2d(x, kernel_size=2, stride=2),
            tol_class="elementwise_f32",
            seed=10400,
        ),
        DeviceParityCase(
            name="avg_pool2d",
            build_inputs=_f32((2, 3, 8, 8), seed=10410),
            lucid_fn=lambda x: F.avg_pool2d(x, kernel_size=2, stride=2),
            tol_class="elementwise_f32",
            seed=10410,
        ),
        DeviceParityCase(
            name="adaptive_avg_pool2d",
            build_inputs=_f32((2, 3, 8, 8), seed=10420),
            lucid_fn=lambda x: F.adaptive_avg_pool2d(x, output_size=(2, 2)),
            tol_class="elementwise_f32",
            seed=10420,
        ),
    ]
)

CASES.extend(
    [
        DeviceParityCase(
            name="layer_norm_last",
            build_inputs=_f32((3, 8), seed=10500),
            lucid_fn=lambda a: F.layer_norm(a, normalized_shape=(8,)),
            tol_class="norm_f32",
            seed=10500,
        ),
        DeviceParityCase(
            name="group_norm_g2",
            build_inputs=_f32((2, 4, 5, 5), seed=10510),
            lucid_fn=lambda a: F.group_norm(a, num_groups=2),
            tol_class="norm_f32",
            seed=10510,
        ),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_nn_device_parity(case: DeviceParityCase) -> None:
    run_device_parity_case(case)
