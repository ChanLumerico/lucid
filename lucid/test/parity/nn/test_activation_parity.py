import pytest

import torch

import torch.nn.functional as TF

import lucid

import lucid.nn.functional as F

from lucid.test.parity import data

from lucid.test.parity.core import ParityCase, TensorInput, run_parity_case


def _float(
    shape, *, seed: int, grad: bool = True, low: float = -2.0, high: float = 2.0
):
    return lambda s: [
        TensorInput(
            data.random_floats(shape, seed=s, low=low, high=high), requires_grad=grad
        )
    ]


def _pos(shape, *, seed: int, grad: bool = True):
    return lambda s: [TensorInput(data.pos_floats(shape, seed=s), requires_grad=grad)]


def _nonzero(shape, *, seed: int, grad: bool = True):
    return lambda s: [
        TensorInput(data.nonzero_floats(shape, seed=s), requires_grad=grad)
    ]


CASES: list[ParityCase] = []

PURE = [
    ("relu_2d", F.relu, TF.relu, _nonzero((3, 4), seed=2000)),
    ("relu_3d", F.relu, TF.relu, _nonzero((2, 3, 4), seed=2010)),
    (
        "leaky_relu_default",
        lambda a: F.leaky_relu(a),
        lambda a: TF.leaky_relu(a),
        _nonzero((3, 4), seed=2020),
    ),
    (
        "leaky_relu_slope02",
        lambda a: F.leaky_relu(a, negative_slope=0.2),
        lambda a: TF.leaky_relu(a, negative_slope=0.2),
        _nonzero((3, 4), seed=2030),
    ),
    ("elu_default", lambda a: F.elu(a), lambda a: TF.elu(a), _float((3, 4), seed=2040)),
    (
        "elu_alpha2",
        lambda a: F.elu(a, alpha=2.0),
        lambda a: TF.elu(a, alpha=2.0),
        _float((3, 4), seed=2050),
    ),
    ("selu_2d", F.selu, TF.selu, _float((3, 4), seed=2060)),
    (
        "gelu_2d",
        F.gelu,
        lambda a: TF.gelu(a, approximate="tanh"),
        _float((3, 4), seed=2070),
    ),
    ("sigmoid_2d", F.sigmoid, TF.sigmoid, _float((3, 4), seed=2080)),
    (
        "sigmoid_bounded",
        F.sigmoid,
        TF.sigmoid,
        _float((3, 4), seed=2085, low=-4.0, high=4.0),
    ),
    ("tanh_2d", F.tanh, TF.tanh, _float((3, 4), seed=2090)),
    ("silu_2d", F.silu, TF.silu, _float((3, 4), seed=2100)),
]

for name, lfn, tfn, build in PURE:
    CASES.append(
        ParityCase(
            name=name,
            build_inputs=build,
            lucid_fn=lfn,
            torch_fn=tfn,
            tol_class="elementwise_f64",
            seed=2000,
        )
    )
CASES.extend(
    [
        ParityCase(
            name="softmax_last_axis",
            build_inputs=_float((3, 5), seed=2200),
            lucid_fn=lambda a: F.softmax(a, axis=-1),
            torch_fn=lambda a: TF.softmax(a, dim=-1),
            tol_class="softmax_f32",
            seed=2200,
        ),
        ParityCase(
            name="softmax_first_axis",
            build_inputs=_float((3, 5), seed=2210),
            lucid_fn=lambda a: F.softmax(a, axis=0),
            torch_fn=lambda a: TF.softmax(a, dim=0),
            tol_class="softmax_f32",
            seed=2210,
        ),
        ParityCase(
            name="softmax_3d_middle",
            build_inputs=_float((2, 3, 5), seed=2220),
            lucid_fn=lambda a: F.softmax(a, axis=1),
            torch_fn=lambda a: TF.softmax(a, dim=1),
            tol_class="softmax_f32",
            seed=2220,
        ),
        ParityCase(
            name="softmax_large_logits",
            build_inputs=_float((3, 5), seed=2230, low=-20.0, high=20.0),
            lucid_fn=lambda a: F.softmax(a, axis=-1),
            torch_fn=lambda a: TF.softmax(a, dim=-1),
            tol_class="softmax_f32",
            seed=2230,
        ),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_activation_parity(case: ParityCase) -> None:
    run_parity_case(case)
