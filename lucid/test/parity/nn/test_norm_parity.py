import numpy as np

import pytest

import torch

import torch.nn.functional as TF

import lucid

import lucid.nn.functional as F

from lucid.test.parity import data

from lucid.test.parity.core import ParityCase, TensorInput, run_parity_case


def _image(shape, *, seed: int, grad: bool = True):
    return lambda s: [
        TensorInput(data.random_floats(shape, seed=s), requires_grad=grad)
    ]


def _norm_inputs_with_affine(shape, *, seed: int, param_shape):
    def _build(s):
        return [
            TensorInput(data.random_floats(shape, seed=s), requires_grad=True),
            TensorInput(
                data.random_floats(param_shape, seed=s + 1), requires_grad=True
            ),
            TensorInput(
                data.random_floats(param_shape, seed=s + 2), requires_grad=True
            ),
        ]

    return _build


CASES: list[ParityCase] = []

CASES.extend(
    [
        ParityCase(
            name="layer_norm_last_dim_no_affine",
            build_inputs=_image((3, 8), seed=4000),
            lucid_fn=lambda a: F.layer_norm(a, normalized_shape=(8,)),
            torch_fn=lambda a: TF.layer_norm(a, normalized_shape=(8,)),
            tol_class="norm_f32",
            seed=4000,
        ),
        ParityCase(
            name="layer_norm_last_dim_affine",
            build_inputs=_norm_inputs_with_affine((3, 8), seed=4010, param_shape=(8,)),
            lucid_fn=lambda a, w, b: F.layer_norm(
                a, normalized_shape=(8,), weight=w, bias=b
            ),
            torch_fn=lambda a, w, b: TF.layer_norm(
                a, normalized_shape=(8,), weight=w, bias=b
            ),
            tol_class="norm_f32",
            seed=4010,
        ),
        ParityCase(
            name="layer_norm_trailing_two_dims",
            build_inputs=_norm_inputs_with_affine(
                (2, 3, 4, 5), seed=4020, param_shape=(4, 5)
            ),
            lucid_fn=lambda a, w, b: F.layer_norm(
                a, normalized_shape=(4, 5), weight=w, bias=b
            ),
            torch_fn=lambda a, w, b: TF.layer_norm(
                a, normalized_shape=(4, 5), weight=w, bias=b
            ),
            tol_class="norm_f32",
            seed=4020,
        ),
    ]
)

CASES.extend(
    [
        ParityCase(
            name="group_norm_g2_no_affine",
            build_inputs=_image((2, 4, 5, 5), seed=4100),
            lucid_fn=lambda a: F.group_norm(a, num_groups=2),
            torch_fn=lambda a: TF.group_norm(a, num_groups=2),
            tol_class="norm_f32",
            seed=4100,
        ),
        ParityCase(
            name="group_norm_g4_affine",
            build_inputs=lambda s: [
                TensorInput(
                    data.random_floats((2, 8, 4, 4), seed=s), requires_grad=True
                ),
                TensorInput(data.random_floats((8,), seed=s + 1), requires_grad=True),
                TensorInput(data.random_floats((8,), seed=s + 2), requires_grad=True),
            ],
            lucid_fn=lambda a, w, b: F.group_norm(a, num_groups=4, weight=w, bias=b),
            torch_fn=lambda a, w, b: TF.group_norm(a, num_groups=4, weight=w, bias=b),
            tol_class="norm_f32",
            seed=4110,
        ),
    ]
)


def _bn_infer_inputs(shape, C, *, seed: int):
    def _build(s):
        return [
            TensorInput(data.random_floats(shape, seed=s), requires_grad=True),
            TensorInput(data.random_floats((C,), seed=s + 1), requires_grad=False),
            TensorInput(data.pos_floats((C,), seed=s + 2), requires_grad=False),
            TensorInput(data.random_floats((C,), seed=s + 3), requires_grad=True),
            TensorInput(data.random_floats((C,), seed=s + 4), requires_grad=True),
        ]

    return _build


CASES.append(
    ParityCase(
        name="batch_norm_infer_2d",
        build_inputs=_bn_infer_inputs((3, 4, 5, 5), C=4, seed=4200),
        lucid_fn=lambda a, rm, rv, w, b: F.batch_norm(
            a,
            running_mean=rm,
            running_var=rv,
            weight=w,
            bias=b,
            training=False,
            eps=1e-05,
        ),
        torch_fn=lambda a, rm, rv, w, b: TF.batch_norm(
            a,
            running_mean=rm,
            running_var=rv,
            weight=w,
            bias=b,
            training=False,
            eps=1e-05,
        ),
        tol_class="norm_f32",
        seed=4200,
    )
)

CASES.extend(
    [
        ParityCase(
            name="instance_norm_2d_no_affine",
            build_inputs=_image((2, 3, 5, 5), seed=4300),
            lucid_fn=lambda a: F.instance_norm(a),
            torch_fn=lambda a: TF.instance_norm(a),
            tol_class="norm_f32",
            seed=4300,
        ),
        ParityCase(
            name="instance_norm_with_affine",
            build_inputs=lambda s: [
                TensorInput(
                    data.random_floats((2, 3, 5, 5), seed=s), requires_grad=True
                ),
                TensorInput(data.random_floats((3,), seed=s + 1), requires_grad=True),
                TensorInput(data.random_floats((3,), seed=s + 2), requires_grad=True),
            ],
            lucid_fn=lambda a, w, b: F.instance_norm(a, weight=w, bias=b),
            torch_fn=lambda a, w, b: TF.instance_norm(a, weight=w, bias=b),
            tol_class="norm_f32",
            seed=4310,
        ),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_norm_parity(case: ParityCase) -> None:
    run_parity_case(case)
