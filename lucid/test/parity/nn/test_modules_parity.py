import numpy as np

import pytest

import torch

import torch.nn as tnn

import lucid

import lucid.nn as lnn

from lucid.test.parity import data

from lucid.test.parity.core import ModuleParityCase, TensorInput, run_module_parity_case


def _f64(shape, *, seed: int, grad: bool = True):
    return lambda s: [
        TensorInput(data.random_floats(shape, seed=s), requires_grad=grad)
    ]


CASES: list[ModuleParityCase] = []


def _build_linear(in_f, out_f, bias=True):
    def _b(seed):
        return (
            lnn.Linear(in_f, out_f, bias=bias),
            tnn.Linear(in_f, out_f, bias=bias).double(),
        )

    return _b


CASES.extend(
    [
        ModuleParityCase(
            name="Linear_with_bias",
            build_modules=_build_linear(5, 3, bias=True),
            build_inputs=_f64((4, 5), seed=6000),
            tol_class="matmul_f64",
            seed=6000,
        ),
        ModuleParityCase(
            name="Linear_no_bias",
            build_modules=_build_linear(5, 3, bias=False),
            build_inputs=_f64((4, 5), seed=6010),
            tol_class="matmul_f64",
            seed=6010,
        ),
        ModuleParityCase(
            name="Linear_3d_input",
            build_modules=_build_linear(5, 3, bias=True),
            build_inputs=_f64((2, 4, 5), seed=6020),
            tol_class="matmul_f64",
            seed=6020,
        ),
    ]
)


def _build_conv2d(C_in, C_out, k=3, stride=1, padding=1, bias=True, groups=1):
    def _b(seed):
        return (
            lnn.Conv2d(
                C_in,
                C_out,
                kernel_size=k,
                stride=stride,
                padding=padding,
                bias=bias,
                groups=groups,
            ),
            tnn.Conv2d(
                C_in,
                C_out,
                kernel_size=k,
                stride=stride,
                padding=padding,
                bias=bias,
                groups=groups,
            ).double(),
        )

    return _b


CASES.extend(
    [
        ModuleParityCase(
            name="Conv2d_3x3_pad1",
            build_modules=_build_conv2d(3, 4, k=3, padding=1),
            build_inputs=_f64((2, 3, 8, 8), seed=6100),
            tol_class="matmul_f64",
            seed=6100,
        ),
        ModuleParityCase(
            name="Conv2d_stride2_no_bias",
            build_modules=_build_conv2d(3, 4, k=3, stride=2, padding=1, bias=False),
            build_inputs=_f64((2, 3, 10, 10), seed=6110),
            tol_class="matmul_f64",
            seed=6110,
        ),
        ModuleParityCase(
            name="Conv2d_groups2",
            build_modules=_build_conv2d(4, 4, k=3, padding=1, groups=2),
            build_inputs=_f64((2, 4, 6, 6), seed=6120),
            tol_class="matmul_f64",
            seed=6120,
        ),
    ]
)


def _build_ln(normalized_shape, elementwise_affine=True, bias=True):
    def _b(seed):
        return (
            lnn.LayerNorm(
                normalized_shape, elementwise_affine=elementwise_affine, bias=bias
            ),
            tnn.LayerNorm(
                normalized_shape, elementwise_affine=elementwise_affine, bias=bias
            ).double(),
        )

    return _b


def _build_conv_transpose2d(C_in, C_out, k=3, stride=1, padding=0, bias=True):
    def _b(seed):
        return (
            lnn.ConvTranspose2d(
                C_in, C_out, kernel_size=k, stride=stride, padding=padding, bias=bias
            ),
            tnn.ConvTranspose2d(
                C_in, C_out, kernel_size=k, stride=stride, padding=padding, bias=bias
            ).double(),
        )

    return _b


CASES.extend(
    [
        ModuleParityCase(
            name="ConvTranspose2d_3x3_stride2",
            build_modules=_build_conv_transpose2d(3, 4, k=3, stride=2, padding=1),
            build_inputs=_f64((2, 3, 6, 6), seed=6150),
            tol_class="matmul_f64",
            seed=6150,
        ),
        ModuleParityCase(
            name="ConvTranspose2d_no_bias",
            build_modules=_build_conv_transpose2d(3, 2, k=2, stride=2, bias=False),
            build_inputs=_f64((2, 3, 4, 4), seed=6160),
            tol_class="matmul_f64",
            seed=6160,
        ),
    ]
)


def _build_bilinear(in1, in2, out, bias=True):
    def _b(seed):
        return (
            lnn.Bilinear(in1, in2, out, bias=bias),
            tnn.Bilinear(in1, in2, out, bias=bias).double(),
        )

    return _b


def _bilinear_inputs(seed):
    return [
        TensorInput(data.random_floats((4, 5), seed=seed), requires_grad=True),
        TensorInput(data.random_floats((4, 6), seed=seed + 1), requires_grad=True),
    ]


CASES.append(
    ModuleParityCase(
        name="Bilinear_basic",
        build_modules=_build_bilinear(5, 6, 3, bias=True),
        build_inputs=_bilinear_inputs,
        tol_class="matmul_f64",
        seed=6170,
    )
)


CASES.extend(
    [
        ModuleParityCase(
            name="LayerNorm_last_dim",
            build_modules=_build_ln((8,)),
            build_inputs=_f64((3, 8), seed=6200),
            tol_class="norm_f32",
            seed=6200,
        ),
        ModuleParityCase(
            name="LayerNorm_last_dim_no_bias",
            build_modules=_build_ln((8,), elementwise_affine=True, bias=False),
            build_inputs=_f64((3, 8), seed=6210),
            tol_class="norm_f32",
            seed=6210,
        ),
        ModuleParityCase(
            name="LayerNorm_no_affine",
            build_modules=_build_ln((8,), elementwise_affine=False),
            build_inputs=_f64((3, 8), seed=6220),
            tol_class="norm_f32",
            seed=6220,
        ),
    ]
)


def _build_gn(num_groups, num_channels, affine=True):
    def _b(seed):
        return (
            lnn.GroupNorm(num_groups, num_channels, affine=affine),
            tnn.GroupNorm(num_groups, num_channels, affine=affine).double(),
        )

    return _b


CASES.extend(
    [
        ModuleParityCase(
            name="GroupNorm_g4_affine",
            build_modules=_build_gn(4, 8, affine=True),
            build_inputs=_f64((2, 8, 4, 4), seed=6300),
            tol_class="norm_f32",
            seed=6300,
        ),
        ModuleParityCase(
            name="GroupNorm_g2_no_affine",
            build_modules=_build_gn(2, 4, affine=False),
            build_inputs=_f64((2, 4, 5, 5), seed=6310),
            tol_class="norm_f32",
            seed=6310,
        ),
    ]
)


def _build_embedding(vocab, dim):
    def _b(seed):
        return (lnn.Embedding(vocab, dim), tnn.Embedding(vocab, dim).double())

    return _b


def _embed_inputs(seed):
    return [
        TensorInput(
            data.int_array((3, 4), seed=seed, low=0, high=10),
            requires_grad=False,
            dtype_override=lucid.Int64,
        )
    ]


CASES.append(
    ModuleParityCase(
        name="Embedding_basic",
        build_modules=_build_embedding(10, 5),
        build_inputs=_embed_inputs,
        tol_class="elementwise_f64",
        seed=6400,
    )
)


def _build_act(lucid_cls, torch_cls, lucid_kwargs=None, torch_kwargs=None):
    lk = lucid_kwargs or {}
    tk = torch_kwargs or {}

    def _b(seed):
        return (lucid_cls(**lk), torch_cls(**tk).double())

    return _b


CASES.extend(
    [
        ModuleParityCase(
            name="ReLU_module",
            build_modules=_build_act(lnn.ReLU, tnn.ReLU),
            build_inputs=lambda s: [
                TensorInput(data.nonzero_floats((3, 4), seed=s), requires_grad=True)
            ],
            tol_class="elementwise_f64",
            seed=6500,
        ),
        ModuleParityCase(
            name="Sigmoid_module",
            build_modules=_build_act(lnn.Sigmoid, tnn.Sigmoid),
            build_inputs=_f64((3, 4), seed=6510),
            tol_class="elementwise_f64",
            seed=6510,
        ),
        ModuleParityCase(
            name="Tanh_module",
            build_modules=_build_act(lnn.Tanh, tnn.Tanh),
            build_inputs=_f64((3, 4), seed=6520),
            tol_class="elementwise_f64",
            seed=6520,
        ),
        ModuleParityCase(
            name="GELU_module",
            build_modules=_build_act(
                lnn.GELU, tnn.GELU, torch_kwargs={"approximate": "tanh"}
            ),
            build_inputs=_f64((3, 4), seed=6530),
            tol_class="elementwise_f64",
            seed=6530,
        ),
    ]
)


def _build_pool(lucid_cls, torch_cls, **kw):
    def _b(seed):
        return (lucid_cls(**kw), torch_cls(**kw).double())

    return _b


CASES.extend(
    [
        ModuleParityCase(
            name="MaxPool2d_k2_s2",
            build_modules=_build_pool(
                lnn.MaxPool2d, tnn.MaxPool2d, kernel_size=2, stride=2
            ),
            build_inputs=_f64((2, 3, 8, 8), seed=6600),
            tol_class="elementwise_f64",
            seed=6600,
        ),
        ModuleParityCase(
            name="AvgPool2d_k2_s2",
            build_modules=_build_pool(
                lnn.AvgPool2d, tnn.AvgPool2d, kernel_size=2, stride=2
            ),
            build_inputs=_f64((2, 3, 8, 8), seed=6610),
            tol_class="elementwise_f64",
            seed=6610,
        ),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_modules_parity(case: ModuleParityCase) -> None:
    run_module_parity_case(case)
