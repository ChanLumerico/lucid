import numpy as np

import pytest

import torch

import torch.nn.functional as TF

import lucid

import lucid.nn.functional as F

from lucid.test.parity import data

from lucid.test.parity.core import ParityCase, TensorInput, run_parity_case


def _float(shape, *, seed: int, grad: bool = True):
    return lambda s: [
        TensorInput(data.random_floats(shape, seed=s), requires_grad=grad)
    ]


CASES: list[ParityCase] = []


def _linear_inputs(batch, in_f, out_f, *, seed: int, with_bias: bool):
    def _build(s):
        items = [
            TensorInput(data.random_floats((batch, in_f), seed=s), requires_grad=True),
            TensorInput(
                data.random_floats((out_f, in_f), seed=s + 1), requires_grad=True
            ),
        ]
        if with_bias:
            items.append(
                TensorInput(
                    data.random_floats((out_f,), seed=s + 2), requires_grad=True
                )
            )
        return items

    return _build


CASES.extend(
    [
        ParityCase(
            name="linear_with_bias",
            build_inputs=_linear_inputs(4, 5, 3, seed=5000, with_bias=True),
            lucid_fn=F.linear,
            torch_fn=TF.linear,
            tol_class="matmul_f32",
            seed=5000,
        ),
        ParityCase(
            name="linear_no_bias",
            build_inputs=_linear_inputs(4, 5, 3, seed=5010, with_bias=False),
            lucid_fn=F.linear,
            torch_fn=TF.linear,
            tol_class="matmul_f32",
            seed=5010,
        ),
        ParityCase(
            name="linear_batched_3d",
            build_inputs=lambda s: [
                TensorInput(data.random_floats((2, 4, 5), seed=s), requires_grad=True),
                TensorInput(data.random_floats((3, 5), seed=s + 1), requires_grad=True),
                TensorInput(data.random_floats((3,), seed=s + 2), requires_grad=True),
            ],
            lucid_fn=F.linear,
            torch_fn=TF.linear,
            tol_class="matmul_f32",
            seed=5020,
        ),
    ]
)


def _conv_inputs(N, C_in, H, W, C_out, kH, kW, *, seed: int, with_bias: bool = True):
    def _build(s):
        items = [
            TensorInput(
                data.random_floats((N, C_in, H, W), seed=s), requires_grad=True
            ),
            TensorInput(
                data.random_floats((C_out, C_in, kH, kW), seed=s + 1),
                requires_grad=True,
            ),
        ]
        if with_bias:
            items.append(
                TensorInput(
                    data.random_floats((C_out,), seed=s + 2), requires_grad=True
                )
            )
        return items

    return _build


CASES.extend(
    [
        ParityCase(
            name="conv2d_3x3_stride1_pad1",
            build_inputs=_conv_inputs(2, 3, 8, 8, 4, 3, 3, seed=5100),
            lucid_fn=lambda x, w, b: F.conv2d(x, w, b, stride=1, padding=1),
            torch_fn=lambda x, w, b: TF.conv2d(x, w, b, stride=1, padding=1),
            tol_class="matmul_f32",
            seed=5100,
        ),
        ParityCase(
            name="conv2d_stride2_no_bias",
            build_inputs=_conv_inputs(
                2, 3, 10, 10, 4, 3, 3, seed=5110, with_bias=False
            ),
            lucid_fn=lambda x, w: F.conv2d(x, w, stride=2, padding=1),
            torch_fn=lambda x, w: TF.conv2d(x, w, stride=2, padding=1),
            tol_class="matmul_f32",
            seed=5110,
        ),
        ParityCase(
            name="conv2d_groups2",
            build_inputs=_conv_inputs(2, 4, 6, 6, 4, 3, 3, seed=5120),
            lucid_fn=lambda x, w, b: F.conv2d(x, w, b, stride=1, padding=1, groups=2),
            torch_fn=lambda x, w, b: TF.conv2d(x, w, b, stride=1, padding=1, groups=2),
            tol_class="matmul_f32",
            seed=5120,
        ),
    ]
)

CASES[-1] = ParityCase(
    name="conv2d_groups2",
    build_inputs=lambda s: [
        TensorInput(data.random_floats((2, 4, 6, 6), seed=s), requires_grad=True),
        TensorInput(data.random_floats((4, 2, 3, 3), seed=s + 1), requires_grad=True),
        TensorInput(data.random_floats((4,), seed=s + 2), requires_grad=True),
    ],
    lucid_fn=lambda x, w, b: F.conv2d(x, w, b, stride=1, padding=1, groups=2),
    torch_fn=lambda x, w, b: TF.conv2d(x, w, b, stride=1, padding=1, groups=2),
    tol_class="matmul_f32",
    seed=5120,
)

CASES.extend(
    [
        ParityCase(
            name="conv1d_basic",
            build_inputs=lambda s: [
                TensorInput(data.random_floats((2, 3, 10), seed=s), requires_grad=True),
                TensorInput(
                    data.random_floats((4, 3, 3), seed=s + 1), requires_grad=True
                ),
                TensorInput(data.random_floats((4,), seed=s + 2), requires_grad=True),
            ],
            lucid_fn=lambda x, w, b: F.conv1d(x, w, b, stride=1, padding=1),
            torch_fn=lambda x, w, b: TF.conv1d(x, w, b, stride=1, padding=1),
            tol_class="matmul_f32",
            seed=5200,
        )
    ]
)

CASES.extend(
    [
        ParityCase(
            name="max_pool2d_k2_s2",
            build_inputs=_float((2, 3, 8, 8), seed=5300),
            lucid_fn=lambda x: F.max_pool2d(x, kernel_size=2, stride=2),
            torch_fn=lambda x: TF.max_pool2d(x, kernel_size=2, stride=2),
            tol_class="elementwise_f32",
            seed=5300,
        ),
        ParityCase(
            name="avg_pool2d_k2_s2",
            build_inputs=_float((2, 3, 8, 8), seed=5310),
            lucid_fn=lambda x: F.avg_pool2d(x, kernel_size=2, stride=2),
            torch_fn=lambda x: TF.avg_pool2d(x, kernel_size=2, stride=2),
            tol_class="elementwise_f32",
            seed=5310,
        ),
        ParityCase(
            name="adaptive_avg_pool2d_4x4",
            build_inputs=_float((2, 3, 8, 8), seed=5320),
            lucid_fn=lambda x: F.adaptive_avg_pool2d(x, output_size=(4, 4)),
            torch_fn=lambda x: TF.adaptive_avg_pool2d(x, output_size=(4, 4)),
            tol_class="elementwise_f32",
            seed=5320,
        ),
        ParityCase(
            name="adaptive_avg_pool2d_1x1",
            build_inputs=_float((2, 3, 8, 8), seed=5330),
            lucid_fn=lambda x: F.adaptive_avg_pool2d(x, output_size=(1, 1)),
            torch_fn=lambda x: TF.adaptive_avg_pool2d(x, output_size=(1, 1)),
            tol_class="elementwise_f32",
            seed=5330,
        ),
    ]
)

CASES.append(
    ParityCase(
        name="embedding_basic",
        build_inputs=lambda s: [
            TensorInput(
                data.int_array((3, 4), seed=s, low=0, high=10),
                requires_grad=False,
                dtype_override=lucid.Int64,
            ),
            TensorInput(data.random_floats((10, 5), seed=s + 1), requires_grad=True),
        ],
        lucid_fn=F.embedding,
        torch_fn=TF.embedding,
        tol_class="elementwise_f64",
        seed=5400,
    )
)


def _qkv_inputs(B, H, L, D, *, seed: int):
    def _build(s):
        return [
            TensorInput(data.random_floats((B, H, L, D), seed=s), requires_grad=True),
            TensorInput(
                data.random_floats((B, H, L, D), seed=s + 1), requires_grad=True
            ),
            TensorInput(
                data.random_floats((B, H, L, D), seed=s + 2), requires_grad=True
            ),
        ]

    return _build


CASES.extend(
    [
        ParityCase(
            name="sdpa_basic",
            build_inputs=_qkv_inputs(2, 4, 8, 16, seed=5500),
            lucid_fn=lambda q, k, v: F.scaled_dot_product_attention(q, k, v),
            torch_fn=lambda q, k, v: TF.scaled_dot_product_attention(q, k, v),
            tol_class="attention_f32",
            seed=5500,
        ),
        ParityCase(
            name="sdpa_causal",
            build_inputs=_qkv_inputs(2, 4, 8, 16, seed=5510),
            lucid_fn=lambda q, k, v: F.scaled_dot_product_attention(
                q, k, v, is_causal=True
            ),
            torch_fn=lambda q, k, v: TF.scaled_dot_product_attention(
                q, k, v, is_causal=True
            ),
            tol_class="attention_f32",
            seed=5510,
        ),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_conv_pool_linear_parity(case: ParityCase) -> None:
    run_parity_case(case)
