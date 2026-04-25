import numpy as np

import pytest

import torch

import lucid

import lucid.einops as LE

try:
    import einops as TE

    _HAS_EINOPS = True
except ImportError:
    _HAS_EINOPS = False
from lucid.test.parity import data

from lucid.test.parity.core import ParityCase, TensorInput, run_parity_case

pytestmark = pytest.mark.skipif(not _HAS_EINOPS, reason="einops package not installed")


def _f64(shape, *, seed: int, grad: bool = True):
    return lambda s: [
        TensorInput(data.random_floats(shape, seed=s), requires_grad=grad)
    ]


CASES: list[ParityCase] = []

CASES.extend(
    [
        ParityCase(
            name="rearrange_BHWC_to_BCHW",
            build_inputs=_f64((2, 4, 5, 3), seed=8000),
            lucid_fn=lambda a: LE.rearrange(a, "b h w c -> b c h w"),
            torch_fn=lambda a: TE.rearrange(a, "b h w c -> b c h w"),
            tol_class="elementwise_f64",
            seed=8000,
        ),
        ParityCase(
            name="rearrange_flatten_last_two",
            build_inputs=_f64((2, 3, 4), seed=8010),
            lucid_fn=lambda a: LE.rearrange(a, "b h w -> b (h w)"),
            torch_fn=lambda a: TE.rearrange(a, "b h w -> b (h w)"),
            tol_class="elementwise_f64",
            seed=8010,
        ),
        ParityCase(
            name="rearrange_split_axis",
            build_inputs=_f64((2, 12), seed=8020),
            lucid_fn=lambda a: LE.rearrange(a, "b (h w) -> b h w", h=3, w=4),
            torch_fn=lambda a: TE.rearrange(a, "b (h w) -> b h w", h=3, w=4),
            tol_class="elementwise_f64",
            seed=8020,
        ),
    ]
)

CASES.extend(
    [
        ParityCase(
            name="reduce_mean_spatial",
            build_inputs=_f64((2, 3, 4, 5), seed=8100),
            lucid_fn=lambda a: LE.reduce(a, "b c h w -> b c", reduction="mean"),
            torch_fn=lambda a: TE.reduce(a, "b c h w -> b c", reduction="mean"),
            tol_class="reduction_f64",
            seed=8100,
        ),
        ParityCase(
            name="reduce_sum_axis",
            build_inputs=_f64((2, 3, 4), seed=8110),
            lucid_fn=lambda a: LE.reduce(a, "b h w -> b w", reduction="sum"),
            torch_fn=lambda a: TE.reduce(a, "b h w -> b w", reduction="sum"),
            tol_class="reduction_f64",
            seed=8110,
        ),
    ]
)

CASES.append(
    ParityCase(
        name="repeat_add_axis",
        build_inputs=_f64((2, 3), seed=8200),
        lucid_fn=lambda a: LE.repeat(a, "b w -> b h w", h=4),
        torch_fn=lambda a: TE.repeat(a, "b w -> b h w", h=4),
        tol_class="elementwise_f64",
        seed=8200,
    )
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_einops_parity(case: ParityCase) -> None:
    run_parity_case(case)
