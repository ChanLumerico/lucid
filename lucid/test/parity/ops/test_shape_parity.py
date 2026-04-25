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

CASES.extend(
    [
        ParityCase(
            name="reshape_2d_to_1d",
            build_inputs=_float((3, 4), seed=4000),
            lucid_fn=lambda a: lucid.reshape(a, (12,)),
            torch_fn=lambda a: a.reshape(12),
            tol_class="shape_exact",
            seed=4000,
        ),
        ParityCase(
            name="reshape_2d_to_3d",
            build_inputs=_float((2, 6), seed=4010),
            lucid_fn=lambda a: lucid.reshape(a, (2, 3, 2)),
            torch_fn=lambda a: a.reshape(2, 3, 2),
            tol_class="shape_exact",
            seed=4010,
        ),
        ParityCase(
            name="reshape_round_trip",
            build_inputs=_float((3, 4), seed=4020),
            lucid_fn=lambda a: lucid.reshape(lucid.reshape(a, (12,)), (3, 4)),
            torch_fn=lambda a: a.reshape(12).reshape(3, 4),
            tol_class="shape_exact",
            seed=4020,
        ),
        ParityCase(
            name="flatten_all",
            build_inputs=_float((2, 3, 4), seed=4030),
            lucid_fn=lambda a: lucid.flatten(a),
            torch_fn=lambda a: a.flatten(),
            tol_class="shape_exact",
            seed=4030,
        ),
        ParityCase(
            name="flatten_from_axis_1",
            build_inputs=_float((2, 3, 4), seed=4040),
            lucid_fn=lambda a: lucid.flatten(a, start_axis=1),
            torch_fn=lambda a: a.flatten(start_dim=1),
            tol_class="shape_exact",
            seed=4040,
        ),
        ParityCase(
            name="ravel",
            build_inputs=_float((2, 3, 4), seed=4050),
            lucid_fn=lambda a: lucid.ravel(a),
            torch_fn=lambda a: a.reshape(-1),
            tol_class="shape_exact",
            seed=4050,
        ),
    ]
)

CASES.extend(
    [
        ParityCase(
            name="squeeze_all",
            build_inputs=_float((1, 3, 1, 4), seed=4100),
            lucid_fn=lambda a: lucid.squeeze(a),
            torch_fn=lambda a: a.squeeze(),
            tol_class="shape_exact",
            seed=4100,
        ),
        ParityCase(
            name="squeeze_axis",
            build_inputs=_float((1, 3, 1, 4), seed=4110),
            lucid_fn=lambda a: lucid.squeeze(a, axis=0),
            torch_fn=lambda a: a.squeeze(dim=0),
            tol_class="shape_exact",
            seed=4110,
        ),
        ParityCase(
            name="unsqueeze_first",
            build_inputs=_float((3, 4), seed=4120),
            lucid_fn=lambda a: lucid.unsqueeze(a, axis=0),
            torch_fn=lambda a: a.unsqueeze(0),
            tol_class="shape_exact",
            seed=4120,
        ),
        ParityCase(
            name="unsqueeze_last",
            build_inputs=_float((3, 4), seed=4130),
            lucid_fn=lambda a: lucid.unsqueeze(a, axis=-1),
            torch_fn=lambda a: a.unsqueeze(-1),
            tol_class="shape_exact",
            seed=4130,
        ),
    ]
)


def _two_float(shape, *, seed: int, grad: bool = True):
    return lambda s: [
        TensorInput(data.random_floats(shape, seed=s), requires_grad=grad),
        TensorInput(data.random_floats(shape, seed=s + 1), requires_grad=grad),
    ]


CASES.extend(
    [
        ParityCase(
            name="stack_axis0",
            build_inputs=_two_float((3, 4), seed=4200),
            lucid_fn=lambda a, b: lucid.stack((a, b), axis=0),
            torch_fn=lambda a, b: torch.stack((a, b), dim=0),
            tol_class="elementwise_f64",
            seed=4200,
        ),
        ParityCase(
            name="stack_axis1",
            build_inputs=_two_float((3, 4), seed=4210),
            lucid_fn=lambda a, b: lucid.stack((a, b), axis=1),
            torch_fn=lambda a, b: torch.stack((a, b), dim=1),
            tol_class="elementwise_f64",
            seed=4210,
        ),
        ParityCase(
            name="concat_axis0",
            build_inputs=_two_float((3, 4), seed=4220),
            lucid_fn=lambda a, b: lucid.concatenate((a, b), axis=0),
            torch_fn=lambda a, b: torch.cat((a, b), dim=0),
            tol_class="elementwise_f64",
            seed=4220,
        ),
        ParityCase(
            name="concat_axis1",
            build_inputs=_two_float((3, 4), seed=4230),
            lucid_fn=lambda a, b: lucid.concatenate((a, b), axis=1),
            torch_fn=lambda a, b: torch.cat((a, b), dim=1),
            tol_class="elementwise_f64",
            seed=4230,
        ),
        ParityCase(
            name="hstack_2d",
            build_inputs=_two_float((3, 4), seed=4240),
            lucid_fn=lambda a, b: lucid.hstack((a, b)),
            torch_fn=lambda a, b: torch.hstack((a, b)),
            tol_class="elementwise_f64",
            seed=4240,
        ),
        ParityCase(
            name="vstack_2d",
            build_inputs=_two_float((3, 4), seed=4250),
            lucid_fn=lambda a, b: lucid.vstack((a, b)),
            torch_fn=lambda a, b: torch.vstack((a, b)),
            tol_class="elementwise_f64",
            seed=4250,
        ),
    ]
)

CASES.extend(
    [
        ParityCase(
            name="tile_2d",
            build_inputs=_float((2, 3), seed=4300),
            lucid_fn=lambda a: lucid.tile(a, reps=(2, 3)),
            torch_fn=lambda a: a.repeat(2, 3),
            tol_class="elementwise_f64",
            seed=4300,
        ),
        ParityCase(
            name="tile_1d",
            build_inputs=_float((4,), seed=4310),
            lucid_fn=lambda a: lucid.tile(a, reps=3),
            torch_fn=lambda a: a.repeat(3),
            tol_class="elementwise_f64",
            seed=4310,
        ),
    ]
)

CASES.extend(
    [
        ParityCase(
            name="pad_2d_symmetric",
            build_inputs=_float((3, 4), seed=4400),
            lucid_fn=lambda a: lucid.pad(a, pad_width=((1, 1), (2, 2))),
            torch_fn=lambda a: torch.nn.functional.pad(a, (2, 2, 1, 1)),
            tol_class="elementwise_f64",
            seed=4400,
        ),
        ParityCase(
            name="pad_1d",
            build_inputs=_float((5,), seed=4410),
            lucid_fn=lambda a: lucid.pad(a, pad_width=(2, 3)),
            torch_fn=lambda a: torch.nn.functional.pad(a, (2, 3)),
            tol_class="elementwise_f64",
            seed=4410,
        ),
    ]
)

CASES.extend(
    [
        ParityCase(
            name="broadcast_to_row",
            build_inputs=_float((1, 4), seed=4500),
            lucid_fn=lambda a: lucid.broadcast_to(a, (3, 4)),
            torch_fn=lambda a: a.expand(3, 4).contiguous(),
            tol_class="elementwise_f64",
            seed=4500,
        )
    ]
)

CASES.extend(
    [
        ParityCase(
            name="split_by_size_first",
            build_inputs=_float((4, 6), seed=4600),
            lucid_fn=lambda a: lucid.split(a, 3, axis=1)[0],
            torch_fn=lambda a: torch.chunk(a, 2, dim=1)[0],
            tol_class="elementwise_f64",
            seed=4600,
        ),
        ParityCase(
            name="chunk_axis0_first",
            build_inputs=_float((6, 4), seed=4610),
            lucid_fn=lambda a: lucid.chunk(a, 3, axis=0)[0],
            torch_fn=lambda a: torch.chunk(a, 3, dim=0)[0],
            tol_class="elementwise_f64",
            seed=4610,
        ),
        ParityCase(
            name="chunk_axis1_last",
            build_inputs=_float((4, 6), seed=4620),
            lucid_fn=lambda a: lucid.chunk(a, 3, axis=1)[-1],
            torch_fn=lambda a: torch.chunk(a, 3, dim=1)[-1],
            tol_class="elementwise_f64",
            seed=4620,
        ),
    ]
)

CASES.extend(
    [
        ParityCase(
            name="tril_diag0",
            build_inputs=_float((5, 5), seed=4700),
            lucid_fn=lambda a: lucid.tril(a),
            torch_fn=lambda a: torch.tril(a),
            tol_class="elementwise_f64",
            seed=4700,
        ),
        ParityCase(
            name="tril_diag_offset",
            build_inputs=_float((5, 5), seed=4710),
            lucid_fn=lambda a: lucid.tril(a, diagonal=1),
            torch_fn=lambda a: torch.tril(a, diagonal=1),
            tol_class="elementwise_f64",
            seed=4710,
        ),
        ParityCase(
            name="triu_diag0",
            build_inputs=_float((5, 5), seed=4720),
            lucid_fn=lambda a: lucid.triu(a),
            torch_fn=lambda a: torch.triu(a),
            tol_class="elementwise_f64",
            seed=4720,
        ),
        ParityCase(
            name="triu_diag_offset_neg",
            build_inputs=_float((5, 5), seed=4730),
            lucid_fn=lambda a: lucid.triu(a, diagonal=-1),
            torch_fn=lambda a: torch.triu(a, diagonal=-1),
            tol_class="elementwise_f64",
            seed=4730,
        ),
    ]
)

CASES.extend(
    [
        ParityCase(
            name="roll_1d",
            build_inputs=_float((6,), seed=4800, grad=False),
            lucid_fn=lambda a: lucid.roll(a, shifts=2, axis=None),
            torch_fn=lambda a: torch.roll(a, shifts=2),
            tol_class="shape_exact",
            check_backward=False,
            seed=4800,
        ),
        ParityCase(
            name="roll_2d_axis",
            build_inputs=_float((3, 4), seed=4810, grad=False),
            lucid_fn=lambda a: lucid.roll(a, shifts=1, axis=1),
            torch_fn=lambda a: torch.roll(a, shifts=1, dims=1),
            tol_class="shape_exact",
            check_backward=False,
            seed=4810,
        ),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_shape_parity(case: ParityCase) -> None:
    run_parity_case(case)
