import numpy as np

import pytest

import torch

import torch.nn as tnn

import lucid

import lucid.nn as lnn

from lucid.test.parity import data

from lucid.test.parity.core import ModuleParityCase, TensorInput, run_module_parity_case


def _seq(L, N, H, *, seed: int, grad: bool = True):
    return lambda s: [
        TensorInput(data.random_floats((L, N, H), seed=s), requires_grad=grad)
    ]


def _build_rnn(input_size, hidden_size, num_layers=1, nonlinearity="tanh"):
    def _b(seed):
        return (
            lnn.RNN(
                input_size,
                hidden_size,
                num_layers=num_layers,
                nonlinearity=nonlinearity,
            ),
            tnn.RNN(
                input_size,
                hidden_size,
                num_layers=num_layers,
                nonlinearity=nonlinearity,
            ).double(),
        )

    return _b


def _build_lstm(input_size, hidden_size, num_layers=1):
    def _b(seed):
        return (
            lnn.LSTM(input_size, hidden_size, num_layers=num_layers),
            tnn.LSTM(input_size, hidden_size, num_layers=num_layers).double(),
        )

    return _b


def _build_gru(input_size, hidden_size, num_layers=1):
    def _b(seed):
        return (
            lnn.GRU(input_size, hidden_size, num_layers=num_layers),
            tnn.GRU(input_size, hidden_size, num_layers=num_layers).double(),
        )

    return _b


CASES: list[ModuleParityCase] = [
    ModuleParityCase(
        name="RNN_tanh_1layer",
        build_modules=_build_rnn(4, 6, num_layers=1, nonlinearity="tanh"),
        build_inputs=_seq(5, 2, 4, seed=8500),
        tol_class="norm_f32",
        output_index=0,
        seed=8500,
    ),
    ModuleParityCase(
        name="RNN_relu_1layer",
        build_modules=_build_rnn(4, 6, num_layers=1, nonlinearity="relu"),
        build_inputs=_seq(5, 2, 4, seed=8510),
        tol_class="norm_f32",
        output_index=0,
        seed=8510,
    ),
    ModuleParityCase(
        name="LSTM_1layer",
        build_modules=_build_lstm(4, 6, num_layers=1),
        build_inputs=_seq(5, 2, 4, seed=8520),
        tol_class="norm_f32",
        output_index=0,
        seed=8520,
    ),
    ModuleParityCase(
        name="GRU_1layer",
        build_modules=_build_gru(4, 6, num_layers=1),
        build_inputs=_seq(5, 2, 4, seed=8530),
        tol_class="norm_f32",
        output_index=0,
        seed=8530,
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_rnn_parity(case: ModuleParityCase) -> None:
    run_module_parity_case(case)
