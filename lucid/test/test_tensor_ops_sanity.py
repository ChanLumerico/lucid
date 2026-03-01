from __future__ import annotations

import numpy as np
import pytest

import lucid
from lucid.test.core import OpSanityCase, OpSanitySuite


def _make_pair(shape=(2, 3), offset=0.3) -> tuple[np.ndarray, np.ndarray]:
    a = np.linspace(-1.0, 1.0, num=np.prod(shape)).reshape(shape)
    b = np.linspace(0.5, 1.5, num=np.prod(shape)).reshape(shape) + offset
    return a, b


def _make_matrix_pair() -> tuple[np.ndarray, np.ndarray]:
    a = np.array([[1.0, -2.0, 3.0], [0.5, 2.0, -1.0]])
    b = np.array([[1.0, 0.0], [2.0, -1.0], [0.5, 3.0]])
    return a, b


class TestBinaryOps(OpSanitySuite):
    cases = [
        OpSanityCase("add", lucid.add, lambda: _make_pair(), lambda a, b: a + b),
        OpSanityCase("sub", lucid.sub, lambda: _make_pair(), lambda a, b: a - b),
        OpSanityCase(
            "multiply", lucid.multiply, lambda: _make_pair(), lambda a, b: a * b
        ),
        OpSanityCase(
            "div", lucid.div, lambda: _make_pair(offset=1.5), lambda a, b: a / b
        ),
        OpSanityCase(
            "minimum", lucid.minimum, lambda: _make_pair(), lambda a, b: np.minimum(a, b)
        ),
        OpSanityCase(
            "maximum", lucid.maximum, lambda: _make_pair(), lambda a, b: np.maximum(a, b)
        ),
        OpSanityCase(
            "power",
            lucid.power,
            lambda: (np.full((2, 3), 2.0), np.array([[1.0, 2.0, 3.0], [1.5, 0.5, 2.5]])),
            lambda a, b: np.power(a, b),
        ),
        OpSanityCase("matmul", lucid.matmul, _make_matrix_pair, lambda a, b: a @ b),
        OpSanityCase(
            "dot",
            lucid.dot,
            lambda: (np.array([1.0, 2.0, 3.0]), np.array([0.5, -1.0, 2.0])),
            lambda a, b: np.dot(a, b),
            differentiable=False,
        ),
    ]

    @pytest.mark.parametrize("case", cases, ids=[case.name for case in cases])
    def test_forward_matches_numpy(self, case: OpSanityCase) -> None:
        self._run_forward(case)

    @pytest.mark.parametrize("case", cases, ids=[case.name for case in cases])
    def test_backward_sanity(self, case: OpSanityCase) -> None:
        self._run_backward(case)


class TestUnaryOps(OpSanitySuite):
    cases = [
        OpSanityCase("exp", lucid.exp, lambda: (np.array([-1.0, 0.0, 1.0]),), np.exp),
        OpSanityCase("log", lucid.log, lambda: (np.array([0.5, 1.0, 2.0]),), np.log),
        OpSanityCase("sqrt", lucid.sqrt, lambda: (np.array([0.25, 1.0, 4.0]),), np.sqrt),
        OpSanityCase("sin", lucid.sin, lambda: (np.array([-0.3, 0.2, 0.7]),), np.sin),
        OpSanityCase("cos", lucid.cos, lambda: (np.array([-0.3, 0.2, 0.7]),), np.cos),
        OpSanityCase("tanh", lucid.tanh, lambda: (np.array([-1.5, 0.0, 1.5]),), np.tanh),
        OpSanityCase("abs", lucid.abs, lambda: (np.array([-2.0, -0.5, 1.5]),), np.abs),
        OpSanityCase("square", lucid.square, lambda: (np.array([-2.0, -0.5, 1.5]),), np.square),
        OpSanityCase("cube", lucid.cube, lambda: (np.array([-2.0, -0.5, 1.5]),), lambda a: a**3),
        OpSanityCase(
            "floor",
            lucid.floor,
            lambda: (np.array([-2.2, -0.1, 1.8]),),
            np.floor,
            differentiable=False,
        ),
        OpSanityCase(
            "ceil",
            lucid.ceil,
            lambda: (np.array([-2.2, -0.1, 1.8]),),
            np.ceil,
            differentiable=False,
        ),
    ]

    @pytest.mark.parametrize("case", cases, ids=[case.name for case in cases])
    def test_forward_matches_numpy(self, case: OpSanityCase) -> None:
        self._run_forward(case)

    @pytest.mark.parametrize("case", cases, ids=[case.name for case in cases])
    def test_backward_sanity(self, case: OpSanityCase) -> None:
        self._run_backward(case)


class TestReductionAndUtilityOps(OpSanitySuite):
    cases = [
        OpSanityCase(
            "sum_axis1",
            lambda a: lucid.sum(a, axis=1),
            lambda: (np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0]]),),
            lambda a: np.sum(a, axis=1),
        ),
        OpSanityCase(
            "mean_axis0",
            lambda a: lucid.mean(a, axis=0),
            lambda: (np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0]]),),
            lambda a: np.mean(a, axis=0),
        ),
        OpSanityCase(
            "var_axis0",
            lambda a: lucid.var(a, axis=0),
            lambda: (np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0]]),),
            lambda a: np.var(a, axis=0),
            differentiable=False,
        ),
        OpSanityCase(
            "transpose",
            lambda a: lucid.transpose(a, axes=[1, 0]),
            lambda: (np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0]]),),
            lambda a: np.transpose(a, axes=[1, 0]),
        ),
        OpSanityCase(
            "swapaxes",
            lambda a: lucid.swapaxes(a, 0, 1),
            lambda: (np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0]]),),
            lambda a: np.swapaxes(a, 0, 1),
        ),
        OpSanityCase(
            "clip",
            lambda a: lucid.clip(a, min_value=-0.5, max_value=0.5),
            lambda: (np.array([-1.0, -0.2, 0.3, 2.0]),),
            lambda a: np.clip(a, -0.5, 0.5),
        ),
        OpSanityCase(
            "round",
            lambda a: lucid.round(a, decimals=1),
            lambda: (np.array([-1.24, -0.26, 0.31, 2.09]),),
            lambda a: np.round(a, decimals=1),
            differentiable=False,
        ),
        OpSanityCase(
            "cumsum",
            lambda a: lucid.cumsum(a, axis=0),
            lambda: (np.array([1.0, -1.0, 2.0]),),
            lambda a: np.cumsum(a, axis=0),
        ),
        OpSanityCase(
            "cumprod",
            lambda a: lucid.cumprod(a, axis=0),
            lambda: (np.array([1.5, -1.0, 2.0]),),
            lambda a: np.cumprod(a, axis=0),
        ),
    ]

    @pytest.mark.parametrize("case", cases, ids=[case.name for case in cases])
    def test_forward_matches_numpy(self, case: OpSanityCase) -> None:
        self._run_forward(case)

    @pytest.mark.parametrize("case", cases, ids=[case.name for case in cases])
    def test_backward_sanity(self, case: OpSanityCase) -> None:
        self._run_backward(case)
