from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest

import lucid


ArrayFactory = Callable[[], tuple[np.ndarray, ...]]
ForwardCallable = Callable[..., lucid.Tensor]
ReferenceCallable = Callable[..., np.ndarray]


@dataclass(frozen=True)
class OpSanityCase:
    name: str
    op: ForwardCallable
    make_inputs: ArrayFactory
    ref: ReferenceCallable
    differentiable: bool = True
    atol: float = 1e-6
    rtol: float = 1e-6


class OpSanitySuite:
    """Reusable OOP-style sanity checks for Lucid Tensor operations."""

    cases: list[OpSanityCase] = []

    @staticmethod
    def _to_tensor(arr: np.ndarray, requires_grad: bool = False) -> lucid.Tensor:
        return lucid.tensor(arr.astype(np.float64), requires_grad=requires_grad)

    @classmethod
    @pytest.mark.parametrize("case", [], ids=[])
    def test_forward_matches_numpy(cls, case: OpSanityCase) -> None:  # pragma: no cover
        cls._run_forward(case)

    @classmethod
    @pytest.mark.parametrize("case", [], ids=[])
    def test_backward_sanity(cls, case: OpSanityCase) -> None:  # pragma: no cover
        cls._run_backward(case)

    @classmethod
    def _run_forward(cls, case: OpSanityCase) -> None:
        raw_inputs = case.make_inputs()
        tensors = [cls._to_tensor(x, requires_grad=False) for x in raw_inputs]

        out = case.op(*tensors)
        ref = case.ref(*raw_inputs)

        np.testing.assert_allclose(
            out.numpy(),
            np.asarray(ref),
            atol=case.atol,
            rtol=case.rtol,
            err_msg=f"forward mismatch for {case.name}",
        )

    @classmethod
    def _run_backward(cls, case: OpSanityCase) -> None:
        raw_inputs = case.make_inputs()
        tensors = [cls._to_tensor(x, requires_grad=case.differentiable) for x in raw_inputs]

        out = case.op(*tensors)
        loss = out if out.ndim == 0 else lucid.sum(out)
        loss.backward()

        for idx, tensor in enumerate(tensors):
            if case.differentiable:
                assert tensor.grad is not None, f"{case.name}: grad missing at input[{idx}]"
                assert np.asarray(tensor.grad).shape == tensor.shape
                assert np.isfinite(np.asarray(tensor.grad)).all()
            else:
                assert tensor.grad is None, f"{case.name}: unexpected grad at input[{idx}]"
