from typing import Any, Callable, ClassVar

import numpy as np
import torch

import lucid
from lucid.test.core import TensorOpWithTorchBase, TensorOpTorchCase
from lucid.test._utils.test_utils import _UtilsTupleTorchOpBase


def _unwrap_callable(operation: Any) -> Callable[..., Any]:
    if isinstance(operation, staticmethod):
        return operation.__func__
    if callable(operation):
        return operation
    raise TypeError(f"Expected callable operation, got: {type(operation)!r}")


class _LinalgUnaryTorchBase(TensorOpWithTorchBase):
    _collect_nested_test_classes = True
    case_name: ClassVar[str]
    input: ClassVar[Any]

    forward_op: ClassVar[Callable[..., Any]]
    torch_forward: ClassVar[Callable[..., Any] | None]
    expected_forward: ClassVar[Any]

    requires_grad: ClassVar[bool] = True
    dtype: ClassVar[Any | None] = None

    compare_grad_with_torch: ClassVar[bool] = True
    check_finite: ClassVar[bool] = True
    expected_input_grads: ClassVar[tuple[object | None, ...] | None] = None

    rtol: ClassVar[float] = 1e-7
    atol: ClassVar[float] = 1e-8

    def tensor_op_devices(self) -> tuple[str, ...]:
        return ("cpu",)

    def _build_case(self) -> TensorOpTorchCase:
        forward_op = vars(type(self)).get("forward_op")
        torch_forward = vars(type(self)).get("torch_forward")

        if isinstance(forward_op, staticmethod):
            forward_op = forward_op.__func__
        if isinstance(torch_forward, staticmethod):
            torch_forward = torch_forward.__func__

        return TensorOpTorchCase(
            name=self.case_name,
            build_inputs=lambda _case: (
                self.tensor(
                    self.input, requires_grad=self.requires_grad, dtype=self.dtype
                ),
            ),
            forward_op=forward_op,
            expected_forward=self.expected_forward,
            expected_input_grads=self.expected_input_grads,
            requires_grad=(self.requires_grad,),
            check_finite=self.check_finite,
            torch_forward=cast_callable(torch_forward),
            compare_grad_with_torch=self.compare_grad_with_torch,
            rtol=self.rtol,
            atol=self.atol,
        )

    def tensor_op_cases(self) -> tuple[TensorOpTorchCase, ...]:
        return (self._build_case(),)


class _LinalgBinaryTorchBase(TensorOpWithTorchBase):
    _collect_nested_test_classes = True
    case_name: ClassVar[str]
    left: ClassVar[Any]
    right: ClassVar[Any]

    forward_op: ClassVar[Callable[..., Any]]
    torch_forward: ClassVar[Callable[..., Any] | None]
    expected_forward: ClassVar[Any]

    requires_grad: ClassVar[bool | tuple[bool, bool]] = True
    compare_grad_with_torch: ClassVar[bool] = True

    check_finite: ClassVar[bool] = True
    expected_input_grads: ClassVar[tuple[object | None, ...] | None] = None
    dtype: ClassVar[Any | None] = None

    rtol: ClassVar[float] = 1e-7
    atol: ClassVar[float] = 1e-8

    def tensor_op_devices(self) -> tuple[str, ...]:
        return ("cpu",)

    def _build_case(self) -> TensorOpTorchCase:
        forward_op = vars(type(self)).get("forward_op")
        torch_forward = vars(type(self)).get("torch_forward")

        if isinstance(forward_op, staticmethod):
            forward_op = forward_op.__func__
        if isinstance(torch_forward, staticmethod):
            torch_forward = torch_forward.__func__

        requires_grad = self.requires_grad
        if isinstance(requires_grad, bool):
            requires_grad = (requires_grad, requires_grad)

        return TensorOpTorchCase(
            name=self.case_name,
            build_inputs=lambda _case: (
                self.tensor(
                    self.left, requires_grad=requires_grad[0], dtype=self.dtype
                ),
                self.tensor(
                    self.right, requires_grad=requires_grad[1], dtype=self.dtype
                ),
            ),
            forward_op=forward_op,
            expected_forward=self.expected_forward,
            expected_input_grads=self.expected_input_grads,
            requires_grad=requires_grad,
            check_finite=self.check_finite,
            torch_forward=cast_callable(torch_forward),
            compare_grad_with_torch=self.compare_grad_with_torch,
            rtol=self.rtol,
            atol=self.atol,
        )

    def tensor_op_cases(self) -> tuple[TensorOpTorchCase, ...]:
        return (self._build_case(),)


def cast_callable(value: Callable[..., Any] | None) -> Callable[..., Any] | None:
    if value is None:
        return None
    return _unwrap_callable(value)


class TestInv(_LinalgUnaryTorchBase):
    case_name = "inv"
    input = np.array([[4.0, 7.0], [2.0, 6.0]], dtype=np.float64)
    forward_op = staticmethod(lambda a: lucid.linalg.inv(a))
    torch_forward = staticmethod(torch.linalg.inv)
    expected_forward = np.linalg.inv(input)
    compare_grad_with_torch = False


class TestDet(_LinalgUnaryTorchBase):
    case_name = "det"
    input = np.array([[3.0, 2.0], [1.0, 4.0]], dtype=np.float64)
    forward_op = staticmethod(lambda a: lucid.linalg.det(a))
    torch_forward = staticmethod(torch.linalg.det)
    expected_forward = np.linalg.det(input)
    compare_grad_with_torch = False


class TestSolve(_LinalgBinaryTorchBase):
    case_name = "solve"
    left = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    right = np.array([[9.0], [8.0]], dtype=np.float64)
    forward_op = staticmethod(lambda a, b: lucid.linalg.solve(a, b))
    torch_forward = staticmethod(torch.linalg.solve)
    expected_forward = np.linalg.solve(left, right)
    compare_grad_with_torch = False


class TestCholesky(_LinalgUnaryTorchBase):
    case_name = "cholesky"
    input = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    forward_op = staticmethod(lambda a: lucid.linalg.cholesky(a))
    torch_forward = staticmethod(torch.linalg.cholesky)
    expected_forward = np.linalg.cholesky(input)
    compare_grad_with_torch = False


class TestNorm(_LinalgUnaryTorchBase):
    case_name = "norm"
    input = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    forward_op = staticmethod(
        lambda a: lucid.linalg.norm(a, ord=1, axis=1, keepdims=True)
    )
    torch_forward = staticmethod(
        lambda a: torch.linalg.norm(a, ord=1, dim=1, keepdim=True)
    )
    expected_forward = np.array([6.0, 15.0], dtype=np.float64).reshape(2, 1)


class TestMatrixPower(_LinalgUnaryTorchBase):
    case_name = "matrix_power"
    input = np.array([[2.0, 1.0], [0.0, 3.0]], dtype=np.float64)
    forward_op = staticmethod(lambda a: lucid.linalg.matrix_power(a, n=3))
    torch_forward = staticmethod(lambda a: torch.linalg.matrix_power(a, n=3))
    expected_forward = np.linalg.matrix_power(input, 3)
    compare_grad_with_torch = False


class TestPInv(_LinalgUnaryTorchBase):
    case_name = "pinv"
    input = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    forward_op = staticmethod(lambda a: lucid.linalg.pinv(a))
    torch_forward = staticmethod(torch.linalg.pinv)
    expected_forward = np.linalg.pinv(input)


class _LinalgTupleTorchBase(_UtilsTupleTorchOpBase):
    def tensor_op_devices(self) -> tuple[str, ...]:
        return ("cpu",)


class TestQR(_LinalgTupleTorchBase):
    case_name = "qr"
    input = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    inputs = (input,)
    forward_op = staticmethod(lambda a: lucid.linalg.qr(a))
    expected_forward = np.linalg.qr(input)
    compare_grad_with_torch = False
    check_finite = True


class TestSVD(_LinalgTupleTorchBase):
    case_name = "svd"
    input = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    inputs = (input,)
    forward_op = staticmethod(lambda a: lucid.linalg.svd(a))
    expected_forward = np.linalg.svd(input)
    compare_grad_with_torch = False
    check_finite = True


class TestEig(_LinalgTupleTorchBase):
    case_name = "eig"
    input = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    inputs = (input,)
    forward_op = staticmethod(lambda a: lucid.linalg.eig(a))
    expected_forward = np.linalg.eig(input)
    compare_grad_with_torch = False
    check_finite = True
