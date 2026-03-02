from typing import Any, Callable, ClassVar

import numpy as np
import torch

import lucid
from lucid._func import bfunc
from lucid.test.core import TensorOpTorchCase, TensorOpTorchBase


_ARITH_LEFT: list[list[float]] = [[1.0, 2.0], [3.0, 4.0]]
_ARITH_RIGHT: list[list[float]] = [[2.0, 1.0], [0.0, -1.0]]

_CMP_LEFT: list[list[float]] = _ARITH_LEFT
_CMP_RIGHT: list[list[float]] = [[2.0, 2.0], [3.0, 1.0]]

_MATRIX_RIGHT: list[list[float]] = [[5.0, 6.0], [7.0, 8.0]]

_BITWISE_LEFT: list[list[int]] = [[1, 2], [3, 4]]
_BITWISE_RIGHT: list[list[int]] = [[2, 1], [0, 7]]


def _bind_bfunc(op: type[Any]) -> Callable[[Any, Any], Any]:
    def _forward(left: Any, right: Any) -> Any:
        return op()(left, right)

    return _forward


class _TestBinaryFuncOpBase(TensorOpTorchBase):
    case_name: ClassVar[str]
    left: ClassVar[Any]
    right: ClassVar[Any]

    forward_op: ClassVar[Callable[[Any, Any], Any]]
    torch_forward: ClassVar[Callable[[Any, Any], Any]]
    expected_forward: ClassVar[Any]

    requires_grad: ClassVar[bool | tuple[bool, bool]] = True
    dtype: ClassVar[Any | None] = None
    compare_grad_with_torch: ClassVar[bool] = True

    check_finite: ClassVar[bool] = True
    expected_input_grads: ClassVar[tuple[object | None, ...] | None] = None

    def tensor_op_devices(self) -> tuple[str, ...]:
        return ("cpu", "gpu")

    def _build_binary_inputs(
        self,
        left: Any,
        right: Any,
        *,
        dtype: Any | None = None,
        requires_grad: bool | tuple[bool, bool] = True,
    ) -> tuple[Any, ...]:
        if isinstance(requires_grad, bool):
            requires_grad = (requires_grad, requires_grad)

        if dtype is None:
            return (
                self.tensor(left, requires_grad=requires_grad[0]),
                self.tensor(right, requires_grad=requires_grad[1]),
            )

        return (
            self.tensor(left, requires_grad=requires_grad[0], dtype=dtype),
            self.tensor(right, requires_grad=requires_grad[1], dtype=dtype),
        )

    def _case_grad_requires(self) -> tuple[bool, bool]:
        if isinstance(self.requires_grad, tuple):
            return self.requires_grad
        return (self.requires_grad, self.requires_grad)

    def _build_case(self) -> TensorOpTorchCase:
        forward_op = vars(type(self)).get("forward_op")
        torch_forward = vars(type(self)).get("torch_forward")

        if isinstance(forward_op, staticmethod):
            forward_op = forward_op.__func__
        if isinstance(torch_forward, staticmethod):
            torch_forward = torch_forward.__func__

        return TensorOpTorchCase(
            name=self.case_name,
            build_inputs=lambda _case: self._build_binary_inputs(
                self.left,
                self.right,
                dtype=self.dtype,
                requires_grad=self._case_grad_requires(),
            ),
            forward_op=forward_op,
            expected_forward=self.expected_forward,
            expected_input_grads=self.expected_input_grads,
            requires_grad=self._case_grad_requires(),
            check_finite=self.check_finite,
            torch_forward=torch_forward,
            compare_grad_with_torch=self.compare_grad_with_torch,
        )

    def tensor_op_cases(self) -> tuple[TensorOpTorchCase, ...]:
        return (self._build_case(),)


class _TestBinaryNoGradFuncOpBase(_TestBinaryFuncOpBase):
    requires_grad: ClassVar[tuple[bool, bool]] = (False, False)
    compare_grad_with_torch: ClassVar[bool] = False


class TestAdd(_TestBinaryFuncOpBase):
    case_name = "add"
    left = _ARITH_LEFT
    right = _ARITH_RIGHT
    forward_op = lucid.add
    torch_forward = torch.add
    expected_forward = np.array([[3.0, 3.0], [3.0, 3.0]], dtype=np.float64)


class TestSub(_TestBinaryFuncOpBase):
    case_name = "sub"
    left = _ARITH_LEFT
    right = _ARITH_RIGHT
    forward_op = lucid.sub
    torch_forward = torch.sub
    expected_forward = np.array([[-1.0, 1.0], [3.0, 5.0]], dtype=np.float64)


class TestMultiply(_TestBinaryFuncOpBase):
    case_name = "mul"
    left = _ARITH_LEFT
    right = _ARITH_RIGHT
    forward_op = lucid.multiply
    torch_forward = torch.mul
    expected_forward = np.array([[2.0, 2.0], [0.0, -4.0]], dtype=np.float64)


class TestDiv(_TestBinaryFuncOpBase):
    __test__ = False

    class TestTruediv(_TestBinaryFuncOpBase):
        case_name = "truediv"
        left = [[1.0, 2.0], [3.0, 3.0]]
        right = [[2.0, 1.0], [2.0, 2.0]]
        forward_op = lucid.div
        torch_forward = torch.div
        expected_forward = np.array([[0.5, 2.0], [1.5, 1.5]], dtype=np.float64)

    class TestFloorDiv(_TestBinaryNoGradFuncOpBase):
        case_name = "floordiv_no_grad"
        left = [[1, 2], [3, 5]]
        right = [[2, 3], [2, 2]]
        forward_op = staticmethod(lambda a, b: lucid.div(a, b, floor=True))
        torch_forward = torch.floor_divide
        dtype = lucid.Int64
        expected_forward = np.array([[0, 0], [1, 2]], dtype=np.int64)


class TestMinimum(_TestBinaryFuncOpBase):
    case_name = "minimum"
    left = _ARITH_LEFT
    right = _ARITH_RIGHT
    forward_op = lucid.minimum
    torch_forward = torch.minimum
    expected_forward = np.array([[1.0, 1.0], [0.0, -1.0]], dtype=np.float64)


class TestMaximum(_TestBinaryFuncOpBase):
    case_name = "maximum"
    left = _ARITH_LEFT
    right = _ARITH_RIGHT
    forward_op = lucid.maximum
    torch_forward = torch.maximum
    expected_forward = np.array([[2.0, 2.0], [3.0, 4.0]], dtype=np.float64)


class TestPower(_TestBinaryFuncOpBase):
    case_name = "power"
    left = [[1.0, 2.0], [3.0, 4.0]]
    right = [[2.0, 3.0], [1.0, 2.0]]
    forward_op = lucid.power
    torch_forward = torch.pow
    expected_forward = np.array([[1.0, 8.0], [3.0, 16.0]], dtype=np.float64)


class TestDot(_TestBinaryFuncOpBase):
    case_name = "dot"
    left = [1.0, 2.0, 3.0]
    right = [4.0, 5.0, 6.0]
    forward_op = lucid.dot
    torch_forward = torch.dot
    expected_forward = np.array(32.0)


class TestInner(_TestBinaryFuncOpBase):
    case_name = "inner"
    left = _ARITH_LEFT
    right = _MATRIX_RIGHT
    forward_op = lucid.inner
    torch_forward = torch.inner
    expected_forward = np.array([[17.0, 23.0], [39.0, 53.0]], dtype=np.float64)


class TestOuter(_TestBinaryFuncOpBase):
    case_name = "outer"
    left = [1.0, 2.0, 3.0]
    right = [4.0, 5.0]
    forward_op = lucid.outer
    torch_forward = torch.outer
    expected_forward = np.array(
        [[4.0, 5.0], [8.0, 10.0], [12.0, 15.0]], dtype=np.float64
    )


class TestMatmul(_TestBinaryFuncOpBase):
    case_name = "matmul"
    left = _ARITH_LEFT
    right = [[1.0, 0.0], [2.0, 1.0]]
    forward_op = lucid.matmul
    torch_forward = torch.matmul
    expected_forward = np.array([[5.0, 2.0], [11.0, 4.0]], dtype=np.float64)


class TestTensorDot(_TestBinaryFuncOpBase):
    case_name = "tensordot"
    left = _ARITH_LEFT
    right = _MATRIX_RIGHT
    forward_op = lucid.tensordot
    torch_forward = lambda a, b: torch.tensordot(a, b, dims=2)
    expected_forward = np.array(70.0)


class TestEqual(_TestBinaryNoGradFuncOpBase):
    case_name = "equal_no_grad"
    left = _CMP_LEFT
    right = _CMP_RIGHT
    forward_op = _bind_bfunc(bfunc._equal)
    torch_forward = torch.eq
    expected_forward = np.array([[False, True], [True, False]], dtype=bool)


class TestNotEqual(_TestBinaryNoGradFuncOpBase):
    case_name = "not_equal_no_grad"
    left = _CMP_LEFT
    right = _CMP_RIGHT
    forward_op = _bind_bfunc(bfunc._not_equal)
    torch_forward = torch.ne
    expected_forward = np.array([[True, False], [False, True]], dtype=bool)


class TestGreater(_TestBinaryNoGradFuncOpBase):
    case_name = "greater_no_grad"
    left = _CMP_LEFT
    right = _CMP_RIGHT
    forward_op = _bind_bfunc(bfunc._greater)
    torch_forward = torch.gt
    expected_forward = np.array([[False, False], [False, True]], dtype=bool)


class TestGreaterOrEqual(_TestBinaryNoGradFuncOpBase):
    case_name = "greater_equal_no_grad"
    left = _CMP_LEFT
    right = _CMP_RIGHT
    forward_op = _bind_bfunc(bfunc._greater_or_equal)
    torch_forward = torch.ge
    expected_forward = np.array([[False, True], [True, True]], dtype=bool)


class TestLess(_TestBinaryNoGradFuncOpBase):
    case_name = "less_no_grad"
    left = _CMP_LEFT
    right = _CMP_RIGHT
    forward_op = _bind_bfunc(bfunc._less)
    torch_forward = torch.lt
    expected_forward = np.array([[True, False], [False, False]], dtype=bool)


class TestLessOrEqual(_TestBinaryNoGradFuncOpBase):
    case_name = "less_equal_no_grad"
    left = _CMP_LEFT
    right = _CMP_RIGHT
    forward_op = _bind_bfunc(bfunc._less_or_equal)
    torch_forward = torch.le
    expected_forward = np.array([[True, True], [True, False]], dtype=bool)


class TestBitwiseAnd(_TestBinaryNoGradFuncOpBase):
    case_name = "bitwise_and_no_grad"
    left = _BITWISE_LEFT
    right = _BITWISE_RIGHT
    forward_op = _bind_bfunc(bfunc._bitwise_and)
    torch_forward = torch.bitwise_and
    dtype = lucid.Int64
    expected_forward = np.array([[0, 0], [0, 4]], dtype=np.int64)


class TestBitwiseOr(_TestBinaryNoGradFuncOpBase):
    case_name = "bitwise_or_no_grad"
    left = _BITWISE_LEFT
    right = _BITWISE_RIGHT
    forward_op = _bind_bfunc(bfunc._bitwise_or)
    torch_forward = torch.bitwise_or
    dtype = lucid.Int64
    expected_forward = np.array([[3, 3], [3, 7]], dtype=np.int64)
