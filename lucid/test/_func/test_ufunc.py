from typing import Any, Callable, ClassVar, Final

import numpy as np
import torch

import lucid
from lucid.test.core import TensorOpTorchBase, TensorOpTorchCase


_UNARY_FLOAT_2x2: Final[list[list[float]]] = [[1.0, 2.0], [3.0, 4.0]]
_UNARY_INT_2x2: Final[list[list[int]]] = [[1, 2], [3, 4]]
_UNARY_NEG_2x2: Final[list[list[float]]] = [[-1.0, -2.0], [3.0, -4.0]]

_UNARY_ARC_RANGE: Final[list[list[float]]] = [[-0.75, -0.25], [0.25, 0.75]]
_UNARY_LOG_RANGE: Final[list[list[float]]] = [[1.0, 2.0], [3.0, 4.0]]
_UNARY_CUM_2x3: Final[list[list[float]]] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

_UNARY_RECT_2x3: Final[list[list[float]]] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
_UNARY_DECIMALS: Final[list[list[float]]] = [[1.14, 2.73], [3.36, 4.99]]
_UNARY_CLIP_1B: Final[list[list[float]]] = [[-1.0, 0.5], [1.5, 2.5]]


class _TestUnaryFuncOpBase(TensorOpTorchBase):
    case_name: ClassVar[str]
    input: ClassVar[Any]

    forward_op: ClassVar[Callable[[Any], Any]]
    torch_forward: ClassVar[Callable[[Any], Any]]
    expected_forward: ClassVar[Any]

    requires_grad: ClassVar[bool] = True
    dtype: ClassVar[Any | None] = None
    compare_grad_with_torch: ClassVar[bool] = True
    check_finite: ClassVar[bool] = True
    expected_input_grads: ClassVar[tuple[object | None, ...] | None] = None
    rtol: ClassVar[float] = 1e-7
    atol: ClassVar[float] = 1e-8

    def tensor_op_devices(self) -> tuple[str, ...]:
        return ("cpu", "gpu")

    def _build_unary_inputs(
        self,
        input_data: Any,
        *,
        dtype: Any | None = None,
        requires_grad: bool = True,
    ) -> tuple[Any]:
        if dtype is None:
            return (self.tensor(input_data, requires_grad=requires_grad),)
        return (
            self.tensor(
                input_data,
                dtype=dtype,
                requires_grad=requires_grad,
            ),
        )

    def _build_case(self) -> TensorOpTorchCase:
        forward_op = vars(type(self)).get("forward_op")
        torch_forward = vars(type(self)).get("torch_forward")

        if isinstance(forward_op, staticmethod):
            forward_op = forward_op.__func__
        if isinstance(torch_forward, staticmethod):
            torch_forward = torch_forward.__func__

        return TensorOpTorchCase(
            name=self.case_name,
            build_inputs=lambda _case: self._build_unary_inputs(
                self.input,
                dtype=self.dtype,
                requires_grad=self.requires_grad,
            ),
            forward_op=forward_op,
            expected_forward=self.expected_forward,
            expected_input_grads=self.expected_input_grads,
            requires_grad=(self.requires_grad,),
            check_finite=self.check_finite,
            torch_forward=torch_forward,
            compare_grad_with_torch=self.compare_grad_with_torch,
            rtol=self.rtol,
            atol=self.atol,
        )

    def tensor_op_cases(self) -> tuple[TensorOpTorchCase, ...]:
        return (self._build_case(),)


class _TestUnaryNoGradFuncOpBase(_TestUnaryFuncOpBase):
    requires_grad: ClassVar[bool] = False
    compare_grad_with_torch: ClassVar[bool] = False


class TestPow(_TestUnaryFuncOpBase):
    case_name = "pow"
    input = _UNARY_FLOAT_2x2
    forward_op = staticmethod(lambda a: a**2.0)
    torch_forward = staticmethod(lambda a: torch.pow(a, 2.0))
    expected_forward = np.array([[1.0, 4.0], [9.0, 16.0]], dtype=np.float64)


class TestRpow(_TestUnaryFuncOpBase):
    case_name = "rpow"
    input = _UNARY_FLOAT_2x2
    forward_op = staticmethod(lambda a: 2.0**a)
    torch_forward = staticmethod(lambda a: torch.pow(2.0, a))
    expected_forward = np.array([[2.0, 4.0], [8.0, 16.0]], dtype=np.float64)


class TestNeg(_TestUnaryFuncOpBase):
    case_name = "neg"
    input = _UNARY_FLOAT_2x2
    forward_op = staticmethod(lambda a: -a)
    torch_forward = staticmethod(lambda a: -a)
    expected_forward = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float64)


class TestInvert(_TestUnaryNoGradFuncOpBase):
    case_name = "invert_no_grad"
    input = _UNARY_INT_2x2
    forward_op = staticmethod(lambda a: ~a)
    torch_forward = staticmethod(torch.bitwise_not)
    dtype = lucid.Int64
    expected_forward = np.array([[-2, -3], [-4, -5]], dtype=np.int64)


class TestExp(_TestUnaryFuncOpBase):
    case_name = "exp"
    input = _UNARY_FLOAT_2x2
    forward_op = staticmethod(lucid.exp)
    torch_forward = staticmethod(torch.exp)
    expected_forward = np.array(np.exp(_UNARY_FLOAT_2x2), dtype=np.float64)


class TestLog(_TestUnaryFuncOpBase):
    case_name = "log"
    input = _UNARY_LOG_RANGE
    forward_op = staticmethod(lucid.log)
    torch_forward = staticmethod(torch.log)
    expected_forward = np.array(np.log(_UNARY_LOG_RANGE), dtype=np.float64)


class TestLog2(_TestUnaryFuncOpBase):
    case_name = "log2"
    input = _UNARY_LOG_RANGE
    forward_op = staticmethod(lucid.log2)
    torch_forward = staticmethod(torch.log2)
    expected_forward = np.array(np.log2(_UNARY_LOG_RANGE), dtype=np.float64)


class TestSqrt(_TestUnaryFuncOpBase):
    case_name = "sqrt"
    input = _UNARY_LOG_RANGE
    forward_op = staticmethod(lucid.sqrt)
    torch_forward = staticmethod(torch.sqrt)
    expected_forward = np.array(np.sqrt(_UNARY_LOG_RANGE), dtype=np.float64)


class TestSin(_TestUnaryFuncOpBase):
    case_name = "sin"
    input = _UNARY_FLOAT_2x2
    forward_op = staticmethod(lucid.sin)
    torch_forward = staticmethod(torch.sin)
    expected_forward = np.array(np.sin(_UNARY_FLOAT_2x2), dtype=np.float64)


class TestCos(_TestUnaryFuncOpBase):
    case_name = "cos"
    input = _UNARY_FLOAT_2x2
    forward_op = staticmethod(lucid.cos)
    torch_forward = staticmethod(torch.cos)
    expected_forward = np.array(np.cos(_UNARY_FLOAT_2x2), dtype=np.float64)


class TestTan(_TestUnaryFuncOpBase):
    case_name = "tan"
    input = _UNARY_FLOAT_2x2
    forward_op = staticmethod(lucid.tan)
    torch_forward = staticmethod(torch.tan)
    expected_forward = np.array(np.tan(_UNARY_FLOAT_2x2), dtype=np.float64)
    rtol = 1e-6
    atol = 1e-6


class TestArcsin(_TestUnaryFuncOpBase):
    case_name = "arcsin"
    input = _UNARY_ARC_RANGE
    forward_op = staticmethod(lucid.arcsin)
    torch_forward = staticmethod(torch.arcsin)
    expected_forward = np.array(np.arcsin(_UNARY_ARC_RANGE), dtype=np.float64)
    rtol = 1e-6
    atol = 1e-6


class TestArccos(_TestUnaryFuncOpBase):
    case_name = "arccos"
    input = _UNARY_ARC_RANGE
    forward_op = staticmethod(lucid.arccos)
    torch_forward = staticmethod(torch.arccos)
    expected_forward = np.array(np.arccos(_UNARY_ARC_RANGE), dtype=np.float64)
    rtol = 1e-6
    atol = 1e-6


class TestArctan(_TestUnaryFuncOpBase):
    case_name = "arctan"
    input = _UNARY_FLOAT_2x2
    forward_op = staticmethod(lucid.arctan)
    torch_forward = staticmethod(torch.atan)
    expected_forward = np.array(np.arctan(_UNARY_FLOAT_2x2), dtype=np.float64)


class TestSinh(_TestUnaryFuncOpBase):
    case_name = "sinh"
    input = _UNARY_FLOAT_2x2
    forward_op = staticmethod(lucid.sinh)
    torch_forward = staticmethod(torch.sinh)
    expected_forward = np.array(np.sinh(_UNARY_FLOAT_2x2), dtype=np.float64)
    dtype = lucid.Float64


class TestCosh(_TestUnaryFuncOpBase):
    case_name = "cosh"
    input = _UNARY_FLOAT_2x2
    forward_op = staticmethod(lucid.cosh)
    torch_forward = staticmethod(torch.cosh)
    expected_forward = np.array(np.cosh(_UNARY_FLOAT_2x2), dtype=np.float64)
    dtype = lucid.Float64


class TestTanh(_TestUnaryFuncOpBase):
    case_name = "tanh"
    input = _UNARY_FLOAT_2x2
    forward_op = staticmethod(lucid.tanh)
    torch_forward = staticmethod(torch.tanh)
    expected_forward = np.array(np.tanh(_UNARY_FLOAT_2x2), dtype=np.float64)
    rtol = 1e-6
    atol = 1e-6


class TestClip(_TestUnaryFuncOpBase):
    input: ClassVar[Any] = _UNARY_CLIP_1B
    __test__ = False

    class TestDefault(_TestUnaryFuncOpBase):
        case_name = "clip"
        input = _UNARY_CLIP_1B
        forward_op = staticmethod(lambda a: lucid.clip(a))
        torch_forward = staticmethod(
            lambda a: torch.clamp(a, min=torch.min(a), max=torch.max(a))
        )
        expected_forward = np.array(_UNARY_CLIP_1B, dtype=np.float64)

    class TestBetween(_TestUnaryFuncOpBase):
        case_name = "clip_min_max"
        input = _UNARY_CLIP_1B
        forward_op = staticmethod(
            lambda a: lucid.clip(a, min_value=-0.25, max_value=1.25)
        )
        torch_forward = staticmethod(lambda a: torch.clamp(a, min=-0.25, max=1.25))
        expected_forward = np.array([[-0.25, -0.25], [0.5, 1.25]], dtype=np.float64)

    class TestMinOnly(_TestUnaryFuncOpBase):
        case_name = "clip_min_only"
        input = _UNARY_CLIP_1B
        forward_op = staticmethod(lambda a: lucid.clip(a, min_value=-0.2))
        torch_forward = staticmethod(lambda a: torch.clamp(a, min=-0.2))
        expected_forward = np.array([[-0.2, 0.5], [1.5, 2.5]], dtype=np.float64)

    class TestMaxOnly(_TestUnaryFuncOpBase):
        case_name = "clip_max_only"
        input = _UNARY_CLIP_1B
        forward_op = staticmethod(lambda a: lucid.clip(a, max_value=1.8))
        torch_forward = staticmethod(lambda a: torch.clamp(a, max=1.8))
        expected_forward = np.array([[-1.0, 0.5], [1.5, 1.8]], dtype=np.float64)


class TestAbs(_TestUnaryFuncOpBase):
    case_name = "abs"
    input = _UNARY_NEG_2x2
    forward_op = staticmethod(lucid.abs)
    torch_forward = staticmethod(torch.abs)
    expected_forward = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)


class TestSign(_TestUnaryNoGradFuncOpBase):
    case_name = "sign_no_grad"
    input = _UNARY_NEG_2x2
    forward_op = staticmethod(lucid.sign)
    torch_forward = staticmethod(torch.sign)
    expected_forward = np.array([[-1.0, -1.0], [1.0, -1.0]], dtype=np.float64)


class TestReciprocal(_TestUnaryFuncOpBase):
    case_name = "reciprocal"
    input = [[1.0, 2.0], [4.0, 8.0]]
    forward_op = staticmethod(lucid.reciprocal)
    torch_forward = staticmethod(torch.reciprocal)
    expected_forward = np.array([[1.0, 0.5], [0.25, 0.125]], dtype=np.float64)


class TestSquare(_TestUnaryFuncOpBase):
    case_name = "square"
    input = _UNARY_FLOAT_2x2
    forward_op = staticmethod(lucid.square)
    torch_forward = staticmethod(torch.square)
    expected_forward = np.array([[1.0, 4.0], [9.0, 16.0]], dtype=np.float64)


class TestCube(_TestUnaryFuncOpBase):
    case_name = "cube"
    input = _UNARY_FLOAT_2x2
    forward_op = staticmethod(lucid.cube)
    torch_forward = staticmethod(lambda a: torch.pow(a, 3))
    expected_forward = np.array([[1.0, 8.0], [27.0, 64.0]], dtype=np.float64)


class TestTranspose(_TestUnaryFuncOpBase):
    case_name = "transpose"
    input = _UNARY_FLOAT_2x2
    forward_op = staticmethod(lambda a: lucid.transpose(a, axes=[1, 0]))
    torch_forward = staticmethod(lambda a: torch.permute(a, (1, 0)))
    expected_forward = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float64)


class TestT(_TestUnaryFuncOpBase):
    case_name = "T"
    input = _UNARY_FLOAT_2x2
    forward_op = staticmethod(lambda a: a.T)
    torch_forward = staticmethod(lambda a: torch.t(a))
    expected_forward = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float64)


class TestMT(_TestUnaryFuncOpBase):
    case_name = "mT"
    input = _UNARY_FLOAT_2x2
    forward_op = staticmethod(lambda a: a.mT)
    torch_forward = staticmethod(lambda a: a.mT)
    expected_forward = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float64)


class TestTrace(_TestUnaryFuncOpBase):
    case_name = "trace"
    input = _UNARY_FLOAT_2x2
    forward_op = staticmethod(lucid.trace)
    torch_forward = staticmethod(torch.trace)
    expected_forward = np.array(5.0)


class TestSum(_TestUnaryFuncOpBase):
    input: ClassVar[Any] = _UNARY_FLOAT_2x2
    __test__ = False

    class TestDefault(_TestUnaryFuncOpBase):
        case_name = "sum"
        input = _UNARY_FLOAT_2x2
        forward_op = staticmethod(lambda a: lucid.sum(a))
        torch_forward = staticmethod(lambda a: torch.sum(a))
        expected_forward = np.array(10.0)

    class TestAxisKeepDims(_TestUnaryFuncOpBase):
        case_name = "sum_axis_keepdims"
        input = _UNARY_FLOAT_2x2
        forward_op = staticmethod(lambda a: lucid.sum(a, axis=1, keepdims=True))
        torch_forward = staticmethod(lambda a: torch.sum(a, dim=1, keepdim=True))
        expected_forward = np.array([[3.0], [7.0]], dtype=np.float64)

    class TestAxisKeepDimsFalse(_TestUnaryFuncOpBase):
        case_name = "sum_axis_keepdims_false"
        input = _UNARY_FLOAT_2x2
        forward_op = staticmethod(lambda a: lucid.sum(a, axis=1, keepdims=False))
        torch_forward = staticmethod(lambda a: torch.sum(a, dim=1, keepdim=False))
        expected_forward = np.array([3.0, 7.0], dtype=np.float64)

    class TestAxisNegativeKeepDimsFalse(_TestUnaryFuncOpBase):
        case_name = "sum_axis_negative_keepdims_false"
        input = _UNARY_CUM_2x3
        forward_op = staticmethod(lambda a: lucid.sum(a, axis=-1, keepdims=False))
        torch_forward = staticmethod(lambda a: torch.sum(a, dim=-1, keepdim=False))
        expected_forward = np.array([6.0, 15.0], dtype=np.float64)


class TestMean(_TestUnaryFuncOpBase):
    input: ClassVar[Any] = _UNARY_FLOAT_2x2
    __test__ = False

    class TestDefault(_TestUnaryFuncOpBase):
        case_name = "mean"
        input = _UNARY_FLOAT_2x2
        forward_op = staticmethod(lambda a: lucid.mean(a))
        torch_forward = staticmethod(torch.mean)
        expected_forward = np.array(2.5)

    class TestAxis(_TestUnaryFuncOpBase):
        case_name = "mean_axis"
        input = _UNARY_FLOAT_2x2
        forward_op = staticmethod(lambda a: lucid.mean(a, axis=1, keepdims=False))
        torch_forward = staticmethod(lambda a: torch.mean(a, dim=1))
        expected_forward = np.array([1.5, 3.5], dtype=np.float64)

    class TestAxisKeepDims(_TestUnaryFuncOpBase):
        case_name = "mean_axis_keepdims"
        input = _UNARY_FLOAT_2x2
        forward_op = staticmethod(lambda a: lucid.mean(a, axis=1, keepdims=True))
        torch_forward = staticmethod(lambda a: torch.mean(a, dim=1, keepdim=True))
        expected_forward = np.array([[1.5], [3.5]], dtype=np.float64)

    class TestAxisNegative(_TestUnaryFuncOpBase):
        case_name = "mean_axis_negative"
        input = _UNARY_CUM_2x3
        forward_op = staticmethod(lambda a: lucid.mean(a, axis=-1, keepdims=False))
        torch_forward = staticmethod(lambda a: torch.mean(a, dim=-1, keepdim=False))
        expected_forward = np.array([2.0, 5.0], dtype=np.float64)


class TestVar(_TestUnaryFuncOpBase):
    input: ClassVar[Any] = _UNARY_FLOAT_2x2
    __test__ = False

    class TestDefault(_TestUnaryFuncOpBase):
        case_name = "var"
        input = _UNARY_FLOAT_2x2
        forward_op = staticmethod(lambda a: lucid.var(a))
        torch_forward = staticmethod(lambda a: torch.var(a, unbiased=False))
        expected_forward = np.array(1.25, dtype=np.float64)

    class TestAxis(_TestUnaryFuncOpBase):
        case_name = "var_axis"
        input = _UNARY_FLOAT_2x2
        forward_op = staticmethod(lambda a: lucid.var(a, axis=0, keepdims=False))
        torch_forward = staticmethod(lambda a: torch.var(a, dim=0, unbiased=False))
        expected_forward = np.array([1.0, 1.0], dtype=np.float64)

    class TestAxisKeepDims(_TestUnaryFuncOpBase):
        case_name = "var_axis_keepdims"
        input = _UNARY_FLOAT_2x2
        forward_op = staticmethod(lambda a: lucid.var(a, axis=1, keepdims=True))
        torch_forward = staticmethod(
            lambda a: torch.var(a, dim=1, unbiased=False, keepdim=True)
        )
        expected_forward = np.array([[0.25], [0.25]], dtype=np.float64)

    class TestAxisNegative(_TestUnaryFuncOpBase):
        case_name = "var_axis_negative"
        input = _UNARY_CUM_2x3
        forward_op = staticmethod(lambda a: lucid.var(a, axis=-1, keepdims=False))
        torch_forward = staticmethod(
            lambda a: torch.var(a, dim=-1, unbiased=False, keepdim=False)
        )
        expected_forward = np.array([2 / 3, 2 / 3], dtype=np.float64)


class TestMin(_TestUnaryFuncOpBase):
    input: ClassVar[Any] = _UNARY_FLOAT_2x2
    __test__ = False

    class TestDefault(_TestUnaryFuncOpBase):
        case_name = "min"
        input = _UNARY_FLOAT_2x2
        forward_op = staticmethod(lucid.min)
        torch_forward = staticmethod(torch.amin)
        expected_forward = np.array(1.0)

    class TestAxis(_TestUnaryFuncOpBase):
        case_name = "min_axis"
        input = _UNARY_FLOAT_2x2
        forward_op = staticmethod(lambda a: lucid.min(a, axis=1, keepdims=True))
        torch_forward = staticmethod(lambda a: torch.amin(a, dim=1, keepdim=True))
        expected_forward = np.array([[1.0], [3.0]], dtype=np.float64)

    class TestAxisKeepDimsFalse(_TestUnaryFuncOpBase):
        case_name = "min_axis_keepdims_false"
        input = _UNARY_FLOAT_2x2
        forward_op = staticmethod(lambda a: lucid.min(a, axis=0, keepdims=False))
        torch_forward = staticmethod(lambda a: torch.amin(a, dim=0, keepdim=False))
        expected_forward = np.array([1.0, 2.0], dtype=np.float64)

    class TestAxisNegative(_TestUnaryFuncOpBase):
        case_name = "min_axis_negative"
        input = _UNARY_CUM_2x3
        forward_op = staticmethod(lambda a: lucid.min(a, axis=-1, keepdims=False))
        torch_forward = staticmethod(lambda a: torch.amin(a, dim=-1, keepdim=False))
        expected_forward = np.array([1.0, 4.0], dtype=np.float64)


class TestMax(_TestUnaryFuncOpBase):
    input: ClassVar[Any] = _UNARY_FLOAT_2x2
    __test__ = False

    class TestDefault(_TestUnaryFuncOpBase):
        case_name = "max"
        input = _UNARY_FLOAT_2x2
        forward_op = staticmethod(lucid.max)
        torch_forward = staticmethod(torch.amax)
        expected_forward = np.array(4.0)

    class TestAxis(_TestUnaryFuncOpBase):
        case_name = "max_axis"
        input = _UNARY_FLOAT_2x2
        forward_op = staticmethod(lambda a: lucid.max(a, axis=1, keepdims=False))
        torch_forward = staticmethod(lambda a: torch.amax(a, dim=1))
        expected_forward = np.array([2.0, 4.0], dtype=np.float64)

    class TestAxisKeepDims(_TestUnaryFuncOpBase):
        case_name = "max_axis_keepdims"
        input = _UNARY_FLOAT_2x2
        forward_op = staticmethod(lambda a: lucid.max(a, axis=0, keepdims=True))
        torch_forward = staticmethod(lambda a: torch.amax(a, dim=0, keepdim=True))
        expected_forward = np.array([[3.0, 4.0]], dtype=np.float64)

    class TestAxisNegative(_TestUnaryFuncOpBase):
        case_name = "max_axis_negative"
        input = _UNARY_CUM_2x3
        forward_op = staticmethod(lambda a: lucid.max(a, axis=-1, keepdims=False))
        torch_forward = staticmethod(lambda a: torch.amax(a, dim=-1, keepdim=False))
        expected_forward = np.array([3.0, 6.0], dtype=np.float64)


class TestSwapaxes(_TestUnaryFuncOpBase):
    case_name = "swapaxes"
    input = _UNARY_FLOAT_2x2
    forward_op = staticmethod(lambda a: lucid.swapaxes(a, axis1=0, axis2=1))
    torch_forward = staticmethod(lambda a: torch.swapaxes(a, 0, 1))
    expected_forward = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float64)


class TestTransposeDefaultAxes(_TestUnaryFuncOpBase):
    case_name = "transpose_default_axes"
    input = _UNARY_RECT_2x3
    forward_op = staticmethod(lambda a: lucid.transpose(a, axes=None))
    torch_forward = staticmethod(lambda a: torch.permute(a, (1, 0)))
    expected_forward = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float64)


class TestRound(_TestUnaryNoGradFuncOpBase):
    input: ClassVar[Any] = _UNARY_DECIMALS
    __test__ = False

    class TestDefault(_TestUnaryNoGradFuncOpBase):
        case_name = "round_no_grad"
        input = [[1.2, 2.7], [3.9, 4.0]]
        forward_op = staticmethod(lambda a: lucid.round(a, decimals=0))
        torch_forward = staticmethod(lambda a: torch.round(a))
        expected_forward = np.array([[1.0, 3.0], [4.0, 4.0]], dtype=np.float64)

    class TestDecimals(_TestUnaryNoGradFuncOpBase):
        case_name = "round_decimals_no_grad"
        input = _UNARY_DECIMALS
        forward_op = staticmethod(lambda a: lucid.round(a, decimals=1))
        torch_forward = staticmethod(lambda a: torch.round(a, decimals=1))
        expected_forward = np.array([[1.1, 2.7], [3.4, 5.0]], dtype=np.float64)


class TestFloor(_TestUnaryNoGradFuncOpBase):
    case_name = "floor_no_grad"
    input = [[1.2, 2.7], [3.9, 4.1]]
    forward_op = staticmethod(lucid.floor)
    torch_forward = staticmethod(torch.floor)
    expected_forward = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)


class TestCeil(_TestUnaryNoGradFuncOpBase):
    case_name = "ceil_no_grad"
    input = [[1.2, 2.7], [3.1, 4.0]]
    forward_op = staticmethod(lucid.ceil)
    torch_forward = staticmethod(torch.ceil)
    expected_forward = np.array([[2.0, 3.0], [4.0, 4.0]], dtype=np.float64)


class TestCumprod(_TestUnaryFuncOpBase):
    input: ClassVar[Any] = _UNARY_CUM_2x3
    __test__ = False

    class TestDefault(_TestUnaryFuncOpBase):
        case_name = "cumprod"
        input = _UNARY_CUM_2x3
        forward_op = staticmethod(lambda a: lucid.cumprod(a, axis=1))
        torch_forward = staticmethod(lambda a: torch.cumprod(a, dim=1))
        expected_forward = np.array(
            [[1.0, 2.0, 6.0], [4.0, 20.0, 120.0]], dtype=np.float64
        )

    class TestDefaultAxis(_TestUnaryFuncOpBase):
        case_name = "cumprod_default_axis"
        input = _UNARY_CUM_2x3
        forward_op = staticmethod(lucid.cumprod)
        torch_forward = staticmethod(lambda a: torch.cumprod(a, dim=-1))
        expected_forward = np.array(
            [[1.0, 2.0, 6.0], [4.0, 20.0, 120.0]], dtype=np.float64
        )

    class TestAxisZero(_TestUnaryFuncOpBase):
        case_name = "cumprod_axis_zero"
        input = _UNARY_CUM_2x3
        forward_op = staticmethod(lambda a: lucid.cumprod(a, axis=0))
        torch_forward = staticmethod(lambda a: torch.cumprod(a, dim=0))
        expected_forward = np.array(
            [[1.0, 2.0, 3.0], [4.0, 10.0, 18.0]], dtype=np.float64
        )

    class TestNegativeAxis(_TestUnaryFuncOpBase):
        case_name = "cumprod_negative_axis"
        input = _UNARY_CUM_2x3
        forward_op = staticmethod(lambda a: lucid.cumprod(a, axis=-1))
        torch_forward = staticmethod(lambda a: torch.cumprod(a, dim=-1))
        expected_forward = np.array(
            [[1.0, 2.0, 6.0], [4.0, 20.0, 120.0]], dtype=np.float64
        )


class TestCumsum(_TestUnaryFuncOpBase):
    input: ClassVar[Any] = _UNARY_CUM_2x3
    __test__ = False

    class TestDefault(_TestUnaryFuncOpBase):
        case_name = "cumsum"
        input = _UNARY_CUM_2x3
        forward_op = staticmethod(lambda a: lucid.cumsum(a, axis=1))
        torch_forward = staticmethod(lambda a: torch.cumsum(a, dim=1))
        expected_forward = np.array(
            [[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]], dtype=np.float64
        )

    class TestDefaultAxis(_TestUnaryFuncOpBase):
        case_name = "cumsum_default_axis"
        input = _UNARY_CUM_2x3
        forward_op = staticmethod(lucid.cumsum)
        torch_forward = staticmethod(lambda a: torch.cumsum(a, dim=-1))
        expected_forward = np.array(
            [[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]], dtype=np.float64
        )

    class TestAxisZero(_TestUnaryFuncOpBase):
        case_name = "cumsum_axis_zero"
        input = _UNARY_CUM_2x3
        forward_op = staticmethod(lambda a: lucid.cumsum(a, axis=0))
        torch_forward = staticmethod(lambda a: torch.cumsum(a, dim=0))
        expected_forward = np.array(
            [[1.0, 2.0, 3.0], [5.0, 7.0, 9.0]], dtype=np.float64
        )

    class TestNegativeAxis(_TestUnaryFuncOpBase):
        case_name = "cumsum_negative_axis"
        input = _UNARY_CUM_2x3
        forward_op = staticmethod(lambda a: lucid.cumsum(a, axis=-1))
        torch_forward = staticmethod(lambda a: torch.cumsum(a, dim=-1))
        expected_forward = np.array(
            [[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]], dtype=np.float64
        )
