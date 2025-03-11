from lucid._tensor import Tensor

from lucid._backend.core import create_bfunc_op, _FuncOpReturnType
from lucid._backend.metal import mx, _MLXArray


@create_bfunc_op(device="gpu")
def _add(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.add(self.data, other.data), device="gpu")

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return result.grad, result.grad

    return result, compute_grad


@create_bfunc_op(device="gpu")
def _sub(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.subtract(self.data, other.data), device="gpu")

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return result.grad, -result.grad

    return result, compute_grad


@create_bfunc_op(device="gpu")
def _mul(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.multiply(self.data, other.data), device="gpu")

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return other.data * result.grad, self.data * result.grad

    return result, compute_grad


@create_bfunc_op(device="gpu")
def _truediv(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.divide(self.data, other.data), device="gpu")

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return (
            (1 / other.data) * result.grad,
            (-self.data / (other.data**2)) * result.grad,
        )

    return result, compute_grad
