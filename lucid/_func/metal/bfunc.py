from lucid._tensor import Tensor

from lucid._backend.core import create_bfunc_op, _FuncOpReturnType
from lucid._backend.metal import mx, _MLXArray


@create_bfunc_op(device="gpu")
def _add(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.add(self.data, other.data), device="gpu")

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return result.grad, result.grad

    return result, compute_grad
