from functools import partial
from types import ModuleType
import math
import numpy as np

from lucid._tensor import Tensor
from lucid._backend.core import (
    operation,
    binary_func_op,
    _FuncOpReturnType,
    _GradFuncType,
)
from lucid._backend.metal import mx


class add(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op()
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(a.data + b.data)
        return self.result, self.__grad__

    @binary_func_op(device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.add(a.data, b.data))
        return self.result, self.__grad__

    def __grad__(self) -> _GradFuncType:
        return self.result.grad, self.result.grad

    def __flops__(self, a: Tensor, b: Tensor) -> int:
        if a.shape != b.shape:
            max_dims = max(a.ndim, b.ndim)
            shape_a = (1,) * (max_dims - a.ndim) if a.ndim < max_dims else ()
            shape_b = (1,) * (max_dims - b.ndim) if b.ndim < max_dims else ()

            shape_a += a.shape
            shape_b += b.shape

            out_shape = tuple(max(da, db) for da, db in zip(shape_a, shape_b))
            return math.prod(out_shape)
        else:
            return a.size


class sub(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op()
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(a.data - b.data)
        return self.result, self.__grad__

    @binary_func_op(device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.subtract(a.data, b.data))
        return self.result, self.__grad__

    def __grad__(self) -> _GradFuncType:
        return self.result.grad, -self.result.grad


class multiply(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op()
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(a.data * b.data)
        return self.result, partial(self.__grad__, a=a, b=b)

    @binary_func_op(device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.multiply(a.data, b.data))
        return self.result, partial(self.__grad__, a=a, b=b)

    def __grad__(self, a: Tensor, b: Tensor) -> _GradFuncType:
        return b.data * self.result.grad, a.data * self.result.grad


class truediv(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op()
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(a.data / b.data)
        return self.result, partial(self.__grad__, a=a, b=b)

    @binary_func_op(device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.divide(a.data, b.data))
        return self.result, partial(self.__grad__, a=a, b=b)

    def __grad__(self, a: Tensor, b: Tensor) -> _GradFuncType:
        return (
            (1 / b.data) * self.result.grad,
            (-a.data / (b.data**2)) * self.result.grad,
        )


class _equal(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op(has_gradient=False)
    def cpu(self, a: Tensor, b: Tensor) -> Tensor:
        self.result = Tensor((a.data == b.data).astype(a.data.dtype))
        return self.result, partial(self.__grad__, lib=np)

    @binary_func_op(has_gradient=False, device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> Tensor:
        self.result = Tensor((a.data == b.data).astype(a.data.dtype))
        return self.result, partial(self.__grad__, lib=mx)

    def __grad__(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.array(0.0), lib_.array(0.0)


class _not_equal(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op(has_gradient=False)
    def cpu(self, a: Tensor, b: Tensor) -> Tensor:
        self.result = Tensor((a.data != b.data).astype(a.data.dtype))
        return self.result, partial(self.__grad__, lib=np)

    @binary_func_op(has_gradient=False, device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> Tensor:
        self.result = Tensor((a.data != b.data).astype(a.data.dtype))
        return self.result, partial(self.__grad__, lib=np)

    def __grad__(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.array(0.0), lib_.array(0.0)


class _greater(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op(has_gradient=False)
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor((a.data > b.data).astype(a.data.dtype))
        return self.result, partial(self.__grad__, lib=np)

    @binary_func_op(has_gradient=False, device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor((a.data > b.data).astype(a.data.dtype))
        return self.result, partial(self.__grad__, lib=np)

    def __grad__(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.array(0.0), lib_.array(0.0)


class _greater_or_equal(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op(has_gradient=False)
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor((a.data >= b.data).astype(a.data.dtype))
        return self.result, partial(self.__grad__, lib=np)

    @binary_func_op(has_gradient=False, device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor((a.data >= b.data).astype(a.data.dtype))
        return self.result, partial(self.__grad__, lib=np)

    def __grad__(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.array(0.0), lib_.array(0.0)


class _less(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op(has_gradient=False)
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor((a.data < b.data).astype(a.data.dtype))
        return self.result, partial(self.__grad__, lib=np)

    @binary_func_op(has_gradient=False, device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor((a.data < b.data).astype(a.data.dtype))
        return self.result, partial(self.__grad__, lib=np)

    def __grad__(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.array(0.0), lib_.array(0.0)


class _less_or_equal(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op(has_gradient=False)
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor((a.data <= b.data).astype(a.data.dtype))
        return self.result, partial(self.__grad__, lib=np)

    @binary_func_op(has_gradient=False, device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor((a.data <= b.data).astype(a.data.dtype))
        return self.result, partial(self.__grad__, lib=np)

    def __grad__(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.array(0.0), lib_.array(0.0)


class minimum(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op()
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.minimum(a.data, b.data))
        return self.result, partial(self.__grad__, a=a, b=b)

    @binary_func_op(device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.minimum(a.data, b.data))
        return self.result, partial(self.__grad__, a=a, b=b)

    def __grad__(self, a: Tensor, b: Tensor) -> _GradFuncType:
        a_grad = (a.data <= b.data).astype(a.data.dtype)
        b_grad = (a.data > b.data).astype(b.data.dtype)

        return a_grad * self.result.grad, b_grad * self.result.grad


class maximum(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op()
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.maximum(a.data, b.data))
        return self.result, partial(self.__grad__, a=a, b=b)

    @binary_func_op(device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.maximum(a.data, b.data))
        return self.result, partial(self.__grad__, a=a, b=b)

    def __grad__(self, a: Tensor, b: Tensor) -> _GradFuncType:
        a_grad = (a.data >= b.data).astype(a.data.dtype)
        b_grad = (a.data < b.data).astype(b.data.dtype)

        return a_grad * self.result.grad, b_grad * self.result.grad


class power(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op()
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.power(a.data, b.data))
        return self.result, partial(self.__grad__, a=a, b=b, lib_=np)

    @binary_func_op(device="gpu")
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.power(a.data, b.data))
        return self.result, partial(self.__grad__, a=a, b=b, lib_=mx)

    def __grad__(self, a: Tensor, b: Tensor, lib_: ModuleType) -> _GradFuncType:
        a_grad = b.data * lib_.power(a.data, b.data - 1)
        b_grad = lib_.power(a.data, b.data) * lib_.log(a.data)

        return a_grad * self.result.grad, b_grad * self.result.grad


class dot(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op()
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.dot(a.data, b.data))
        return self.result, partial(self.__grad_cpu__, a=a, b=b)

    @binary_func_op(device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        if a.ndim != 1 or b.ndum != 1:
            raise ValueError(f"Only 1D dot product is supported for Metal backend.")

        self.result = Tensor(mx.sum(a.data * b.data))
        return self.result, partial(self.__grad_gpu__, a=a, b=b)

    def __grad_cpu__(self, a: Tensor, b: Tensor) -> _GradFuncType:
        return self.result.grad.dot(b.data.mT), a.data.mT.dot(self.result.grad)

    def __grad_gpu__(self, a: Tensor, b: Tensor) -> _GradFuncType:
        return b.data * self.result.grad, a.data * self.result.grad


class inner(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op()
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.inner(a.data, b.data))
        return self.result, partial(self.__grad__, a=a, b=b, lib=np)

    @binary_func_op(device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.inner(a.data, b.data))
        return self.result, partial(self.__grad__, a=a, b=b, lib=mx)

    def __grad__(self, a: Tensor, b: Tensor, lib_: ModuleType) -> _GradFuncType:
        return (
            lib_.tensordot(self.result.grad, b.data, axes=([-1], [-1])),
            lib_.tensordot(a.data, self.result.grad, axes=([-1], [-1])),
        )


class outer(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op()
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.outer(a.data, b.data))
        return self.result, partial(self.__grad__, a=a, b=b, lib_=np)

    @binary_func_op(device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.outer(a.data, b.data))
        return self.result, partial(self.__grad__, a=a, b=b, lib_=mx)

    def __grad__(self, a: Tensor, b: Tensor, lib_: ModuleType) -> _GradFuncType:
        return (
            lib_.tensordot(self.result.grad, b.data, axes=([1], [0])),
            lib_.tensordot(self.result.grad, a.data, axes=([0], [0])),
        )


class matmul(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op()
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.matmul(a.data, b.data))
        return self.result, partial(self.__grad_cpu__, a=a, b=b)

    @binary_func_op(device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.matmul(a.data, b.data))
        return self.result, partial(self.__grad_gpu__, a=a, b=b)

    def __grad_cpu__(self, a: Tensor, b: Tensor) -> _GradFuncType:
        return (
            np.matmul(self.result.grad, b.data.mT),
            np.matmul(a.data.mT, self.result.grad),
        )

    def __grad_gpu__(self, a: Tensor, b: Tensor) -> _GradFuncType:
        return (
            mx.matmul(self.result.grad, mx.swapaxes(b.data, -1, -2)),
            mx.matmul(mx.swapaxes(a.data, -1, -2), self.result.grad),
        )
