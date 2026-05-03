"""
Lucid — Apple Silicon ML framework.
"""

from lucid.version import __version__, _EXPECTED_ABI
from lucid._C import engine as _C_engine

if _C_engine.ABI_VERSION != _EXPECTED_ABI:
    raise RuntimeError(
        f"lucid C++ engine ABI mismatch: expected {_EXPECTED_ABI}, "
        f"got {_C_engine.ABI_VERSION}. Rebuild the engine."
    )

from lucid._dtype import (
    dtype,
    float16,
    float32,
    float64,
    bfloat16,
    int8,
    int16,
    int32,
    int64,
    bool_,
    complex64,
    half,
    double,
    short,
    long,
)
from lucid._device import device
from lucid._globals import (
    set_default_dtype,
    get_default_dtype,
    set_default_device,
    get_default_device,
)

__all__ = [
    "__version__",
    "dtype",
    "float16", "float32", "float64", "bfloat16",
    "int8", "int16", "int32", "int64",
    "bool_", "complex64",
    "half", "double", "short", "long",
    "dtypes",
    "device",
    "set_default_dtype", "get_default_dtype",
    "set_default_device", "get_default_device",
    "Tensor",
    "tensor", "as_tensor", "from_numpy",
    "zeros", "ones", "empty", "full", "eye",
    "arange", "linspace",
    "zeros_like", "ones_like", "empty_like", "full_like",
    "rand", "randn", "randint", "bernoulli", "normal",
    "rand_like", "randn_like", "manual_seed",
    "add", "sub", "mul", "div", "matmul", "cat", "stack",
    "where", "sum", "mean", "max", "min",
    "no_grad", "enable_grad", "is_grad_enabled", "set_grad_enabled",
    "inference_mode",
    "save", "load",
    "nn", "optim", "autograd", "linalg", "metal", "backends", "utils",
]


def __getattr__(name: str) -> object:
    import sys as _sys
    if name == "Tensor":
        from lucid._tensor.tensor import Tensor
        return Tensor

    _factory_names = {
        "zeros", "ones", "empty", "full", "eye",
        "arange", "linspace",
        "zeros_like", "ones_like", "empty_like", "full_like",
        "tensor", "as_tensor", "from_numpy",
    }
    if name in _factory_names:
        import lucid._factories as _fac
        return getattr(_fac, name)

    _random_names = {
        "rand", "randn", "randint", "bernoulli", "normal",
        "rand_like", "randn_like", "manual_seed",
    }
    if name in _random_names:
        import lucid._factories as _fac
        return getattr(_fac, name)

    _ops_names = {
        "add", "sub", "mul", "div", "matmul", "cat", "stack",
        "where", "sum", "mean", "max", "min", "abs", "neg",
        "exp", "log", "sqrt", "relu", "sigmoid", "tanh",
        "softmax", "log_softmax", "reshape", "permute",
        "transpose", "unsqueeze", "squeeze", "flatten",
        "expand", "broadcast_to", "repeat", "tile",
        "concatenate", "hstack", "vstack", "split", "chunk",
        "unbind", "pad", "roll", "gather", "sort", "argsort",
        "nonzero", "unique", "topk", "meshgrid",
        "einsum", "inner", "outer", "tensordot", "dot",
        "argmax", "argmin", "cumsum", "cumprod", "prod",
        "std", "var", "trace", "any", "all",
        "equal", "not_equal", "greater", "greater_equal",
        "less", "less_equal", "maximum", "minimum",
        "tril", "triu", "masked_fill", "contiguous",
        "detach", "clone",
    }
    if name in _ops_names:
        import lucid._ops as _ops
        return getattr(_ops, name)

    if name == "nn":
        if "lucid.nn" in _sys.modules:
            return _sys.modules["lucid.nn"]
        import lucid.nn as _nn
        return _nn

    if name == "optim":
        if "lucid.optim" in _sys.modules:
            return _sys.modules["lucid.optim"]
        import lucid.optim as _optim
        return _optim

    if name == "autograd":
        if "lucid.autograd" in _sys.modules:
            return _sys.modules["lucid.autograd"]
        import lucid.autograd as _ag
        return _ag

    if name == "linalg":
        if "lucid.linalg" in _sys.modules:
            return _sys.modules["lucid.linalg"]
        import lucid.linalg as _la
        return _la

    if name == "metal":
        if "lucid.metal" in _sys.modules:
            return _sys.modules["lucid.metal"]
        import lucid.metal as _metal
        return _metal

    if name == "backends":
        if "lucid.backends" in _sys.modules:
            return _sys.modules["lucid.backends"]
        import lucid.backends as _backends
        return _backends

    _grad_names = {
        "no_grad", "enable_grad", "is_grad_enabled",
        "set_grad_enabled", "inference_mode",
    }
    if name in _grad_names:
        from lucid.autograd._grad_mode import (
            no_grad, enable_grad, is_grad_enabled,
            set_grad_enabled, inference_mode,
        )
        return {
            "no_grad": no_grad,
            "enable_grad": enable_grad,
            "is_grad_enabled": is_grad_enabled,
            "set_grad_enabled": set_grad_enabled,
            "inference_mode": inference_mode,
        }[name]

    if name in ("save", "load"):
        import lucid.serialization as _ser
        return getattr(_ser, name)

    if name == "utils":
        if "lucid.utils" in _sys.modules:
            return _sys.modules["lucid.utils"]
        import lucid.utils as _utils
        return _utils

    if name == "dtypes":
        import lucid.dtypes as _dtypes
        return _dtypes

    raise AttributeError(f"module 'lucid' has no attribute '{name}'")
