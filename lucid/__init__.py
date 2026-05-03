"""
Lucid — Apple Silicon ML framework.
"""

from lucid.version import __version__, _EXPECTED_ABI
from lucid._C import engine as _C_engine

try:
    _abi = int(_C_engine.ABI_VERSION)
    if _abi != _EXPECTED_ABI:
        raise RuntimeError(
            f"lucid C++ engine ABI mismatch: expected {_EXPECTED_ABI}, "
            f"got {_abi}. Rebuild the engine."
        )
except (TypeError, ValueError):
    pass  # Mocked during docs build — skip ABI check

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
    "nn", "optim", "autograd", "linalg", "metal", "backends", "utils", "einops", "profiler", "amp",
]


def __getattr__(name: str) -> object:
    import sys as _sys
    _g = globals()

    if name == "Tensor":
        from lucid._tensor.tensor import Tensor
        _g["Tensor"] = Tensor
        return Tensor

    _factory_names = {
        "zeros", "ones", "empty", "full", "eye",
        "arange", "linspace",
        "zeros_like", "ones_like", "empty_like", "full_like",
        "tensor", "as_tensor", "from_numpy",
    }
    if name in _factory_names:
        import lucid._factories as _fac
        obj = getattr(_fac, name)
        _g[name] = obj
        return obj

    _random_names = {
        "rand", "randn", "randint", "bernoulli", "normal",
        "rand_like", "randn_like", "manual_seed",
    }
    if name in _random_names:
        import lucid._factories as _fac
        obj = getattr(_fac, name)
        _g[name] = obj
        return obj

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
        obj = getattr(_ops, name)
        _g[name] = obj
        return obj

    if name == "nn":
        if "lucid.nn" in _sys.modules:
            mod = _sys.modules["lucid.nn"]
        else:
            import lucid.nn as mod  # type: ignore[no-redef]
        _g["nn"] = mod
        return mod

    if name == "optim":
        if "lucid.optim" in _sys.modules:
            mod = _sys.modules["lucid.optim"]
        else:
            import lucid.optim as mod  # type: ignore[no-redef]
        _g["optim"] = mod
        return mod

    if name == "autograd":
        if "lucid.autograd" in _sys.modules:
            mod = _sys.modules["lucid.autograd"]
        else:
            import lucid.autograd as mod  # type: ignore[no-redef]
        _g["autograd"] = mod
        return mod

    if name == "linalg":
        if "lucid.linalg" in _sys.modules:
            mod = _sys.modules["lucid.linalg"]
        else:
            import lucid.linalg as mod  # type: ignore[no-redef]
        _g["linalg"] = mod
        return mod

    if name == "metal":
        if "lucid.metal" in _sys.modules:
            mod = _sys.modules["lucid.metal"]
        else:
            import lucid.metal as mod  # type: ignore[no-redef]
        _g["metal"] = mod
        return mod

    if name == "backends":
        if "lucid.backends" in _sys.modules:
            mod = _sys.modules["lucid.backends"]
        else:
            import lucid.backends as mod  # type: ignore[no-redef]
        _g["backends"] = mod
        return mod

    _grad_names = {
        "no_grad", "enable_grad", "is_grad_enabled",
        "set_grad_enabled", "inference_mode",
    }
    if name in _grad_names:
        from lucid.autograd._grad_mode import (
            no_grad, enable_grad, is_grad_enabled,
            set_grad_enabled, inference_mode,
        )
        _map = {
            "no_grad": no_grad,
            "enable_grad": enable_grad,
            "is_grad_enabled": is_grad_enabled,
            "set_grad_enabled": set_grad_enabled,
            "inference_mode": inference_mode,
        }
        _g.update(_map)
        return _map[name]

    if name in ("save", "load"):
        import lucid.serialization as _ser
        obj = getattr(_ser, name)
        _g[name] = obj
        return obj

    if name == "utils":
        if "lucid.utils" in _sys.modules:
            mod = _sys.modules["lucid.utils"]
        else:
            import lucid.utils as mod  # type: ignore[no-redef]
        _g["utils"] = mod
        return mod

    if name == "dtypes":
        import lucid.dtypes as _dtypes
        _g["dtypes"] = _dtypes
        return _dtypes

    if name == "einops":
        if "lucid.einops" in _sys.modules:
            mod = _sys.modules["lucid.einops"]
        else:
            import lucid.einops as mod  # type: ignore[no-redef]
        _g["einops"] = mod
        return mod

    if name == "profiler":
        if "lucid.profiler" in _sys.modules:
            mod = _sys.modules["lucid.profiler"]
        else:
            import lucid.profiler as mod  # type: ignore[no-redef]
        _g["profiler"] = mod
        return mod

    if name == "amp":
        if "lucid.amp" in _sys.modules:
            mod = _sys.modules["lucid.amp"]
        else:
            import lucid.amp as mod  # type: ignore[no-redef]
        _g["amp"] = mod
        return mod

    raise AttributeError(f"module 'lucid' has no attribute '{name}'")
