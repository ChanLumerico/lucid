"""
Lucid — Apple Silicon ML framework.
"""

# ── Save Python builtins before dtype aliases shadow them ────────────────────
_py_int = int
_py_float = float
_py_bool = bool

from lucid.version import __version__, _EXPECTED_ABI
from lucid._C import engine as _C_engine

try:
    _abi = _py_int(_C_engine.ABI_VERSION)
    if _abi != _EXPECTED_ABI:
        raise RuntimeError(
            f"lucid C++ engine ABI mismatch: expected {_EXPECTED_ABI}, "
            f"got {_abi}. Rebuild the engine."
        )
except (TypeError, ValueError):
    pass  # Mocked during docs build — skip ABI check

# fmt: off
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
    # Existing aliases kept for backward compat
    half,
    double,
    short,
    long,
    # New PyTorch-style module-level aliases
    # lucid.float / lucid.int / lucid.bool shadow Python builtins
    # only as module attributes — no impact on user code
    float32 as float,  # lucid.float  == torch.float  == float32
    int32 as int,  # lucid.int    == torch.int    == int32
    bool_ as bool,  # lucid.bool   == torch.bool   == bool_
)
from lucid._device import device
from lucid._globals import (
    set_default_dtype,
    get_default_dtype,
    set_default_device,
    get_default_device,
)

# ── Public API ────────────────────────────────────────────────────────────────
# Organised by category — mirrors PyTorch's torch.* surface.
# Invariant: no Tier-2 symbols (Module, Parameter, Linear, Adam, ...) here.

__all__ = [
    # ── metadata ──────────────────────────────────────────────────────────
    "__version__",
    # ── dtype ─────────────────────────────────────────────────────────────
    "dtype", "dtypes",
    "float16", "bfloat16", "float32", "float64",
    "int8", "int16", "int32", "int64",
    "bool_", "complex64",
    "half", "double", "short", "long",   # aliases (excluded from import *)
    # ── device / defaults ─────────────────────────────────────────────────
    "device",
    "set_default_dtype", "get_default_dtype",
    "set_default_device", "get_default_device",
    # ── core tensor ───────────────────────────────────────────────────────
    "Tensor",
    # ── factory — deterministic ───────────────────────────────────────────
    "tensor", "as_tensor", "from_numpy",
    "zeros", "ones", "empty", "full", "eye", "arange", "linspace",
    "zeros_like", "ones_like", "empty_like", "full_like",
    # ── factory — random ──────────────────────────────────────────────────
    "rand", "randn", "randint", "bernoulli", "normal",
    "rand_like", "randn_like", "manual_seed",
    # ── ops — unary ───────────────────────────────────────────────────────
    "abs", "neg", "sign",
    "exp", "log", "log2", "sqrt", "square", "reciprocal", "rsqrt",
    "floor", "ceil", "round",
    "sin", "cos", "tan", "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh",
    "relu", "sigmoid", "silu", "gelu", "mish", "selu", "softplus",
    "softmax", "log_softmax",
    "clip", "clamp",
    "isinf", "isnan", "isfinite", "nan_to_num",
    # ── ops — binary ──────────────────────────────────────────────────────
    "add", "sub", "mul", "div", "pow",
    "matmul", "dot", "inner", "outer", "tensordot",
    "maximum", "minimum",
    "equal", "not_equal", "greater", "greater_equal", "less", "less_equal",
    # ── ops — reduction ───────────────────────────────────────────────────
    "sum", "mean", "max", "min", "prod",
    "argmax", "argmin", "cumsum", "cumprod",
    "std", "var", "trace", "any", "all",
    # ── tensor manipulation ───────────────────────────────────────────────
    "reshape", "permute", "transpose", "unsqueeze", "squeeze", "flatten",
    "expand", "broadcast_to", "repeat", "tile",
    "cat", "stack", "hstack", "vstack",
    "split", "chunk", "unbind",
    "gather", "where", "masked_fill", "pad", "roll",
    "sort", "argsort", "topk",
    "nonzero", "unique", "meshgrid",
    "tril", "triu",
    "contiguous", "detach", "clone",
    # ── grad control ──────────────────────────────────────────────────────
    "no_grad", "enable_grad", "is_grad_enabled", "set_grad_enabled", "inference_mode",
    # ── type predicates ───────────────────────────────────────────────────
    "is_tensor", "is_floating_point", "is_complex", "is_signed",
    # ── serialization ─────────────────────────────────────────────────────
    "save", "load",
    # ── subpackages ───────────────────────────────────────────────────────
    "nn", "optim", "autograd", "linalg",
    "utils", "amp", "profiler", "einops",
    "metal", "backends", "testing",
]

# ── Sets derived from __all__ — used by __getattr__ below ────────────────────
_FACTORY_NAMES: frozenset[str] = frozenset([
    "tensor", "as_tensor", "from_numpy",
    "zeros", "ones", "empty", "full", "eye", "arange", "linspace",
    "zeros_like", "ones_like", "empty_like", "full_like",
    "rand", "randn", "randint", "bernoulli", "normal",
    "rand_like", "randn_like", "manual_seed",
])

_OPS_NAMES: frozenset[str] = frozenset([
    # unary — elementwise
    "abs", "neg", "sign",
    "exp", "log", "log2", "sqrt", "square", "reciprocal", "rsqrt",
    "floor", "ceil", "round",
    "sin", "cos", "tan", "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh",
    "relu", "sigmoid", "silu", "gelu", "mish", "selu", "softplus",
    "softmax", "log_softmax",
    "clip", "clamp",
    "isinf", "isnan", "isfinite", "nan_to_num",
    # binary
    "add", "sub", "mul", "div", "pow",
    "matmul", "dot", "inner", "outer", "tensordot",
    "maximum", "minimum",
    "equal", "not_equal", "greater", "greater_equal", "less", "less_equal",
    # reduction
    "sum", "mean", "max", "min", "prod",
    "argmax", "argmin", "cumsum", "cumprod",
    "std", "var", "trace", "any", "all",
    # manipulation
    "reshape", "permute", "transpose", "unsqueeze", "squeeze", "flatten",
    "expand", "broadcast_to", "repeat", "tile",
    "cat", "stack", "hstack", "vstack",
    "split", "chunk", "unbind",
    "gather", "where", "masked_fill", "pad", "roll",
    "sort", "argsort", "topk",
    "nonzero", "unique", "meshgrid",
    "tril", "triu",
    "contiguous", "detach", "clone",
])

_GRAD_NAMES: frozenset[str] = frozenset([
    "no_grad", "enable_grad", "is_grad_enabled", "set_grad_enabled", "inference_mode",
])

_SUBPKG_NAMES: frozenset[str] = frozenset([
    "nn", "optim", "autograd", "linalg",
    "utils", "amp", "profiler", "einops",
    "metal", "backends", "testing",
])
# fmt: on


def __getattr__(name: str) -> object:
    import sys as _sys

    _g = globals()

    if name == "Tensor":
        from lucid._tensor.tensor import Tensor

        _g["Tensor"] = Tensor
        return Tensor

    if name in _FACTORY_NAMES:
        import lucid._factories as _fac

        obj = getattr(_fac, name)
        _g[name] = obj
        return obj

    if name in _OPS_NAMES:
        import lucid._ops as _ops

        obj = getattr(_ops, name)
        _g[name] = obj
        return obj

    if name in _GRAD_NAMES:
        from lucid.autograd._grad_mode import (
            no_grad,
            enable_grad,
            is_grad_enabled,
            set_grad_enabled,
            inference_mode,
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

    if name in ("is_tensor", "is_floating_point", "is_complex", "is_signed"):
        from lucid._C import engine as _ce
        from lucid._dtype import (
            float16,
            bfloat16,
            float32,
            float64,
            complex64,
            int8,
            int16,
            int32,
            int64,
            bool_,
        )

        _FLOAT_DTYPES = frozenset([_ce.F16, _ce.F32, _ce.F64])
        _COMPLEX_DTYPES = frozenset([_ce.C64])
        _SIGNED_DTYPES = frozenset(
            [_ce.F16, _ce.F32, _ce.F64, _ce.C64, _ce.I8, _ce.I16, _ce.I32, _ce.I64]
        )

        def is_tensor(obj) -> bool:  # type: ignore
            """Return True if *obj* is a lucid Tensor."""
            from lucid._tensor.tensor import Tensor as _T

            return isinstance(obj, _T)

        def is_floating_point(t) -> bool:  # type: ignore
            """Return True if *t* has a floating-point dtype."""
            from lucid._dispatch import _unwrap

            return _unwrap(t).dtype in _FLOAT_DTYPES

        def is_complex(t) -> bool:  # type: ignore
            """Return True if *t* has a complex dtype."""
            from lucid._dispatch import _unwrap

            return _unwrap(t).dtype in _COMPLEX_DTYPES

        def is_signed(t) -> bool:  # type: ignore
            """Return True if *t* has a signed numeric dtype."""
            from lucid._dispatch import _unwrap

            return _unwrap(t).dtype in _SIGNED_DTYPES

        _g["is_tensor"] = is_tensor
        _g["is_floating_point"] = is_floating_point
        _g["is_complex"] = is_complex
        _g["is_signed"] = is_signed
        return _g[name]

    if name in ("save", "load"):
        import lucid.serialization as _ser

        obj = getattr(_ser, name)
        _g[name] = obj
        return obj

    if name in _SUBPKG_NAMES:
        pkg_key = f"lucid.{name}"
        if pkg_key in _sys.modules:
            mod = _sys.modules[pkg_key]
        else:
            import importlib

            mod = importlib.import_module(pkg_key)
        _g[name] = mod
        return mod

    if name == "dtypes":
        import lucid.dtypes as _dtypes

        _g["dtypes"] = _dtypes
        return _dtypes

    raise AttributeError(f"module 'lucid' has no attribute '{name}'")
