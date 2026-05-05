"""
Lucid — Apple Silicon ML framework.
"""

from typing import Callable

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
    # New API-compatible module-level aliases
    # lucid.float / lucid.int / lucid.bool shadow Python builtins
    # only as module attributes — no impact on user code
    float32 as float,   # lucid.float  == float32  == float32
    int32   as int,     # lucid.int    == int32    == int32
    bool_   as bool,    # lucid.bool   == bool_   == bool_
)
from lucid._device import device
from lucid._globals import (
    set_default_dtype,
    get_default_dtype,
    set_default_device,
    get_default_device,
)

# ── Public API ────────────────────────────────────────────────────────────────
# Organised by category — mirrors the standard tensor framework surface.
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
    "zeros", "ones", "empty", "full", "eye", "arange", "linspace", "logspace",
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
    "expand", "broadcast_to", "repeat", "repeat_interleave", "tile",
    "cat", "stack", "hstack", "vstack",
    "split", "chunk", "unbind",
    "gather", "scatter_add", "where", "masked_fill", "pad", "roll",
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
    # ── public type aliases ───────────────────────────────────────────────
    "Scalar", "TensorLike", "DeviceLike", "DTypeLike", "ShapeLike",
    "StateDict", "TensorOrScalar",
    "HasShape", "SupportsNumpyConversion", "SupportsGrad", "TensorLikeProtocol",
]

# ── Name sets used by the dispatch table ─────────────────────────────────────
_FACTORY_NAMES: frozenset[str] = frozenset(
    [
        "tensor",
        "as_tensor",
        "from_numpy",
        "zeros",
        "ones",
        "empty",
        "full",
        "eye",
        "arange",
        "linspace",
        "logspace",
        "zeros_like",
        "ones_like",
        "empty_like",
        "full_like",
        "rand",
        "randn",
        "randint",
        "bernoulli",
        "normal",
        "rand_like",
        "randn_like",
        "manual_seed",
    ]
)

_SCATTER_NAMES: frozenset[str] = frozenset(["scatter_add"])

_OPS_NAMES: frozenset[str] = frozenset(
    [
        "abs",
        "neg",
        "sign",
        "exp",
        "log",
        "log2",
        "sqrt",
        "square",
        "reciprocal",
        "rsqrt",
        "floor",
        "ceil",
        "round",
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "relu",
        "sigmoid",
        "silu",
        "gelu",
        "mish",
        "selu",
        "softplus",
        "softmax",
        "log_softmax",
        "clip",
        "clamp",
        "isinf",
        "isnan",
        "isfinite",
        "nan_to_num",
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "matmul",
        "dot",
        "inner",
        "outer",
        "tensordot",
        "maximum",
        "minimum",
        "equal",
        "not_equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "sum",
        "mean",
        "max",
        "min",
        "prod",
        "argmax",
        "argmin",
        "cumsum",
        "cumprod",
        "std",
        "var",
        "trace",
        "any",
        "all",
        "reshape",
        "permute",
        "transpose",
        "unsqueeze",
        "squeeze",
        "flatten",
        "expand",
        "broadcast_to",
        "repeat",
        "repeat_interleave",
        "tile",
        "cat",
        "stack",
        "hstack",
        "vstack",
        "split",
        "chunk",
        "unbind",
        "gather",
        "scatter_add",
        "where",
        "masked_fill",
        "pad",
        "roll",
        "sort",
        "argsort",
        "topk",
        "nonzero",
        "unique",
        "meshgrid",
        "tril",
        "triu",
        "contiguous",
        "detach",
        "clone",
        "ravel",
        "diagonal",
    ]
)

_GRAD_NAMES: frozenset[str] = frozenset(
    [
        "no_grad",
        "enable_grad",
        "is_grad_enabled",
        "set_grad_enabled",
        "inference_mode",
    ]
)

_PREDICATE_NAMES: frozenset[str] = frozenset(
    [
        "is_tensor",
        "is_floating_point",
        "is_complex",
        "is_signed",
    ]
)

_SERIALIZATION_NAMES: frozenset[str] = frozenset(["save", "load"])

_SUBPKG_NAMES: frozenset[str] = frozenset(
    [
        "nn",
        "optim",
        "autograd",
        "linalg",
        "utils",
        "amp",
        "profiler",
        "einops",
        "metal",
        "backends",
        "testing",
    ]
)

_TYPE_ALIAS_NAMES: frozenset[str] = frozenset(
    [
        "Scalar",
        "TensorLike",
        "DeviceLike",
        "DTypeLike",
        "ShapeLike",
        "StateDict",
        "TensorOrScalar",
        "HasShape",
        "SupportsNumpyConversion",
        "SupportsGrad",
        "TensorLikeProtocol",
    ]
)

# ── Lazy group loaders ────────────────────────────────────────────────────────
# Each loader is called at most once per interpreter session: it populates
# globals() for every name in the group, then is never called again.
#
# To add a new lazily-imported group:
#   1. Define its name frozenset above.
#   2. Write a _load_<group>() function below.
#   3. Register it via @_register(<frozenset>).
#
# fmt: on

_GROUP_LOADERS: dict[str, Callable[[], dict[str, object]]] = {}


def _register(
    names: frozenset[str],
) -> Callable[[Callable[[], dict[str, object]]], Callable[[], dict[str, object]]]:
    """Register a batch loader for a set of lazily-imported names."""

    def decorator(
        fn: Callable[[], dict[str, object]],
    ) -> Callable[[], dict[str, object]]:
        for name in names:
            _GROUP_LOADERS[name] = fn
        return fn

    return decorator


@_register(_FACTORY_NAMES)
def _load_factories() -> dict[str, object]:
    import lucid._factories as _fac

    return {n: getattr(_fac, n) for n in _FACTORY_NAMES}


@_register(_OPS_NAMES)
def _load_ops() -> dict[str, object]:
    import lucid._ops as _ops

    return {n: getattr(_ops, n) for n in _OPS_NAMES}


@_register(_SCATTER_NAMES)
def _load_scatter() -> dict[str, object]:
    from lucid._ops import scatter_add

    return {"scatter_add": scatter_add}


@_register(_GRAD_NAMES)
def _load_grad() -> dict[str, object]:
    from lucid.autograd._grad_mode import (
        no_grad,
        enable_grad,
        is_grad_enabled,
        set_grad_enabled,
        inference_mode,
    )

    return dict(
        no_grad=no_grad,
        enable_grad=enable_grad,
        is_grad_enabled=is_grad_enabled,
        set_grad_enabled=set_grad_enabled,
        inference_mode=inference_mode,
    )


@_register(_PREDICATE_NAMES)
def _load_predicates() -> dict[str, object]:
    from lucid._C import engine as _ce

    _FLOAT_DTYPES = frozenset([_ce.F16, _ce.F32, _ce.F64])
    _COMPLEX_DTYPES = frozenset([_ce.C64])
    _SIGNED_DTYPES = frozenset(
        [_ce.F16, _ce.F32, _ce.F64, _ce.C64, _ce.I8, _ce.I16, _ce.I32, _ce.I64]
    )

    def is_tensor(obj: object) -> bool:  # type: ignore
        """Return True if *obj* is a lucid Tensor."""
        from lucid._tensor.tensor import Tensor as _T

        return isinstance(obj, _T)

    def is_floating_point(t: object) -> bool:  # type: ignore
        """Return True if *t* has a floating-point dtype."""
        from lucid._dispatch import _unwrap

        return _unwrap(t).dtype in _FLOAT_DTYPES  # type: ignore[arg-type]

    def is_complex(t: object) -> bool:  # type: ignore
        """Return True if *t* has a complex dtype."""
        from lucid._dispatch import _unwrap

        return _unwrap(t).dtype in _COMPLEX_DTYPES  # type: ignore[arg-type]

    def is_signed(t: object) -> bool:  # type: ignore
        """Return True if *t* has a signed numeric dtype."""
        from lucid._dispatch import _unwrap

        return _unwrap(t).dtype in _SIGNED_DTYPES  # type: ignore[arg-type]

    return dict(
        is_tensor=is_tensor,
        is_floating_point=is_floating_point,
        is_complex=is_complex,
        is_signed=is_signed,
    )


@_register(_SERIALIZATION_NAMES)
def _load_serialization() -> dict[str, object]:
    import lucid.serialization as _ser

    return {"save": _ser.save, "load": _ser.load}


@_register(_TYPE_ALIAS_NAMES)
def _load_type_aliases() -> dict[str, object]:
    from lucid._types import (
        Scalar,
        TensorLike,
        DeviceLike,
        DTypeLike,
        ShapeLike,
        StateDict,
        TensorOrScalar,
        HasShape,
        SupportsNumpyConversion,
        SupportsGrad,
        TensorLikeProtocol,
    )

    return dict(
        Scalar=Scalar,
        TensorLike=TensorLike,
        DeviceLike=DeviceLike,
        DTypeLike=DTypeLike,
        ShapeLike=ShapeLike,
        StateDict=StateDict,
        TensorOrScalar=TensorOrScalar,
        HasShape=HasShape,
        SupportsNumpyConversion=SupportsNumpyConversion,
        SupportsGrad=SupportsGrad,
        TensorLikeProtocol=TensorLikeProtocol,
    )


# ── lucid.eval(*tensors) ──────────────────────────────────────────────────────


def eval(*tensors: object) -> None:  # type: ignore[override]
    """Force immediate evaluation of one or more tensors.

    On Metal (MLX backend) this flushes the lazy computation graph for all
    supplied tensors in a single ``mlx.core.eval()`` call — more efficient
    than calling ``.eval()`` on each tensor individually because MLX can
    schedule them together.

    On CPU this is a no-op.

    Recommended training-loop pattern on Metal::

        loss.eval()                          # flush forward graph BEFORE backward
        loss.backward()
        optimizer.step()
        lucid.eval(*model.parameters())      # flush param updates AFTER step

    Flushing the forward graph before backward keeps each MLX evaluation
    scope small.  Deferring both flushes into one call after the optimizer
    step causes the forward + backward + optimizer graph to accumulate as a
    single large graph, which is significantly slower to schedule.
    """
    from lucid._C import engine as _ce
    from lucid._tensor.tensor import Tensor as _T

    impls = [t._impl for t in tensors if isinstance(t, _T)]
    if impls:
        _ce.eval_tensors(impls)  # C++ batch eval — no mlx import


# ── Module __getattr__ ────────────────────────────────────────────────────────


def __getattr__(name: str) -> object:
    _g = globals()

    # Tensor is special-cased: it lives in a TYPE_CHECKING-guarded module.
    if name == "Tensor":
        from lucid._tensor.tensor import Tensor

        _g["Tensor"] = Tensor
        return Tensor

    # All other lazily-imported groups share the same dispatch pattern.
    loader = _GROUP_LOADERS.get(name)
    if loader is not None:
        _g.update(loader())
        return _g[name]

    # Subpackages: import on demand and cache.
    if name in _SUBPKG_NAMES:
        import importlib
        import sys as _sys

        pkg_key = f"lucid.{name}"
        mod = _sys.modules.get(pkg_key) or importlib.import_module(pkg_key)
        _g[name] = mod
        return mod

    # dtypes namespace object (separate from the dtype class).
    if name == "dtypes":
        import lucid.dtypes as _dtypes

        _g["dtypes"] = _dtypes
        return _dtypes

    raise AttributeError(f"module 'lucid' has no attribute '{name}'")
