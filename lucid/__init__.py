"""
# `Lucid `

**Lucid** is an educational deep learning framework developed to help users understand
the underlying mechanics of deep learning models and tensor operations.

It is designed to provide a simple yet powerful environment to experiment with neural networks,
optimization, and backpropagation using only `NumPy`.

Lucid is ideal for those who want to learn about the inner workings of deep learning
algorithms and operations without the complexity of high-level frameworks.

[📑 Lucid Documentation](https://chanlumerico.github.io/lucid/build/html/index.html)
"""

from contextlib import contextmanager, AbstractContextManager
from typing import Any, Generator, SupportsIndex, Callable, Self, Optional, Type
from types import TracebackType, ModuleType
from functools import wraps
from pathlib import Path
import inspect

import os
import sys
import json
import math
import numpy as np

_GlobalFlag = bool

USE_CPP_FUNC_OP: _GlobalFlag = False
USE_BAKCWARD_FUSION: _GlobalFlag = True

_CPP_USAGE: int = 0

from lucid._tensor import *
from lucid._func import *
from lucid._utils import *

from lucid._backend.metal import mx

from lucid.types import (
    _ArrayOrScalar,
    _NumPyArray,
    _MLXArray,
    _ArrayLike,
    _ShapeLike,
    _DeviceType,
    _BuiltinNumeric,
    Numeric,
)
from lucid.error import *
try:
    from lucid.port import *
except Exception:
    pass

import lucid.linalg as linalg
import lucid.random as random
import lucid.einops as einops
try:
    import lucid.nn as nn
except Exception:
    class _NNFallback:
        Module = Any

    nn = _NNFallback()
import lucid.types as types
import lucid.autograd as autograd
try:
    import lucid.visual as visual
except Exception:
    visual = None

_grad_enabled: bool = True
_flops_enabled: bool = False

newaxis = None

pi = math.pi
inf = math.inf

Int = types.Int
Int8, Int16, Int32, Int64 = (types.Int8, types.Int16, types.Int32, types.Int64)
Char, Short, Long = (Int8, Int16, Int64)

Float = types.Float
Float16, Float32, Float64 = (types.Float16, types.Float32, types.Float64)
Half, Double = (Float16, Float64)

Complex = types.Complex
Complex64 = types.Complex64


def tensor(
    data: Tensor | _ArrayOrScalar,
    requires_grad: bool = False,
    keep_grad: bool = False,
    dtype: _BuiltinNumeric | Numeric | None = None,
    device: _DeviceType = "cpu",
) -> Tensor:
    if isinstance(data, Tensor):
        data = data.data
    return Tensor(data, requires_grad, keep_grad, dtype, device)


def to_tensor(
    a: _ArrayLike,
    requires_grad: bool = False,
    keep_grad: bool = False,
    dtype: _BuiltinNumeric | Numeric | None = None,
    device: _DeviceType = "cpu",
) -> Tensor:
    return tensor(a, requires_grad, keep_grad, dtype, device)


class _NoGrad(AbstractContextManager):
    __slots__ = ("_prev_state",)

    def __enter__(self) -> Self:
        global _grad_enabled
        self._prev_state = _grad_enabled

        _grad_enabled = False
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        _ = (exc_type, exc_value, traceback)
        global _grad_enabled

        _grad_enabled = self._prev_state
        return False

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with _NoGrad():
                return func(*args, **kwargs)

        return wrapper


no_grad = _NoGrad


def grad_enabled() -> bool:
    return _grad_enabled


@contextmanager
def count_flops() -> Generator:
    global _flops_enabled
    prev_state = _flops_enabled
    _flops_enabled = True
    try:
        yield
    finally:
        _flops_enabled = prev_state


def flops_enabled() -> bool:
    return _flops_enabled


def shape(a: Tensor | _NumPyArray | _MLXArray) -> _ShapeLike:
    if hasattr(a, "shape"):
        return a.shape
    raise ValueError(f"The argument must be a Tensor or a NumPy array.")


def _check_input_dim(tensor: Tensor, dim: int) -> None:
    if tensor.ndim != dim:
        raise ValueError(f"expected {dim}D input (got {tensor.ndim}D input).")


def _set_tensor_grad(
    tensor: Tensor, grad: _NumPyArray | _MLXArray, at: SupportsIndex = ...
) -> None:
    if not tensor.requires_grad:
        return
    if tensor.grad is None:
        tensor.grad = grad
    else:
        if tensor.is_cpu() and not tensor.grad.flags.writeable:
            tensor.grad = tensor.grad.copy()

        if tensor.is_gpu():
            if at == Ellipsis:
                at = slice(None, None, None)

        if tensor.grad.ndim == 0:
            tensor.grad += grad
        else:
            tensor.grad[at] = tensor.grad[at] + grad


def _check_is_tensor(
    any: Tensor | _ArrayOrScalar,
    device: _DeviceType = "cpu",
    dtype: _BuiltinNumeric | Numeric | None = None,
) -> Tensor:
    if isinstance(any, Tensor):
        return any

    is_scalar = not isinstance(any, (_NumPyArray, _MLXArray, list, tuple))
    if dtype is not None and is_scalar:
        return Tensor(any, device=device, dtype=dtype)

    return Tensor(any, device=device)


def _match_grad_shape(
    data: _NumPyArray | _MLXArray,
    grad: _NumPyArray | _MLXArray,
    device: _DeviceType = "cpu",
) -> _NumPyArray | _MLXArray:
    if data.shape == grad.shape:
        return grad
    if data.ndim == 0:
        return grad.sum()
    if grad.ndim == 0:
        return (
            np.broadcast_to(grad, data.shape)
            if device == "cpu"
            else mx.broadcast_to(grad, data.shape)
        )

    if data.size == grad.size:
        return grad.reshape(data.shape)

    elif data.size > grad.size:
        grad_squeeze = grad.flatten()
        expand_factor = data.size / grad.size
        if expand_factor % 1 != 0:
            raise ValueError(
                f"Cannot broadcast grad of {grad.shape} to data of {data.shape}."
            )
        grad_expand = (
            grad_squeeze[..., None].repeat(int(expand_factor), axis=-1)
            if device == "cpu"
            else mx.repeat(grad_squeeze[..., None], int(expand_factor), axis=1)
        )
        return grad_expand.reshape(data.shape)

    elif data.size < grad.size:
        if grad.size % data.size != 0:
            raise ValueError(
                f"Cannot collapse grad of {grad.shape} to data of {data.shape}."
            )
        new_shape = tuple()
        remain_size = grad.size

        for d_dim in data.shape:
            fac = remain_size // d_dim
            new_shape += (d_dim,)
            remain_size = fac

        new_shape += (fac,)
        return grad.reshape(new_shape).sum(axis=-1)

    else:
        raise ValueError("Unknown error occurred.")


def _get_overloaded_shape(args: int | _ShapeLike) -> _ShapeLike:
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        shape = tuple(args[0])
    else:
        shape = tuple(args)
    return shape


_PACKAGE_DIR: Path = Path(__file__).resolve().parent
MODELS_REGISTRY_PATH: Path = _PACKAGE_DIR / "models" / "registry.json"

_ModuleReturnFunc = Callable[[Any], nn.Module]
_ModuleClass = type[nn.Module]


def _load_models_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({}, f)
        return {}

    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def _is_legacy_flat_registry(registry: dict[str, Any]) -> bool:
    for value in registry.values():
        if not isinstance(value, dict):
            continue
        if "name" in value and ("param_size" in value or "parameter_size" in value):
            return True
    return False


def _migrate_flat_registry(registry: dict[str, Any]) -> dict[str, Any]:
    nested: dict[str, Any] = {}

    for model_name, info in registry.items():
        if not isinstance(info, dict):
            continue

        category = info.get("category")
        if isinstance(category, str) and category:
            category_parts = [category]
        elif isinstance(category, list) and category:
            category_parts = [str(part) for part in category]
        else:
            category_parts = ["uncategorized"]

        cursor = nested
        for part in category_parts:
            cursor = cursor.setdefault(part, {})

        cursor[model_name] = {
            "parameter_size": int(
                info.get("param_size", info.get("parameter_size", 0))
            ),
            "submodule_count": int(info.get("submodule_count", 0)),
        }

    return nested


def _get_registry_category_path(func: _ModuleReturnFunc) -> list[str]:
    module = sys.modules.get(func.__module__)
    module_file = getattr(module, "__file__", None)
    models_dir = _PACKAGE_DIR / "models"

    if module_file:
        try:
            rel = Path(module_file).resolve().relative_to(models_dir).with_suffix("")
            parts = list(rel.parts)
            if parts and parts[-1] == "__init__":
                parts = parts[:-1]
            if parts:
                return parts
        except ValueError:
            pass

    module_parts = func.__module__.split(".")
    if "models" in module_parts:
        parts = module_parts[module_parts.index("models") + 1 :]
        return parts if parts else ["uncategorized"]
    return ["uncategorized"]


def _has_model_entry(
    registry: dict[str, Any], path: list[str], model_name: str
) -> bool:
    cursor: Any = registry
    for part in path:
        if not isinstance(cursor, dict) or part not in cursor:
            return False
        cursor = cursor[part]
    return isinstance(cursor, dict) and model_name in cursor


def _upsert_model_entry(
    registry: dict[str, Any], path: list[str], model_name: str, entry: dict[str, int]
) -> None:
    cursor = registry
    for part in path:
        cursor = cursor.setdefault(part, {})
    cursor[model_name] = entry


def _build_registry_entry(model: nn.Module, *, is_class: bool) -> dict[str, Any]:
    submodule_count = 0
    for _ in model.modules():
        submodule_count += 1

    entry: dict[str, Any] = {
        "parameter_size": int(model.parameter_size),
        "submodule_count": submodule_count - 1 if submodule_count > 0 else 0,
    }
    if is_class:
        entry["factory_type"] = "class"
    return entry


def _maybe_register_model(target: Any, model: nn.Module, *, is_class: bool) -> None:
    registry = _load_models_registry(MODELS_REGISTRY_PATH)
    if _is_legacy_flat_registry(registry):
        registry = _migrate_flat_registry(registry)

    model_name = target.__name__
    category_path = _get_registry_category_path(target)

    if not _has_model_entry(registry, category_path, model_name):
        _upsert_model_entry(
            registry,
            category_path,
            model_name,
            _build_registry_entry(model, is_class=is_class),
        )
        with open(MODELS_REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=4)


def register_model(
    target: _ModuleReturnFunc | _ModuleClass,
) -> _ModuleReturnFunc | _ModuleClass:
    if inspect.isclass(target):
        if not issubclass(target, nn.Module):
            raise TypeError("@register_model class target must inherit from nn.Module.")

        original_init = target.__init__

        @wraps(original_init)
        def wrapped_init(self, *args, **kwargs) -> None:
            weights = kwargs.pop("weights", None)
            original_init(self, *args, **kwargs)

            if os.environ.get("SPHINX_BUILD"):
                return

            _maybe_register_model(target, self, is_class=True)

            if weights is not None:
                import lucid.weights as W

                try:
                    W.apply(self, weights)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to apply pre-trained weights: {e}"
                    ) from e

        target.__init__ = wrapped_init
        return target

    func = target

    @wraps(func)
    def wrapper(*args, **kwargs) -> nn.Module:
        weights = kwargs.pop("weights", None)

        if os.environ.get("SPHINX_BUILD"):
            return func(*args, **kwargs)

        model = func(*args, **kwargs)
        model._alt_name = func.__name__
        _maybe_register_model(func, model, is_class=False)

        if weights is not None:
            import lucid.weights as W

            try:
                W.apply(model, weights)
            except Exception as e:
                raise RuntimeError(f"Failed to apply pre-trained weights: {e}") from e

        return model

    return wrapper


def _conv_view_limit_mb() -> int:
    from lucid.nn._kernel import conv as _conv_kernel

    return _conv_kernel.get_conv_view_limit_mb()


def __getattr__(name: str) -> Any:
    if name == "CONV_VIEW_LIMIT_MB":
        return _conv_view_limit_mb()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + ["CONV_VIEW_LIMIT_MB"])


class _LucidModule(ModuleType):
    def __setattr__(self, name: str, value: Any) -> None:
        if name == "CONV_VIEW_LIMIT_MB":
            raise AttributeError(
                "CONV_VIEW_LIMIT_MB is read-only; set LUCID_CONV_VIEW_LIMIT_MB "
                "before importing lucid."
            )
        super().__setattr__(name, value)


if not isinstance(sys.modules[__name__], _LucidModule):
    sys.modules[__name__].__class__ = _LucidModule
