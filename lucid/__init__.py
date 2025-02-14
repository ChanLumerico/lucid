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

from contextlib import contextmanager
from typing import Any, Generator, SupportsIndex, Callable
from functools import wraps
from pathlib import Path

import os
import sys
import json
import numpy as np

from lucid._tensor import Tensor
from lucid._func import *
from lucid._util import *

from lucid.types import _ArrayOrScalar, _NumPyArray, _ArrayLike, _ShapeLike

import lucid.linalg as linalg
import lucid.random as random
import lucid.einops as einops
import lucid.nn as nn

_grad_enabled: bool = True

newaxis = np.newaxis

pi = np.pi
inf = np.inf


def tensor(
    data: Tensor | _ArrayOrScalar,
    requires_grad: bool = False,
    keep_grad: bool = False,
    dtype: Any = np.float32,
) -> Tensor:
    if isinstance(data, Tensor):
        data = data.data
    return Tensor(data, requires_grad, keep_grad, dtype)


def to_tensor(
    a: _ArrayLike,
    requires_grad: bool = False,
    keep_grad: bool = False,
    dtype: Any = np.float32,
) -> Tensor:
    return tensor(a, requires_grad, keep_grad, dtype)


@contextmanager
def no_grad() -> Generator:
    global _grad_enabled
    prev_state = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = prev_state


def grad_enabled() -> bool:
    return _grad_enabled


def shape(a: Tensor | _NumPyArray) -> _ShapeLike:
    if hasattr(a, "shape"):
        return a.shape

    raise ValueError(f"The argument must be a Tensor or a NumPy array.")


def _check_input_dim(tensor: Tensor, dim: int) -> None:
    if tensor.ndim != dim:
        raise ValueError(f"expected {dim}D input (got {tensor.ndim}D input).")


def _set_tensor_grad(
    tensor: Tensor, grad: _NumPyArray, at: SupportsIndex = ...
) -> None:
    if tensor.requires_grad:
        if tensor.grad is None:
            tensor.grad = grad
        else:
            if not tensor.grad.flags.writeable:
                tensor.grad = tensor.grad.copy()
            tensor.grad[at] = tensor.grad[at] + grad


def _check_is_tensor(any: Tensor | _ArrayOrScalar) -> Tensor:
    if not isinstance(any, Tensor):
        return Tensor(any)
    return any


def _match_grad_shape(data: _NumPyArray, grad: _NumPyArray) -> _NumPyArray:
    if data.shape == grad.shape:
        return grad
    if data.ndim == 0:
        return np.sum(grad)
    if grad.ndim == 0:
        return np.broadcast_to(grad, data.shape)

    if data.size == grad.size:
        return grad.reshape(data.shape)

    elif data.size > grad.size:
        grad_squeeze = grad.flatten()
        expand_factor = data.size / grad.size
        if expand_factor % 1 != 0:
            raise ValueError(
                f"Cannot broadcast grad of {grad.shape} to data of {data.shape}."
            )

        grad_expand = grad_squeeze[..., None].repeat(int(expand_factor), axis=-1)
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


REGISTRY_PATH: Path = Path("lucid/models/registry.json")

_ModuleReturnFunc = Callable[[Any], nn.Module]


def register_model(func: _ModuleReturnFunc) -> _ModuleReturnFunc:
    @wraps(func)
    def wrapper(*args, **kwargs) -> nn.Module:
        if os.environ.get("SPHINX_BUILD"):
            return func(*args, **kwargs)

        if not REGISTRY_PATH.exists():
            REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(REGISTRY_PATH, "w") as f:
                json.dump({}, f)

        with open(REGISTRY_PATH, "r") as f:
            registry = json.load(f)

        model = func(*args, **kwargs)
        model._alt_name = func.__name__
        name = func.__name__

        if name in registry:
            return model

        family = model.__class__.__name__
        param_size = model.parameter_size
        arch = sys.modules[func.__module__].__package__.replace("lucid.models.", "")

        registry[name] = dict(
            name=name,
            family=family,
            param_size=param_size,
            arch=arch,
        )

        with open(REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=4)

        return model

    return wrapper
