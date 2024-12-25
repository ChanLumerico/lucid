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
            tensor.grad[at] = tensor.grad[at] + grad


def _check_is_tensor(any: Tensor | _ArrayOrScalar) -> Tensor:
    if not isinstance(any, Tensor):
        return Tensor(any)
    return any


def _match_grad_shape(data: _NumPyArray, grad: _NumPyArray) -> _NumPyArray:
    if data.shape == grad.shape:
        return grad
    if data.ndim == 0:
        return grad.sum()

    matched_grad = grad
    if data.ndim > grad.ndim:
        matched_grad = grad.reshape(data.shape)

    elif data.ndim < grad.ndim:
        squeezed = grad.squeeze()
        matched_grad = squeezed.reshape(data.shape)

    else:
        if data.size == grad.size:
            matched_grad = grad
        elif data.size < grad.size:
            axis = []
            if data.ndim == 0:
                axis.extend(range(grad.ndim))
            else:
                for ax in range(data.ndim):
                    if data.shape[ax] != grad.shape[ax] and data.shape[ax] == 1:
                        axis.append(ax)

            matched_grad = np.sum(grad, axis=tuple(axis)).reshape(data.shape)
        else:
            if data.size % grad.size == 0:
                matched_grad = np.broadcast_to(grad, data.shape)

    return matched_grad


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
