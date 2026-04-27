"""
lucid.nn.functional._pool — max / average / adaptive pooling.

All ops route to the engine. The Python wrapper just normalizes
int-vs-tuple kernel/stride/padding arguments.
"""

from __future__ import annotations

from typing import Literal

from lucid._C.engine import nn as _C_nn
from lucid._tensor import Tensor
from lucid._bridge import impl_of


def _to_tuple_n(value, n: int) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * n
    if len(value) == 1 and n > 1:
        return tuple(value) * n
    return tuple(int(v) for v in value)


def avg_pool1d(input_: Tensor, kernel_size, stride=1, padding=0) -> Tensor:
    k = _to_tuple_n(kernel_size, 1)
    s = _to_tuple_n(stride, 1)
    p = _to_tuple_n(padding, 1)
    return Tensor._wrap(_C_nn.avg_pool1d(impl_of(input_), k[0], s[0], p[0]))


def avg_pool2d(input_: Tensor, kernel_size, stride=1, padding=0) -> Tensor:
    k = _to_tuple_n(kernel_size, 2)
    s = _to_tuple_n(stride, 2)
    p = _to_tuple_n(padding, 2)
    return Tensor._wrap(_C_nn.avg_pool2d(
        impl_of(input_), k[0], k[1], s[0], s[1], p[0], p[1]))


def avg_pool3d(input_: Tensor, kernel_size, stride=1, padding=0) -> Tensor:
    k = _to_tuple_n(kernel_size, 3)
    s = _to_tuple_n(stride, 3)
    p = _to_tuple_n(padding, 3)
    return Tensor._wrap(_C_nn.avg_pool3d(
        impl_of(input_), k[0], k[1], k[2],
        s[0], s[1], s[2], p[0], p[1], p[2]))


def max_pool1d(input_: Tensor, kernel_size, stride=1, padding=0) -> Tensor:
    k = _to_tuple_n(kernel_size, 1)
    s = _to_tuple_n(stride, 1)
    p = _to_tuple_n(padding, 1)
    return Tensor._wrap(_C_nn.max_pool1d(impl_of(input_), k[0], s[0], p[0]))


def max_pool2d(input_: Tensor, kernel_size, stride=1, padding=0) -> Tensor:
    k = _to_tuple_n(kernel_size, 2)
    s = _to_tuple_n(stride, 2)
    p = _to_tuple_n(padding, 2)
    return Tensor._wrap(_C_nn.max_pool2d(
        impl_of(input_), k[0], k[1], s[0], s[1], p[0], p[1]))


def max_pool3d(input_: Tensor, kernel_size, stride=1, padding=0) -> Tensor:
    k = _to_tuple_n(kernel_size, 3)
    s = _to_tuple_n(stride, 3)
    p = _to_tuple_n(padding, 3)
    return Tensor._wrap(_C_nn.max_pool3d(
        impl_of(input_), k[0], k[1], k[2],
        s[0], s[1], s[2], p[0], p[1], p[2]))


def adaptive_pool1d(input_: Tensor, output_size: int,
                    avg_or_max: Literal["avg", "max"]) -> Tensor:
    fn = (_C_nn.adaptive_avg_pool1d if avg_or_max == "avg"
          else _C_nn.adaptive_max_pool1d)
    return Tensor._wrap(fn(impl_of(input_), int(output_size)))


def adaptive_pool2d(input_: Tensor, output_size,
                    avg_or_max: Literal["avg", "max"]) -> Tensor:
    if isinstance(output_size, int):
        oh = ow = output_size
    else:
        oh, ow = output_size
    fn = (_C_nn.adaptive_avg_pool2d if avg_or_max == "avg"
          else _C_nn.adaptive_max_pool2d)
    return Tensor._wrap(fn(impl_of(input_), int(oh), int(ow)))


def adaptive_pool3d(input_: Tensor, output_size,
                    avg_or_max: Literal["avg", "max"]) -> Tensor:
    if isinstance(output_size, int):
        od = oh = ow = output_size
    else:
        od, oh, ow = output_size
    fn = (_C_nn.adaptive_avg_pool3d if avg_or_max == "avg"
          else _C_nn.adaptive_max_pool3d)
    return Tensor._wrap(fn(impl_of(input_), int(od), int(oh), int(ow)))
