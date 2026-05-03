"""
nn.functional pooling operations.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _int_or_tuple(v: int | tuple[int, ...], n: int) -> tuple[int, ...]:
    return (v,) * n if isinstance(v, int) else tuple(v)


def max_pool1d(
    x: Tensor,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...] | None = None,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    return_indices: bool = False,
    ceil_mode: bool = False,
) -> Tensor:
    """1D max pooling."""
    k = _int_or_tuple(kernel_size, 1)[0]
    s = k if stride is None else _int_or_tuple(stride, 1)[0]
    p = _int_or_tuple(padding, 1)[0]
    d = _int_or_tuple(dilation, 1)[0]
    return _wrap(_C_engine.nn.max_pool1d(_unwrap(x), k, s, p, d, ceil_mode))


def max_pool2d(
    x: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    return_indices: bool = False,
    ceil_mode: bool = False,
) -> Tensor:
    """2D max pooling."""
    kh, kw = _int_or_tuple(kernel_size, 2)
    sh, sw = _int_or_tuple(kernel_size if stride is None else stride, 2)
    ph, pw = _int_or_tuple(padding, 2)
    dh, dw = _int_or_tuple(dilation, 2)
    return _wrap(
        _C_engine.nn.max_pool2d(_unwrap(x), kh, kw, sh, sw, ph, pw, dh, dw, ceil_mode)
    )


def avg_pool1d(
    x: Tensor,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...] | None = None,
    padding: int | tuple[int, ...] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> Tensor:
    """1D average pooling."""
    k = _int_or_tuple(kernel_size, 1)[0]
    s = k if stride is None else _int_or_tuple(stride, 1)[0]
    p = _int_or_tuple(padding, 1)[0]
    return _wrap(
        _C_engine.nn.avg_pool1d(_unwrap(x), k, s, p, ceil_mode, count_include_pad)
    )


def avg_pool2d(
    x: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
) -> Tensor:
    """2D average pooling."""
    kh, kw = _int_or_tuple(kernel_size, 2)
    sh, sw = _int_or_tuple(kernel_size if stride is None else stride, 2)
    ph, pw = _int_or_tuple(padding, 2)
    return _wrap(
        _C_engine.nn.avg_pool2d(
            _unwrap(x), kh, kw, sh, sw, ph, pw, ceil_mode, count_include_pad
        )
    )


def adaptive_avg_pool1d(x: Tensor, output_size: int | tuple[int, ...]) -> Tensor:
    """1D adaptive average pooling."""
    sz = _int_or_tuple(output_size, 1)[0]
    return _wrap(_C_engine.nn.adaptive_avg_pool1d(_unwrap(x), sz))


def adaptive_avg_pool2d(x: Tensor, output_size: int | tuple[int, int]) -> Tensor:
    """2D adaptive average pooling."""
    oh, ow = _int_or_tuple(output_size, 2)
    return _wrap(_C_engine.nn.adaptive_avg_pool2d(_unwrap(x), oh, ow))


def adaptive_max_pool2d(
    x: Tensor,
    output_size: int | tuple[int, int],
    return_indices: bool = False,
) -> Tensor:
    """2D adaptive max pooling."""
    oh, ow = _int_or_tuple(output_size, 2)
    return _wrap(_C_engine.nn.adaptive_max_pool2d(_unwrap(x), oh, ow))


def adaptive_max_pool1d(
    x: Tensor,
    output_size: int | tuple[int, ...],
    return_indices: bool = False,
) -> Tensor:
    """1D adaptive max pooling."""
    sz = _int_or_tuple(output_size, 1)[0]
    return _wrap(_C_engine.nn.adaptive_max_pool1d(_unwrap(x), sz))


def adaptive_max_pool3d(
    x: Tensor,
    output_size: int | tuple[int, int, int],
    return_indices: bool = False,
) -> Tensor:
    """3D adaptive max pooling."""
    od, oh, ow = _int_or_tuple(output_size, 3)
    return _wrap(_C_engine.nn.adaptive_max_pool3d(_unwrap(x), od, oh, ow))


def adaptive_avg_pool3d(
    x: Tensor,
    output_size: int | tuple[int, int, int],
) -> Tensor:
    """3D adaptive average pooling."""
    od, oh, ow = _int_or_tuple(output_size, 3)
    return _wrap(_C_engine.nn.adaptive_avg_pool3d(_unwrap(x), od, oh, ow))


def max_pool3d(
    x: Tensor,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] | None = None,
    padding: int | tuple[int, int, int] = 0,
    dilation: int | tuple[int, int, int] = 1,
    return_indices: bool = False,
    ceil_mode: bool = False,
) -> Tensor:
    """3D max pooling."""
    kd, kh, kw = _int_or_tuple(kernel_size, 3)
    sd, sh, sw = _int_or_tuple(kernel_size if stride is None else stride, 3)
    pd, ph, pw = _int_or_tuple(padding, 3)
    return _wrap(_C_engine.nn.max_pool3d(_unwrap(x), kd, kh, kw, sd, sh, sw, pd, ph, pw))


def avg_pool3d(
    x: Tensor,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] | None = None,
    padding: int | tuple[int, int, int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
) -> Tensor:
    """3D average pooling."""
    kd, kh, kw = _int_or_tuple(kernel_size, 3)
    sd, sh, sw = _int_or_tuple(kernel_size if stride is None else stride, 3)
    pd, ph, pw = _int_or_tuple(padding, 3)
    return _wrap(_C_engine.nn.avg_pool3d(_unwrap(x), kd, kh, kw, sd, sh, sw, pd, ph, pw))
