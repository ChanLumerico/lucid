"""
nn.functional convolution operations.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _normalize_int_or_tuple(v: int | tuple[int, ...], n: int) -> tuple[int, ...]:
    if isinstance(v, int):
        return (v,) * n
    return tuple(v)


def conv1d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    groups: int = 1,
) -> Tensor:
    """1D convolution."""
    s = _normalize_int_or_tuple(stride, 1)[0]
    p = _normalize_int_or_tuple(padding, 1)[0]
    d = _normalize_int_or_tuple(dilation, 1)[0]
    b = _unwrap(bias) if bias is not None else None
    return _wrap(_C_engine.nn.conv1d(_unwrap(x), _unwrap(weight), b, s, p, d, groups))


def conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> Tensor:
    """2D convolution."""
    sh, sw = _normalize_int_or_tuple(stride, 2)
    ph, pw = _normalize_int_or_tuple(padding, 2)
    dh, dw = _normalize_int_or_tuple(dilation, 2)
    b = _unwrap(bias) if bias is not None else None
    return _wrap(
        _C_engine.nn.conv2d(
            _unwrap(x), _unwrap(weight), b, sh, sw, ph, pw, dh, dw, groups
        )
    )


def conv3d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
    dilation: int | tuple[int, int, int] = 1,
    groups: int = 1,
) -> Tensor:
    """3D convolution."""
    sd, sh, sw = _normalize_int_or_tuple(stride, 3)
    pd, ph, pw = _normalize_int_or_tuple(padding, 3)
    dd, dh, dw = _normalize_int_or_tuple(dilation, 3)
    b = _unwrap(bias) if bias is not None else None
    return _wrap(
        _C_engine.nn.conv3d(
            _unwrap(x), _unwrap(weight), b, sd, sh, sw, pd, ph, pw, dd, dh, dw, groups
        )
    )


def conv_transpose1d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    output_padding: int | tuple[int, ...] = 0,
    groups: int = 1,
    dilation: int | tuple[int, ...] = 1,
) -> Tensor:
    """Transposed 1D convolution."""
    s = _normalize_int_or_tuple(stride, 1)[0]
    p = _normalize_int_or_tuple(padding, 1)[0]
    op = _normalize_int_or_tuple(output_padding, 1)[0]
    d = _normalize_int_or_tuple(dilation, 1)[0]
    b = _unwrap(bias) if bias is not None else None
    return _wrap(
        _C_engine.nn.conv_transpose1d(
            _unwrap(x), _unwrap(weight), b, s, p, op, groups, d
        )
    )


def conv_transpose2d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
    groups: int = 1,
    dilation: int | tuple[int, int] = 1,
) -> Tensor:
    """Transposed 2D convolution."""
    sh, sw = _normalize_int_or_tuple(stride, 2)
    ph, pw = _normalize_int_or_tuple(padding, 2)
    oh, ow = _normalize_int_or_tuple(output_padding, 2)
    dh, dw = _normalize_int_or_tuple(dilation, 2)
    b = _unwrap(bias) if bias is not None else None
    return _wrap(
        _C_engine.nn.conv_transpose2d(
            _unwrap(x), _unwrap(weight), b, sh, sw, ph, pw, oh, ow
        )
    )


def conv_transpose3d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
    output_padding: int | tuple[int, int, int] = 0,
    groups: int = 1,
    dilation: int | tuple[int, int, int] = 1,
) -> Tensor:
    """Transposed 3D convolution."""
    sd, sh, sw = _normalize_int_or_tuple(stride, 3)
    pd, ph, pw = _normalize_int_or_tuple(padding, 3)
    od, oh, ow = _normalize_int_or_tuple(output_padding, 3)
    dd, dh, dw = _normalize_int_or_tuple(dilation, 3)
    b = _unwrap(bias) if bias is not None else None
    return _wrap(
        _C_engine.nn.conv_transpose3d(
            _unwrap(x),
            _unwrap(weight),
            b,
            sd,
            sh,
            sw,
            pd,
            ph,
            pw,
            od,
            oh,
            ow,
            groups,
            dd,
            dh,
            dw,
        )
    )
