"""
Convolution and transposed convolution modules.
"""

import math
from typing import Callable

from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike, _Size2d, _Size3d
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._factories.creation import empty
import lucid.nn.init as init
from lucid.nn.functional.conv import (
    conv1d,
    conv2d,
    conv3d,
    conv_transpose1d,
    conv_transpose2d,
    conv_transpose3d,
)
from lucid.nn.functional.sampling import pad as _F_pad

_VALID_PADDING_MODES = frozenset({"zeros", "reflect", "replicate", "circular"})

# Maps Conv padding_mode to F.pad mode.
_PADDING_MODE_TO_FPAD = {
    "zeros": "constant",
    "reflect": "reflect",
    "replicate": "replicate",
    "circular": "circular",
}


def _pair(v: _Size2d) -> tuple[int, int]:
    return (v, v) if isinstance(v, int) else tuple(v)  # type: ignore[return-value]


def _triple(v: _Size3d) -> tuple[int, int, int]:
    return (v, v, v) if isinstance(v, int) else tuple(v)  # type: ignore[return-value]


def _same_pad_pair(
    in_size: int, kernel: int, stride: int, dilation: int
) -> tuple[int, int]:
    """Compute (pad_lo, pad_hi) for `padding="same"` on one spatial dim.

    Reference parity: pad_lo = pad_total // 2, pad_hi = pad_total - pad_lo.
    For odd pad_total this is asymmetric (more padding on the high side).
    """
    out_size = (in_size + stride - 1) // stride
    pad_total = max(0, (out_size - 1) * stride + (kernel - 1) * dilation + 1 - in_size)
    pad_lo = pad_total // 2
    return pad_lo, pad_total - pad_lo


def _check_same_supported(stride_tuple: tuple[int, ...]) -> None:
    if any(s != 1 for s in stride_tuple):
        raise ValueError(
            "padding='same' is not supported with stride > 1 "
            f"(got stride={stride_tuple})"
        )


def _validate_padding_mode(mode: str) -> str:
    if mode not in _VALID_PADDING_MODES:
        raise ValueError(
            f"padding_mode must be one of {sorted(_VALID_PADDING_MODES)}, got {mode!r}"
        )
    return mode


def _validate_int_padding(padding: object, label: str) -> None:
    """Reject string padding for ConvTranspose."""
    if isinstance(padding, str):
        raise ValueError(
            f"{label}: string padding ({padding!r}) is not supported; "
            "use an int or tuple of ints"
        )


_ConvFn = Callable[..., Tensor]


def _conv_forward_with_mode(
    x: Tensor,
    weight: Parameter,
    bias: Parameter | None,
    stride: tuple[int, ...],
    pad_lo: tuple[int, ...],
    pad_hi: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
    padding_mode: str,
    conv_fn: _ConvFn,
) -> Tensor:
    """Dispatch a forward conv with arbitrary padding_mode and asymmetric pad.

    `pad_lo` / `pad_hi` are per-spatial-dim (first→last) padding amounts.
    When `padding_mode == "zeros"` and pad is symmetric, the engine conv
    handles the padding directly.  Otherwise we pre-pad via `F.pad` and
    call conv with padding=0 along that axis.
    """
    n: int = len(stride)
    symmetric: bool = all(pad_lo[i] == pad_hi[i] for i in range(n))
    if padding_mode == "zeros" and symmetric:
        engine_pad: int | tuple[int, ...] = pad_lo[0] if n == 1 else tuple(pad_lo)
        return _call_conv(
            conv_fn, x, weight, bias, stride, engine_pad, dilation, groups, n
        )
    # Pre-pad path.  F.pad uses last-dim-first flat tuple.
    # pad_lo[i], pad_hi[i] are spatial dim i (first→last); we need to reverse
    # so that the LAST spatial dim comes first in F.pad's flat tuple.
    pad_flat: list[int] = []
    for i in reversed(range(n)):
        pad_flat.extend([pad_lo[i], pad_hi[i]])
    fpad_mode: str = _PADDING_MODE_TO_FPAD[padding_mode]
    x_padded: Tensor = _F_pad(x, tuple(pad_flat), mode=fpad_mode)
    zero_pad: int | tuple[int, ...] = 0 if n == 1 else (0,) * n
    return _call_conv(
        conv_fn, x_padded, weight, bias, stride, zero_pad, dilation, groups, n
    )


def _call_conv(
    conv_fn: _ConvFn,
    x: Tensor,
    weight: Parameter,
    bias: Parameter | None,
    stride: tuple[int, ...],
    padding: int | tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
    n: int,
) -> Tensor:
    """Invoke conv_fn with the right argument arity.

    conv1d/2d/3d in `nn.functional.conv` accept (x, weight, bias, stride, padding,
    dilation, groups) but conv1d takes scalar ints while conv2d/3d take tuples.
    """
    if n == 1:
        s: int = stride[0] if isinstance(stride, tuple) else stride
        d: int = dilation[0] if isinstance(dilation, tuple) else dilation
        p: int = padding[0] if isinstance(padding, tuple) else padding
        return conv_fn(x, weight, bias, s, p, d, groups)
    return conv_fn(x, weight, bias, stride, padding, dilation, groups)


class Conv1d(Module):
    """1D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = _validate_padding_mode(padding_mode)
        if isinstance(padding, str):
            mode = padding.lower()
            if mode not in {"same", "valid"}:
                raise ValueError(
                    f"string padding must be 'same' or 'valid', got {padding!r}"
                )
            if mode == "same":
                _check_same_supported((stride,))
            self._padding_str: str | None = mode
            self.padding: int = 0
        else:
            self._padding_str = None
            self.padding = padding
        self.weight = Parameter(
            empty(
                out_channels,
                in_channels // groups,
                kernel_size,
                dtype=dtype,
                device=device,
            )
        )
        self.bias: Parameter | None = (
            Parameter(empty(out_channels, dtype=dtype, device=device)) if bias else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _resolve_pad(self, x: Tensor) -> tuple[tuple[int], tuple[int]]:
        if self._padding_str == "valid":
            return (0,), (0,)
        if self._padding_str == "same":
            lo, hi = _same_pad_pair(
                x.shape[2], self.kernel_size, self.stride, self.dilation
            )
            return (lo,), (hi,)
        return (self.padding,), (self.padding,)

    def forward(self, x: Tensor) -> Tensor:
        pad_lo, pad_hi = self._resolve_pad(x)
        return _conv_forward_with_mode(
            x,
            self.weight,
            self.bias,
            (self.stride,),
            pad_lo,
            pad_hi,
            (self.dilation,),
            self.groups,
            self.padding_mode,
            conv1d,
        )

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self._padding_str if self._padding_str else self.padding}"
        )
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode!r}"
        return s


class Conv2d(Module):
    """2D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Size2d,
        stride: _Size2d = 1,
        padding: _Size2d | str = 0,
        dilation: _Size2d = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        kh, kw = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = _validate_padding_mode(padding_mode)
        if isinstance(padding, str):
            mode = padding.lower()
            if mode not in {"same", "valid"}:
                raise ValueError(
                    f"string padding must be 'same' or 'valid', got {padding!r}"
                )
            if mode == "same":
                _check_same_supported(self.stride)
            self._padding_str: str | None = mode
            self.padding: tuple[int, int] = (0, 0)
        else:
            self._padding_str = None
            self.padding = _pair(padding)
        self.weight = Parameter(
            empty(
                out_channels, in_channels // groups, kh, kw, dtype=dtype, device=device
            )
        )
        self.bias: Parameter | None = (
            Parameter(empty(out_channels, dtype=dtype, device=device)) if bias else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _resolve_pad(self, x: Tensor) -> tuple[tuple[int, int], tuple[int, int]]:
        if self._padding_str == "valid":
            return (0, 0), (0, 0)
        if self._padding_str == "same":
            kh, kw = self.kernel_size
            sh, sw = self.stride
            dh, dw = self.dilation
            lo_h, hi_h = _same_pad_pair(x.shape[2], kh, sh, dh)
            lo_w, hi_w = _same_pad_pair(x.shape[3], kw, sw, dw)
            return (lo_h, lo_w), (hi_h, hi_w)
        return self.padding, self.padding

    def forward(self, x: Tensor) -> Tensor:
        pad_lo, pad_hi = self._resolve_pad(x)
        return _conv_forward_with_mode(
            x,
            self.weight,
            self.bias,
            self.stride,
            pad_lo,
            pad_hi,
            self.dilation,
            self.groups,
            self.padding_mode,
            conv2d,
        )

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self._padding_str if self._padding_str else self.padding}"
        )
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode!r}"
        return s


class Conv3d(Module):
    """3D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Size3d,
        stride: _Size3d = 1,
        padding: _Size3d | str = 0,
        dilation: _Size3d = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        kd, kh, kw = _triple(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.padding_mode = _validate_padding_mode(padding_mode)
        if isinstance(padding, str):
            mode = padding.lower()
            if mode not in {"same", "valid"}:
                raise ValueError(
                    f"string padding must be 'same' or 'valid', got {padding!r}"
                )
            if mode == "same":
                _check_same_supported(self.stride)
            self._padding_str: str | None = mode
            self.padding: tuple[int, int, int] = (0, 0, 0)
        else:
            self._padding_str = None
            self.padding = _triple(padding)
        self.weight = Parameter(
            empty(
                out_channels,
                in_channels // groups,
                kd,
                kh,
                kw,
                dtype=dtype,
                device=device,
            )
        )
        self.bias: Parameter | None = (
            Parameter(empty(out_channels, dtype=dtype, device=device)) if bias else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _resolve_pad(
        self, x: Tensor
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        if self._padding_str == "valid":
            return (0, 0, 0), (0, 0, 0)
        if self._padding_str == "same":
            kd, kh, kw = self.kernel_size
            sd, sh, sw = self.stride
            dd, dh, dw = self.dilation
            lo_d, hi_d = _same_pad_pair(x.shape[2], kd, sd, dd)
            lo_h, hi_h = _same_pad_pair(x.shape[3], kh, sh, dh)
            lo_w, hi_w = _same_pad_pair(x.shape[4], kw, sw, dw)
            return (lo_d, lo_h, lo_w), (hi_d, hi_h, hi_w)
        return self.padding, self.padding

    def forward(self, x: Tensor) -> Tensor:
        pad_lo, pad_hi = self._resolve_pad(x)
        return _conv_forward_with_mode(
            x,
            self.weight,
            self.bias,
            self.stride,
            pad_lo,
            pad_hi,
            self.dilation,
            self.groups,
            self.padding_mode,
            conv3d,
        )

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self._padding_str if self._padding_str else self.padding}"
        )
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode!r}"
        return s


class ConvTranspose1d(Module):
    """Transposed 1D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        _validate_int_padding(padding, "ConvTranspose1d")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation
        self.weight = Parameter(
            empty(
                in_channels,
                out_channels // groups,
                kernel_size,
                dtype=dtype,
                device=device,
            )
        )
        self.bias: Parameter | None = (
            Parameter(empty(out_channels, dtype=dtype, device=device)) if bias else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return conv_transpose1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}"
        )


class ConvTranspose2d(Module):
    """Transposed 2D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Size2d,
        stride: _Size2d = 1,
        padding: _Size2d = 0,
        output_padding: _Size2d = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _Size2d = 1,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        _validate_int_padding(padding, "ConvTranspose2d")
        kh, kw = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.output_padding = _pair(output_padding)
        self.groups = groups
        self.dilation = _pair(dilation)
        self.weight = Parameter(
            empty(
                in_channels, out_channels // groups, kh, kw, dtype=dtype, device=device
            )
        )
        self.bias: Parameter | None = (
            Parameter(empty(out_channels, dtype=dtype, device=device)) if bias else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return conv_transpose2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}"
        )


class ConvTranspose3d(Module):
    """Transposed 3D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Size3d,
        stride: _Size3d = 1,
        padding: _Size3d = 0,
        output_padding: _Size3d = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _Size3d = 1,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        _validate_int_padding(padding, "ConvTranspose3d")
        kd, kh, kw = _triple(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.output_padding = _triple(output_padding)
        self.groups = groups
        self.dilation = _triple(dilation)
        self.weight = Parameter(
            empty(
                in_channels,
                out_channels // groups,
                kd,
                kh,
                kw,
                dtype=dtype,
                device=device,
            )
        )
        self.bias: Parameter | None = (
            Parameter(empty(out_channels, dtype=dtype, device=device)) if bias else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return conv_transpose3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}"
        )


# ── Lazy convolutions ─────────────────────────────────────────────────────────
#
# Each lazy variant inherits from its eager counterpart so the forward path
# (`_resolve_pad` + `_conv_forward_with_mode`) is reused unchanged.  Parent
# `__init__` is intentionally skipped via `Module.__init__(self)` because it
# requires `in_channels`, which is what we are deferring.  The first forward
# call (or `_load_from_state_dict`) materialises the real `weight` / `bias`
# Parameter objects.


def _init_lazy_conv_weights(weight: Parameter, bias: Parameter | None) -> None:
    """Shared kaiming-uniform initialiser for lazily-built conv weights."""
    init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound: float = 1.0 / math.sqrt(fan_in)
        init.uniform_(bias, -bound, bound)


class LazyConv1d(Conv1d):
    """Conv1d with lazy ``in_channels`` inference from the first input."""

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        Module.__init__(self)
        self.in_channels: int | None = None
        self.out_channels: int = out_channels
        self.kernel_size: int = kernel_size
        self.stride: int = stride
        self.dilation: int = dilation
        self.groups: int = groups
        self.padding_mode: str = _validate_padding_mode(padding_mode)
        self._padding_str: str | None
        if isinstance(padding, str):
            mode: str = padding.lower()
            if mode not in {"same", "valid"}:
                raise ValueError(
                    f"string padding must be 'same' or 'valid', got {padding!r}"
                )
            if mode == "same":
                _check_same_supported((stride,))
            self._padding_str = mode
            self.padding: int = 0
        else:
            self._padding_str = None
            self.padding = padding
        self._has_bias: bool = bias
        self._device: DeviceLike = device
        self._dtype: DTypeLike = dtype
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def _initialize(self, in_channels: int) -> None:
        self.in_channels = in_channels
        self.weight = Parameter(
            empty(
                self.out_channels,
                in_channels // self.groups,
                self.kernel_size,
                dtype=self._dtype,
                device=self._device,
            )
        )
        if self._has_bias:
            self.bias = Parameter(
                empty(self.out_channels, dtype=self._dtype, device=self._device)
            )
        else:
            self.bias = None
        _init_lazy_conv_weights(self.weight, self.bias)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if self.weight is None:
            weight: Tensor | None = state_dict.get(f"{prefix}weight")
            if weight is not None:
                if len(weight.shape) != 3:
                    error_msgs.append(
                        f"LazyConv1d expected 3-D weight at '{prefix}weight', "
                        f"got {tuple(weight.shape)}"
                    )
                    return
                if int(weight.shape[0]) != self.out_channels:
                    error_msgs.append(
                        f"LazyConv1d out_channels mismatch at '{prefix}weight': "
                        f"expected {self.out_channels}, got {int(weight.shape[0])}"
                    )
                    return
                self._dtype = self._dtype or weight.dtype
                self._device = self._device or weight.device
                self._initialize(int(weight.shape[1]) * self.groups)
        from lucid.nn._state_dict import _default_load_from_state_dict

        _default_load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.weight is None:
            self._initialize(int(x.shape[1]))
        return Conv1d.forward(self, x)

    def extra_repr(self) -> str:
        s: str = (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self._padding_str if self._padding_str else self.padding}"
        )
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode!r}"
        return s


class LazyConv2d(Conv2d):
    """Conv2d with lazy ``in_channels`` inference from the first input."""

    def __init__(
        self,
        out_channels: int,
        kernel_size: _Size2d,
        stride: _Size2d = 1,
        padding: _Size2d | str = 0,
        dilation: _Size2d = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        Module.__init__(self)
        self.in_channels: int | None = None
        self.out_channels: int = out_channels
        self.kernel_size: tuple[int, int] = _pair(kernel_size)
        self.stride: tuple[int, int] = _pair(stride)
        self.dilation: tuple[int, int] = _pair(dilation)
        self.groups: int = groups
        self.padding_mode: str = _validate_padding_mode(padding_mode)
        self._padding_str: str | None
        if isinstance(padding, str):
            mode: str = padding.lower()
            if mode not in {"same", "valid"}:
                raise ValueError(
                    f"string padding must be 'same' or 'valid', got {padding!r}"
                )
            if mode == "same":
                _check_same_supported(self.stride)
            self._padding_str = mode
            self.padding: tuple[int, int] = (0, 0)
        else:
            self._padding_str = None
            self.padding = _pair(padding)
        self._has_bias: bool = bias
        self._device: DeviceLike = device
        self._dtype: DTypeLike = dtype
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def _initialize(self, in_channels: int) -> None:
        self.in_channels = in_channels
        kh, kw = self.kernel_size
        self.weight = Parameter(
            empty(
                self.out_channels,
                in_channels // self.groups,
                kh,
                kw,
                dtype=self._dtype,
                device=self._device,
            )
        )
        if self._has_bias:
            self.bias = Parameter(
                empty(self.out_channels, dtype=self._dtype, device=self._device)
            )
        else:
            self.bias = None
        _init_lazy_conv_weights(self.weight, self.bias)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if self.weight is None:
            weight: Tensor | None = state_dict.get(f"{prefix}weight")
            if weight is not None:
                if len(weight.shape) != 4:
                    error_msgs.append(
                        f"LazyConv2d expected 4-D weight at '{prefix}weight', "
                        f"got {tuple(weight.shape)}"
                    )
                    return
                if int(weight.shape[0]) != self.out_channels:
                    error_msgs.append(
                        f"LazyConv2d out_channels mismatch at '{prefix}weight': "
                        f"expected {self.out_channels}, got {int(weight.shape[0])}"
                    )
                    return
                self._dtype = self._dtype or weight.dtype
                self._device = self._device or weight.device
                self._initialize(int(weight.shape[1]) * self.groups)
        from lucid.nn._state_dict import _default_load_from_state_dict

        _default_load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.weight is None:
            self._initialize(int(x.shape[1]))
        return Conv2d.forward(self, x)

    def extra_repr(self) -> str:
        s: str = (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self._padding_str if self._padding_str else self.padding}"
        )
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode!r}"
        return s


class LazyConv3d(Conv3d):
    """Conv3d with lazy ``in_channels`` inference from the first input."""

    def __init__(
        self,
        out_channels: int,
        kernel_size: _Size3d,
        stride: _Size3d = 1,
        padding: _Size3d | str = 0,
        dilation: _Size3d = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        Module.__init__(self)
        self.in_channels: int | None = None
        self.out_channels: int = out_channels
        self.kernel_size: tuple[int, int, int] = _triple(kernel_size)
        self.stride: tuple[int, int, int] = _triple(stride)
        self.dilation: tuple[int, int, int] = _triple(dilation)
        self.groups: int = groups
        self.padding_mode: str = _validate_padding_mode(padding_mode)
        self._padding_str: str | None
        if isinstance(padding, str):
            mode: str = padding.lower()
            if mode not in {"same", "valid"}:
                raise ValueError(
                    f"string padding must be 'same' or 'valid', got {padding!r}"
                )
            if mode == "same":
                _check_same_supported(self.stride)
            self._padding_str = mode
            self.padding: tuple[int, int, int] = (0, 0, 0)
        else:
            self._padding_str = None
            self.padding = _triple(padding)
        self._has_bias: bool = bias
        self._device: DeviceLike = device
        self._dtype: DTypeLike = dtype
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def _initialize(self, in_channels: int) -> None:
        self.in_channels = in_channels
        kd, kh, kw = self.kernel_size
        self.weight = Parameter(
            empty(
                self.out_channels,
                in_channels // self.groups,
                kd,
                kh,
                kw,
                dtype=self._dtype,
                device=self._device,
            )
        )
        if self._has_bias:
            self.bias = Parameter(
                empty(self.out_channels, dtype=self._dtype, device=self._device)
            )
        else:
            self.bias = None
        _init_lazy_conv_weights(self.weight, self.bias)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if self.weight is None:
            weight: Tensor | None = state_dict.get(f"{prefix}weight")
            if weight is not None:
                if len(weight.shape) != 5:
                    error_msgs.append(
                        f"LazyConv3d expected 5-D weight at '{prefix}weight', "
                        f"got {tuple(weight.shape)}"
                    )
                    return
                if int(weight.shape[0]) != self.out_channels:
                    error_msgs.append(
                        f"LazyConv3d out_channels mismatch at '{prefix}weight': "
                        f"expected {self.out_channels}, got {int(weight.shape[0])}"
                    )
                    return
                self._dtype = self._dtype or weight.dtype
                self._device = self._device or weight.device
                self._initialize(int(weight.shape[1]) * self.groups)
        from lucid.nn._state_dict import _default_load_from_state_dict

        _default_load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.weight is None:
            self._initialize(int(x.shape[1]))
        return Conv3d.forward(self, x)

    def extra_repr(self) -> str:
        s: str = (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self._padding_str if self._padding_str else self.padding}"
        )
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode!r}"
        return s


# ── Lazy ConvTranspose ────────────────────────────────────────────────────────
# Weight layout is (in_channels, out_channels // groups, *K).  The lazy
# dimension is ``in_channels`` — the leading axis of the saved weight — so
# materialisation reads ``weight.shape[0]``.


class LazyConvTranspose1d(ConvTranspose1d):
    """ConvTranspose1d with lazy ``in_channels`` inference."""

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        Module.__init__(self)
        _validate_int_padding(padding, "LazyConvTranspose1d")
        self.in_channels: int | None = None
        self.out_channels: int = out_channels
        self.kernel_size: int = kernel_size
        self.stride: int = stride
        self.padding: int = padding
        self.output_padding: int = output_padding
        self.groups: int = groups
        self.dilation: int = dilation
        self._has_bias: bool = bias
        self._device: DeviceLike = device
        self._dtype: DTypeLike = dtype
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def _initialize(self, in_channels: int) -> None:
        self.in_channels = in_channels
        self.weight = Parameter(
            empty(
                in_channels,
                self.out_channels // self.groups,
                self.kernel_size,
                dtype=self._dtype,
                device=self._device,
            )
        )
        if self._has_bias:
            self.bias = Parameter(
                empty(self.out_channels, dtype=self._dtype, device=self._device)
            )
        else:
            self.bias = None
        _init_lazy_conv_weights(self.weight, self.bias)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if self.weight is None:
            weight: Tensor | None = state_dict.get(f"{prefix}weight")
            if weight is not None:
                if len(weight.shape) != 3:
                    error_msgs.append(
                        f"LazyConvTranspose1d expected 3-D weight at '{prefix}weight', "
                        f"got {tuple(weight.shape)}"
                    )
                    return
                if int(weight.shape[1]) != self.out_channels // self.groups:
                    error_msgs.append(
                        f"LazyConvTranspose1d out_channels mismatch at '{prefix}weight': "
                        f"expected {self.out_channels // self.groups}, "
                        f"got {int(weight.shape[1])}"
                    )
                    return
                self._dtype = self._dtype or weight.dtype
                self._device = self._device or weight.device
                self._initialize(int(weight.shape[0]))
        from lucid.nn._state_dict import _default_load_from_state_dict

        _default_load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.weight is None:
            self._initialize(int(x.shape[1]))
        return ConvTranspose1d.forward(self, x)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}"
        )


class LazyConvTranspose2d(ConvTranspose2d):
    """ConvTranspose2d with lazy ``in_channels`` inference."""

    def __init__(
        self,
        out_channels: int,
        kernel_size: _Size2d,
        stride: _Size2d = 1,
        padding: _Size2d = 0,
        output_padding: _Size2d = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _Size2d = 1,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        Module.__init__(self)
        _validate_int_padding(padding, "LazyConvTranspose2d")
        self.in_channels: int | None = None
        self.out_channels: int = out_channels
        self.kernel_size: tuple[int, int] = _pair(kernel_size)
        self.stride: tuple[int, int] = _pair(stride)
        self.padding: tuple[int, int] = _pair(padding)
        self.output_padding: tuple[int, int] = _pair(output_padding)
        self.groups: int = groups
        self.dilation: tuple[int, int] = _pair(dilation)
        self._has_bias: bool = bias
        self._device: DeviceLike = device
        self._dtype: DTypeLike = dtype
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def _initialize(self, in_channels: int) -> None:
        self.in_channels = in_channels
        kh, kw = self.kernel_size
        self.weight = Parameter(
            empty(
                in_channels,
                self.out_channels // self.groups,
                kh,
                kw,
                dtype=self._dtype,
                device=self._device,
            )
        )
        if self._has_bias:
            self.bias = Parameter(
                empty(self.out_channels, dtype=self._dtype, device=self._device)
            )
        else:
            self.bias = None
        _init_lazy_conv_weights(self.weight, self.bias)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if self.weight is None:
            weight: Tensor | None = state_dict.get(f"{prefix}weight")
            if weight is not None:
                if len(weight.shape) != 4:
                    error_msgs.append(
                        f"LazyConvTranspose2d expected 4-D weight at '{prefix}weight', "
                        f"got {tuple(weight.shape)}"
                    )
                    return
                if int(weight.shape[1]) != self.out_channels // self.groups:
                    error_msgs.append(
                        f"LazyConvTranspose2d out_channels mismatch at '{prefix}weight': "
                        f"expected {self.out_channels // self.groups}, "
                        f"got {int(weight.shape[1])}"
                    )
                    return
                self._dtype = self._dtype or weight.dtype
                self._device = self._device or weight.device
                self._initialize(int(weight.shape[0]))
        from lucid.nn._state_dict import _default_load_from_state_dict

        _default_load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.weight is None:
            self._initialize(int(x.shape[1]))
        return ConvTranspose2d.forward(self, x)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}"
        )


class LazyConvTranspose3d(ConvTranspose3d):
    """ConvTranspose3d with lazy ``in_channels`` inference."""

    def __init__(
        self,
        out_channels: int,
        kernel_size: _Size3d,
        stride: _Size3d = 1,
        padding: _Size3d = 0,
        output_padding: _Size3d = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _Size3d = 1,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        Module.__init__(self)
        _validate_int_padding(padding, "LazyConvTranspose3d")
        self.in_channels: int | None = None
        self.out_channels: int = out_channels
        self.kernel_size: tuple[int, int, int] = _triple(kernel_size)
        self.stride: tuple[int, int, int] = _triple(stride)
        self.padding: tuple[int, int, int] = _triple(padding)
        self.output_padding: tuple[int, int, int] = _triple(output_padding)
        self.groups: int = groups
        self.dilation: tuple[int, int, int] = _triple(dilation)
        self._has_bias: bool = bias
        self._device: DeviceLike = device
        self._dtype: DTypeLike = dtype
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def _initialize(self, in_channels: int) -> None:
        self.in_channels = in_channels
        kd, kh, kw = self.kernel_size
        self.weight = Parameter(
            empty(
                in_channels,
                self.out_channels // self.groups,
                kd,
                kh,
                kw,
                dtype=self._dtype,
                device=self._device,
            )
        )
        if self._has_bias:
            self.bias = Parameter(
                empty(self.out_channels, dtype=self._dtype, device=self._device)
            )
        else:
            self.bias = None
        _init_lazy_conv_weights(self.weight, self.bias)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if self.weight is None:
            weight: Tensor | None = state_dict.get(f"{prefix}weight")
            if weight is not None:
                if len(weight.shape) != 5:
                    error_msgs.append(
                        f"LazyConvTranspose3d expected 5-D weight at '{prefix}weight', "
                        f"got {tuple(weight.shape)}"
                    )
                    return
                if int(weight.shape[1]) != self.out_channels // self.groups:
                    error_msgs.append(
                        f"LazyConvTranspose3d out_channels mismatch at '{prefix}weight': "
                        f"expected {self.out_channels // self.groups}, "
                        f"got {int(weight.shape[1])}"
                    )
                    return
                self._dtype = self._dtype or weight.dtype
                self._device = self._device or weight.device
                self._initialize(int(weight.shape[0]))
        from lucid.nn._state_dict import _default_load_from_state_dict

        _default_load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.weight is None:
            self._initialize(int(x.shape[1]))
        return ConvTranspose3d.forward(self, x)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}"
        )
