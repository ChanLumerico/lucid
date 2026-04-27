"""
lucid.nn.modules.conv — convolution / transposed-convolution / unfold modules.

Plain `Conv{1,2,3}d` and `ConvTranspose{1,2,3}d` are thin wrappers
over the matching functional ops. `ConstrainedConv{1,2,3}d` adds
weight constraints (non-negativity, sum-to-one, unit-l2, etc.) that
can be applied either lazily during forward or eagerly via
`project_()`.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


__all__ = [
    "Unfold",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "ConstrainedConv1d",
    "ConstrainedConv2d",
    "ConstrainedConv3d",
]


def _single_to_tuple(value: Any, times: int) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    return (value,) * times


def _check_dim(input_: Tensor, dim: int) -> None:
    if input_.ndim != dim:
        raise ValueError(
            f"Expected input with {dim} dimensions, got {input_.ndim}."
        )


class Unfold(nn.Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...],
        padding: int | tuple[int, ...],
        dilation: int | tuple[int, ...] = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = _single_to_tuple(kernel_size, 2)
        self.stride = _single_to_tuple(stride, 2)
        self.padding = _single_to_tuple(padding, 2)
        self.dilation = _single_to_tuple(dilation, 2)

    def forward(self, input_: Tensor) -> Tensor:
        if input_.ndim != 4:
            raise ValueError("'nn.Unfold' only supports 4D-tensors. (i.e. images)")

        unfolded = F.unfold(
            input_, self.kernel_size, self.stride, self.padding, self.dilation
        )

        N, C, *spatial_dims = input_.shape
        out_dims = []
        for in_dim, p, d, k, s in zip(
            spatial_dims, self.padding, self.dilation, self.kernel_size, self.stride
        ):
            eff_k = d * (k - 1) + 1
            out_dim = math.floor((in_dim + 2 * p - eff_k) / s + 1)
            out_dims.append(out_dim)

        L = math.prod(out_dims)
        Ck = C * math.prod(self.kernel_size)
        return unfolded.reshape(N, Ck, L)


_PaddingStr = Literal["same", "valid"]


class _ConvNd(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...],
        padding: _PaddingStr | int | tuple[int, ...],
        dilation: int | tuple[int, ...],
        groups: int,
        bias: bool,
        *,
        D: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.kernel_size = _single_to_tuple(kernel_size, D)
        self.stride = _single_to_tuple(stride, D)
        self.dilation = _single_to_tuple(dilation, D)

        if isinstance(padding, str):
            if padding == "same":
                self.padding = tuple(
                    (self.dilation[i] * (self.kernel_size[i] - 1)) // 2
                    for i in range(D)
                )
            elif padding == "valid":
                self.padding = (0,) * D
            else:
                raise ValueError(f"Unknown padding string: {padding}")
        else:
            self.padding = _single_to_tuple(padding, D)

        if groups <= 0:
            raise ValueError("groups must be a positive integer.")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups.")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups.")

        self.weight = nn.Parameter(
            lucid.empty(out_channels, in_channels // groups, *self.kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(lucid.empty((out_channels,)))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform(self.weight)

        if self.bias is not None:
            fan_in, _ = nn.init._dist._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += f", padding={self.padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += f", dilation={self.dilation}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.bias is None:
            s += ", bias=False"

        return s


_ConstrainedMode = Literal[
    "none",
    "nonneg",
    "sum_to_one",
    "zero_mean",
    "nonneg_sum1",
    "unit_l2",
    "max_l2",
    "fixed_center",
]
_ConstraintEnforce = Literal["forward", "post_step"]


class _ConstrainedConvNd(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...],
        padding: _PaddingStr | int | tuple[int, ...],
        dilation: int | tuple[int, ...],
        groups: int,
        bias: bool,
        *,
        constraint: _ConstrainedMode = "none",
        enforce: _ConstraintEnforce = "forward",
        eps: float = 1e-12,
        max_l2: float | None = None,
        center_value: float = -1.0,
        neighbor_sum: float = 1.0,
        D: int,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            D=D,
        )

        self.constraint = constraint
        self.enforce = enforce
        self.eps = eps
        self.max_l2 = max_l2
        self.center_value = center_value
        self.neighbor_sum = neighbor_sum
        self._reduce_axis = tuple(range(-D, 0))

        valid_constraints = {
            "none",
            "nonneg",
            "sum_to_one",
            "zero_mean",
            "nonneg_sum1",
            "unit_l2",
            "max_l2",
            "fixed_center",
        }
        if constraint not in valid_constraints:
            raise ValueError(f"Unknown constraint: {constraint}")
        if enforce not in {"forward", "post_step"}:
            raise ValueError(f"Unknown enforce mode: {enforce}")
        if constraint == "max_l2" and (max_l2 is None or max_l2 <= 0):
            raise ValueError("`max_l2` must be positive when constraint='max_l2'.")

        if constraint == "fixed_center":
            if any(k % 2 == 0 for k in self.kernel_size):
                raise ValueError(
                    "fixed_center requires odd kernel sizes for all spatial dims."
                )

            # Build a one-hot mask whose only nonzero entry is the kernel center.
            import numpy as np

            mask = np.zeros((1, 1, *self.kernel_size), dtype=np.float32)
            center_idx = (0, 0, *[k // 2 for k in self.kernel_size])
            mask[center_idx] = 1.0
            center_mask = Tensor(mask, device=self.device)

            self.register_buffer("_center_mask", center_mask)
            self.register_buffer("_neighbor_mask", Tensor(1.0 - mask, device=self.device))
        else:
            self.register_buffer("_center_mask", None)
            self.register_buffer("_neighbor_mask", None)

        self._center_mask: nn.Buffer | None
        self._neighbor_mask: nn.Buffer | None

    def _sum_spatial(self, w: Tensor) -> Tensor:
        return lucid.sum(w, axis=self._reduce_axis, keepdims=True)

    def _normalize_sum(self, w: Tensor, target_sum: float) -> Tensor:
        return w / (self._sum_spatial(w) + self.eps) * target_sum

    def _l2_spatial(self, w: Tensor) -> Tensor:
        return lucid.sqrt(self._sum_spatial(w * w) + self.eps)

    def _apply_constraint(self, w: Tensor) -> Tensor:
        if self.constraint == "none":
            return w
        if self.constraint == "nonneg":
            return F.relu(w)
        if self.constraint == "sum_to_one":
            return self._normalize_sum(w, 1.0)
        if self.constraint == "zero_mean":
            return w - lucid.mean(w, axis=self._reduce_axis, keepdims=True)
        if self.constraint == "nonneg_sum1":
            return self._normalize_sum(F.relu(w), 1.0)
        if self.constraint == "unit_l2":
            return w / self._l2_spatial(w)
        if self.constraint == "max_l2":
            ratio = self.max_l2 / self._l2_spatial(w)
            return w * lucid.clip(ratio, min_value=None, max_value=1.0)
        if self.constraint == "fixed_center":
            center = self._center_mask * self.center_value
            neighbors = w * self._neighbor_mask
            neighbors = self._normalize_sum(neighbors, self.neighbor_sum)
            return neighbors + center

        raise RuntimeError(f"Unhandled constraint: {self.constraint}")

    def _constrained_weight(self) -> Tensor:
        if self.enforce == "forward":
            return self._apply_constraint(self.weight)
        return self.weight

    def project_(self) -> "_ConstrainedConvNd":
        projected = self._apply_constraint(self.weight)
        # Swap impl in place so Parameter identity (and any optimizer state
        # keyed on it) survives the projection.
        self.weight._impl = projected._impl
        return self

    def extra_repr(self) -> str:
        s = super().extra_repr()
        s += f", constraint={self.constraint}, enforce={self.enforce}, eps={self.eps}"
        if self.constraint == "max_l2":
            s += f", max_l2={self.max_l2}"
        if self.constraint == "fixed_center":
            s += (
                f", center_value={self.center_value}, "
                f"neighbor_sum={self.neighbor_sum}"
            )
        return s


class Conv1d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: _PaddingStr | int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            D=1,
        )

    def _conv_forward(
        self, input_: Tensor, weight: Tensor, bias: Tensor | None
    ) -> Tensor:
        _check_dim(input_, dim=3)
        return F.conv1d(
            input_, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input_: Tensor) -> Tensor:
        return self._conv_forward(input_, self.weight, self.bias)


class Conv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: _PaddingStr | int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            D=2,
        )

    def _conv_forward(
        self, input_: Tensor, weight: Tensor, bias: Tensor | None
    ) -> Tensor:
        _check_dim(input_, dim=4)
        return F.conv2d(
            input_, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input_: Tensor) -> Tensor:
        return self._conv_forward(input_, self.weight, self.bias)


class Conv3d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: _PaddingStr | int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            D=3,
        )

    def _conv_forward(
        self, input_: Tensor, weight: Tensor, bias: Tensor | None
    ) -> Tensor:
        _check_dim(input_, dim=5)
        return F.conv3d(
            input_, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input_: Tensor) -> Tensor:
        return self._conv_forward(input_, self.weight, self.bias)


class ConstrainedConv1d(_ConstrainedConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: _PaddingStr | int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        *,
        constraint: _ConstrainedMode = "none",
        enforce: _ConstraintEnforce = "forward",
        eps: float = 1e-12,
        max_l2: float | None = None,
        center_value: float = -1.0,
        neighbor_sum: float = 1.0,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            constraint=constraint,
            enforce=enforce,
            eps=eps,
            max_l2=max_l2,
            center_value=center_value,
            neighbor_sum=neighbor_sum,
            D=1,
        )

    def _conv_forward(
        self, input_: Tensor, weight: Tensor, bias: Tensor | None
    ) -> Tensor:
        _check_dim(input_, dim=3)
        return F.conv1d(
            input_, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input_: Tensor) -> Tensor:
        return self._conv_forward(input_, self._constrained_weight(), self.bias)


class ConstrainedConv2d(_ConstrainedConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: _PaddingStr | int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        *,
        constraint: _ConstrainedMode = "none",
        enforce: _ConstraintEnforce = "forward",
        eps: float = 1e-12,
        max_l2: float | None = None,
        center_value: float = -1.0,
        neighbor_sum: float = 1.0,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            constraint=constraint,
            enforce=enforce,
            eps=eps,
            max_l2=max_l2,
            center_value=center_value,
            neighbor_sum=neighbor_sum,
            D=2,
        )

    def _conv_forward(
        self, input_: Tensor, weight: Tensor, bias: Tensor | None
    ) -> Tensor:
        _check_dim(input_, dim=4)
        return F.conv2d(
            input_, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input_: Tensor) -> Tensor:
        return self._conv_forward(input_, self._constrained_weight(), self.bias)


class ConstrainedConv3d(_ConstrainedConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: _PaddingStr | int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        *,
        constraint: _ConstrainedMode = "none",
        enforce: _ConstraintEnforce = "forward",
        eps: float = 1e-12,
        max_l2: float | None = None,
        center_value: float = -1.0,
        neighbor_sum: float = 1.0,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            constraint=constraint,
            enforce=enforce,
            eps=eps,
            max_l2=max_l2,
            center_value=center_value,
            neighbor_sum=neighbor_sum,
            D=3,
        )

    def _conv_forward(
        self, input_: Tensor, weight: Tensor, bias: Tensor | None
    ) -> Tensor:
        _check_dim(input_, dim=5)
        return F.conv3d(
            input_, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input_: Tensor) -> Tensor:
        return self._conv_forward(input_, self._constrained_weight(), self.bias)


class _ConvTransposeNd(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...],
        padding: _PaddingStr | int | tuple[int, ...],
        output_padding: int | tuple[int, ...],
        dilation: int | tuple[int, ...],
        groups: int,
        bias: bool,
        *,
        D: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.kernel_size = _single_to_tuple(kernel_size, D)
        self.stride = _single_to_tuple(stride, D)
        self.dilation = _single_to_tuple(dilation, D)
        self.output_padding = _single_to_tuple(output_padding, D)

        if isinstance(padding, str):
            if padding == "same":
                self.padding = tuple(
                    (self.dilation[i] * (self.kernel_size[i] - 1)) // 2
                    for i in range(D)
                )
            elif padding == "valid":
                self.padding = (0,) * D
            else:
                raise ValueError(f"Unknown padding string: {padding}")
        else:
            self.padding = _single_to_tuple(padding, D)

        if groups <= 0:
            raise ValueError("groups must be a positive integer.")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups.")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups.")

        self.weight = nn.Parameter(
            lucid.empty(in_channels, out_channels // groups, *self.kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(lucid.empty((out_channels,)))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._dist._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += f", padding={self.padding}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += f", output_padding={self.output_padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += f", dilation={self.dilation}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.bias is None:
            s += ", bias=False"

        return s


class ConvTranspose1d(_ConvTransposeNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: _PaddingStr | int | tuple[int, ...] = 0,
        output_padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            bias,
            D=1,
        )

    def _conv_transpose_forward(
        self, input_: Tensor, weight: Tensor, bias: Tensor | None
    ) -> Tensor:
        _check_dim(input_, dim=3)
        return F.conv_transpose1d(
            input_,
            weight,
            bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )

    def forward(self, input_: Tensor) -> Tensor:
        return self._conv_transpose_forward(input_, self.weight, self.bias)


class ConvTranspose2d(_ConvTransposeNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: _PaddingStr | int | tuple[int, ...] = 0,
        output_padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            bias,
            D=2,
        )

    def _conv_transpose_forward(
        self, input_: Tensor, weight: Tensor, bias: Tensor | None
    ) -> Tensor:
        _check_dim(input_, dim=4)
        return F.conv_transpose2d(
            input_,
            weight,
            bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )

    def forward(self, input_: Tensor) -> Tensor:
        return self._conv_transpose_forward(input_, self.weight, self.bias)


class ConvTranspose3d(_ConvTransposeNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: _PaddingStr | int | tuple[int, ...] = 0,
        output_padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            bias,
            D=3,
        )

    def _conv_transpose_forward(
        self, input_: Tensor, weight: Tensor, bias: Tensor | None
    ) -> Tensor:
        _check_dim(input_, dim=5)
        return F.conv_transpose3d(
            input_,
            weight,
            bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )

    def forward(self, input_: Tensor) -> Tensor:
        return self._conv_transpose_forward(input_, self.weight, self.bias)
