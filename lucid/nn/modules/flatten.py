"""
Flatten, Unflatten, Unfold, and Fold modules.
"""

from lucid._tensor.tensor import Tensor
from lucid._types import _Size2d
from lucid.nn.module import Module
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap


class Flatten(Module):
    """Flatten consecutive dimensions into one."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        return _wrap(_C_engine.flatten(_unwrap(x), self.start_dim, self.end_dim))

    def extra_repr(self) -> str:
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"


class Unflatten(Module):
    """Unflatten one dimension into multiple dimensions."""

    def __init__(self, dim: int, unflattened_size: tuple[int, ...]) -> None:
        super().__init__()
        self.dim = dim
        self.unflattened_size = unflattened_size

    def forward(self, x: Tensor) -> Tensor:
        shape = list(x.shape)
        new_shape = (
            shape[: self.dim] + list(self.unflattened_size) + shape[self.dim + 1 :]
        )
        return _wrap(_C_engine.reshape(_unwrap(x), new_shape))

    def extra_repr(self) -> str:
        return f"dim={self.dim}, unflattened_size={self.unflattened_size}"


class Unfold(Module):
    """Extract sliding local blocks from a 4-D input (N, C, H, W).

    This is the module wrapper for ``nn.functional.unfold``.  The output
    has shape ``(N, C*kH*kW, L)`` where *L* is the number of blocks.
    """

    def __init__(
        self,
        kernel_size: _Size2d,
        dilation: _Size2d = 1,
        padding: _Size2d = 0,
        stride: _Size2d = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        from lucid.nn.functional.sampling import unfold as _unfold

        return _unfold(x, self.kernel_size, self.dilation, self.padding, self.stride)

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            f"padding={self.padding}, stride={self.stride}"
        )


class Fold(Module):
    """Combine an array of sliding local blocks into a large containing tensor.

    This is the inverse of ``Unfold``.  The output shape is
    ``(N, C, output_size[0], output_size[1])``.

    Backed by the C++ fold (col2im) op via F.fold.
    """

    def __init__(
        self,
        output_size: _Size2d,
        kernel_size: _Size2d,
        dilation: _Size2d = 1,
        padding: _Size2d = 0,
        stride: _Size2d = 1,
    ) -> None:
        super().__init__()
        self.output_size = (
            output_size
            if isinstance(output_size, tuple)
            else (output_size, output_size)
        )
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.dilation = (
            dilation if isinstance(dilation, tuple) else (dilation, dilation)
        )
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)

    def forward(self, x: Tensor) -> Tensor:
        from lucid.nn.functional.sampling import fold as _fold

        oH, oW = self.output_size
        kH, kW = self.kernel_size
        dH, dW = self.dilation
        pH, pW = self.padding
        sH, sW = self.stride
        return _fold(
            x, (oH, oW), (kH, kW), dilation=(dH, dW), padding=(pH, pW), stride=(sH, sW)
        )

    def extra_repr(self) -> str:
        return (
            f"output_size={self.output_size}, kernel_size={self.kernel_size}, "
            f"dilation={self.dilation}, padding={self.padding}, stride={self.stride}"
        )
