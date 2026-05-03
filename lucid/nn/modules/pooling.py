"""
Pooling modules.
"""

from typing import Any
from lucid.nn.module import Module
# F imported lazily inside forward()


class MaxPool1d(Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0,
                 dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.max_pool1d(x, self.kernel_size, self.stride, self.padding, self.dilation, False, self.ceil_mode)


class MaxPool2d(Module):
    def __init__(self, kernel_size: "int | tuple[int,int]", stride: "int | tuple[int,int] | None" = None,
                 padding: "int | tuple[int,int]" = 0, dilation: "int | tuple[int,int]" = 1,
                 return_indices: bool = False, ceil_mode: bool = False) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation, False, self.ceil_mode)


class AvgPool1d(Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0,
                 ceil_mode: bool = False, count_include_pad: bool = True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.avg_pool1d(x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)


class AvgPool2d(Module):
    def __init__(self, kernel_size: "int | tuple[int,int]", stride: "int | tuple[int,int] | None" = None,
                 padding: "int | tuple[int,int]" = 0, ceil_mode: bool = False,
                 count_include_pad: bool = True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)


class AdaptiveAvgPool1d(Module):
    def __init__(self, output_size: "int | tuple[int,...]") -> None:
        super().__init__()
        self.output_size = output_size
    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.adaptive_avg_pool1d(x, self.output_size)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size: "int | tuple[int,int]") -> None:
        super().__init__()
        self.output_size = output_size
    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.adaptive_avg_pool2d(x, self.output_size)


class AdaptiveMaxPool2d(Module):
    def __init__(self, output_size: "int | tuple[int,int]", return_indices: bool = False) -> None:
        super().__init__()
        self.output_size = output_size
    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.adaptive_max_pool2d(x, self.output_size)
