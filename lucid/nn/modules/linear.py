"""
Linear and related fully-connected layers.
"""

import math
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
import lucid.nn.init as init
from lucid._factories.creation import empty
from lucid.nn.functional.linear import linear, bilinear
from lucid._types import StateDict


class Linear(Module):
    """Apply a linear transformation: :math:`y = xW^T + b`.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, optional
        If ``True`` (default), add a learnable bias term.
    device : optional
        Device for the initial parameters.
    dtype : optional
        Data type for the initial parameters.

    Attributes
    ----------
    weight : Parameter
        Shape ``(out_features, in_features)``. Initialized with Kaiming uniform.
    bias : Parameter or None
        Shape ``(out_features,)``. ``None`` when ``bias=False``.

    Examples
    --------
    >>> m = nn.Linear(20, 10)
    >>> x = lucid.randn(4, 20)
    >>> m(x).shape
    (4, 10)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            empty(out_features, in_features, dtype=dtype, device=device)
        )
        if bias:
            self.bias: Parameter | None = Parameter(
                empty(out_features, dtype=dtype, device=device)
            )
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weight with Kaiming uniform and bias with uniform fan_in bound."""
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Identity(Module):
    """Pass-through layer that returns its input unchanged."""

    def forward(self, x: Tensor) -> Tensor:
        return x


class Bilinear(Module):
    """Bilinear layer: y = x1 @ W @ x2.T + bias."""

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = Parameter(
            empty(out_features, in1_features, in2_features, dtype=dtype, device=device)
        )
        self.bias: Parameter | None = (
            Parameter(empty(out_features, dtype=dtype, device=device)) if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize with Kaiming uniform."""
        import math

        bound = 1.0 / math.sqrt(self.weight.shape[1])
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return bilinear(x1, x2, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in1_features={self.in1_features}, in2_features={self.in2_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )


class LazyLinear(Module):
    """Linear layer with lazy weight initialization.

    The ``in_features`` dimension is inferred from the first forward call.
    Until then, :attr:`weight` and :attr:`bias` are uninitialized placeholders.

    Parameters
    ----------
    out_features : int
        Size of each output sample.
    bias : bool
        If True (default), add a learnable bias.
    """

    def __init__(
        self,
        out_features: int,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.out_features = out_features
        self.in_features: int | None = None
        self._has_bias = bias
        self._device = device
        self._dtype = dtype
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def _initialize(self, in_features: int) -> None:
        self.in_features = in_features
        self.weight = Parameter(
            empty(
                self.out_features, in_features, dtype=self._dtype, device=self._device
            )
        )
        if self._has_bias:
            self.bias = Parameter(
                empty(self.out_features, dtype=self._dtype, device=self._device)
            )
        else:
            self.bias = None
        bound = 1.0 / math.sqrt(in_features)
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def _initialize_from_state_dict(
        self,
        state_dict: StateDict,
        prefix: str,
    ) -> None:
        if self.weight is not None:
            return
        weight = state_dict.get(f"{prefix}weight")
        if weight is None:
            return
        if len(weight.shape) != 2:
            raise RuntimeError(
                f"LazyLinear expected 2-D weight in state_dict, got {weight.shape}"
            )
        if int(weight.shape[0]) != self.out_features:
            raise RuntimeError(
                "LazyLinear out_features mismatch: "
                f"expected {self.out_features}, got {int(weight.shape[0])}"
            )

        self.in_features = int(weight.shape[1])
        param_dtype = self._dtype or weight.dtype
        param_device = self._device or weight.device
        self.weight = Parameter(
            empty(
                self.out_features,
                self.in_features,
                dtype=param_dtype,
                device=param_device,
            )
        )

        bias = state_dict.get(f"{prefix}bias")
        if self._has_bias:
            if bias is not None and len(bias.shape) != 1:
                raise RuntimeError(
                    f"LazyLinear expected 1-D bias in state_dict, got {bias.shape}"
                )
            bias_dtype = self._dtype or (
                bias.dtype if bias is not None else weight.dtype
            )
            bias_device = self._device or (
                bias.device if bias is not None else weight.device
            )
            self.bias = Parameter(
                empty(self.out_features, dtype=bias_dtype, device=bias_device)
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        if self.weight is None:
            self._initialize(x.shape[-1])
        return linear(x, self.weight, self.bias)  # type: ignore[arg-type]

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self._has_bias}"
        )
