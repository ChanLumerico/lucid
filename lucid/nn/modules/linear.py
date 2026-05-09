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
from lucid.nn.functional.linear import linear, bilinear, fused_linear_relu, fused_linear_gelu
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


class FusedLinear(Module):
    """Linear layer with a fused activation (Phase 19 FusionPass).

    ``y = activation(x @ weight.T + bias)``

    In **inference mode** the computation is performed as a single
    BLAS + Accelerate pass, bypassing the intermediate activation
    tensor allocation.  In **training mode** it falls back to standard
    unfused ops so gradients are computed correctly.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    activation : str
        Fused activation function.  Supported values:

        * ``'relu'`` (default) — uses ``_fused_linear_relu``
        * ``'gelu'``           — uses ``_fused_linear_gelu`` (tanh approx)
    bias : bool
        If ``True`` (default), add a learnable bias.
    device : optional
        Device for initial parameters.
    dtype : optional
        Dtype for initial parameters.

    Examples
    --------
    >>> m = nn.FusedLinear(64, 256, activation='relu')
    >>> x = lucid.randn(4, 64)
    >>> with lucid.no_grad():
    ...     y = m(x)   # single-pass fused kernel on CPU
    >>> y.shape
    (4, 256)
    """

    _SUPPORTED = frozenset({"relu", "gelu"})

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "relu",
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        if activation not in self._SUPPORTED:
            raise ValueError(
                f"FusedLinear: unsupported activation '{activation}'. "
                f"Choose from {sorted(self._SUPPORTED)}."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        import math as _math
        import lucid.nn.init as _init

        self.weight = Parameter(empty(out_features, in_features, dtype=dtype, device=device))
        if bias:
            self.bias: Parameter | None = Parameter(
                empty(out_features, dtype=dtype, device=device)
            )
        else:
            self.bias = None

        _init.kaiming_uniform_(self.weight, a=_math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / _math.sqrt(fan_in) if fan_in > 0 else 0.0
            _init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if self.bias is None:
            # No fused kernel for bias=False — fall back to standard ops.
            import lucid.nn.functional as F
            act = F.relu if self.activation == "relu" else F.gelu
            return act(linear(x, self.weight, None))

        if self.activation == "relu":
            return fused_linear_relu(x, self.weight, self.bias)
        return fused_linear_gelu(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"activation={self.activation!r}, bias={self.bias is not None}"
        )


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

    def _load_from_state_dict(
        self,
        state_dict: StateDict,
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # If still uninitialized, materialize from the checkpoint shape first.
        if self.weight is None:
            weight = state_dict.get(f"{prefix}weight")
            if weight is not None:
                if len(weight.shape) != 2:
                    error_msgs.append(
                        f"LazyLinear expected 2-D weight at '{prefix}weight', "
                        f"got {tuple(weight.shape)}"
                    )
                    return
                if int(weight.shape[0]) != self.out_features:
                    error_msgs.append(
                        f"LazyLinear out_features mismatch at '{prefix}weight': "
                        f"expected {self.out_features}, got {int(weight.shape[0])}"
                    )
                    return
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
                if self._has_bias:
                    bias = state_dict.get(f"{prefix}bias")
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
        # Delegate the actual copy / shape-check to the default loader.
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
            self._initialize(x.shape[-1])
        return linear(x, self.weight, self.bias)  # type: ignore[arg-type]

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self._has_bias}"
        )
