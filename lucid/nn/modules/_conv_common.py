"""Helpers the convolution and transposed-convolution modules share."""

import math
from typing import ClassVar, override

from lucid._factories.creation import empty
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike, _Size2d, _Size3d
import lucid.nn.init as init
from lucid.nn.module import Module
from lucid.nn.modules._lazy import fill as _fill
from lucid.nn.parameter import Parameter, UninitializedParameter


def _pair(v: _Size2d) -> tuple[int, int]:
    return (v, v) if isinstance(v, int) else (v[0], v[1])


def _triple(v: _Size3d) -> tuple[int, int, int]:
    return (v, v, v) if isinstance(v, int) else (v[0], v[1], v[2])


def _in_channels(x: Tensor, n: int) -> int:
    """The channel count of a batched or unbatched ``n``-D conv input."""
    return int(x.shape[0 if x.ndim == n + 1 else 1])


def _init_conv_weights(weight: Parameter, bias: Parameter | None) -> None:
    """Kaiming-uniform weight; a bias bounded by the weight's fan-in."""
    init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound: float = 1.0 / math.sqrt(fan_in)
        init.uniform_(bias, -bound, bound)


def _conv_parameters(
    shape: tuple[int, ...],
    out_channels: int,
    bias: bool,
    device: DeviceLike,
    dtype: DTypeLike,
) -> tuple[Parameter, Parameter | None]:
    """A convolution's ``weight`` of ``shape`` and its bias, initialised."""
    weight = Parameter(empty(*shape, dtype=dtype, device=device))
    b = Parameter(empty(out_channels, dtype=dtype, device=device)) if bias else None
    _init_conv_weights(weight, b)
    return weight, b


class _LazyConvMixin(Module):
    """What the six lazy convolutions share: deferring the weight to the first input.

    Each lazy class puts this before its eager counterpart, sets
    ``_lazy_rank`` (and ``_lazy_transposed`` for a transposed one), and
    calls :meth:`_defer` last in its ``__init__``.  The weight layout is
    ``(out, in // groups, *K)``, or ``(in, out // groups, *K)`` transposed.
    """

    _lazy_rank: ClassVar[int]
    _lazy_transposed: ClassVar[bool] = False
    out_channels: int
    groups: int
    _has_bias: bool
    _device: DeviceLike
    _dtype: DTypeLike

    def _defer(self, bias: bool, device: DeviceLike, dtype: DTypeLike) -> None:
        """Leave ``in_channels`` unknown and the parameters as placeholders."""
        # Through ``Module.__setattr__`` and not ``self.x = ...``: the eager
        # base declares ``in_channels: int`` / ``bias``, and an assignment
        # here would declare them a second time, differently.
        Module.__setattr__(self, "in_channels", None)
        self._has_bias = bias
        self._device = device
        self._dtype = dtype
        # Placeholders, not ``None``: an optimizer built before the first
        # forward must be handed objects that survive being filled in.
        self.register_parameter("weight", UninitializedParameter())
        self.register_parameter("bias", UninitializedParameter() if bias else None)

    def _kernel(self) -> tuple[int, ...]:
        """The kernel's extent along each spatial axis."""
        raise NotImplementedError

    def _initialize(self, in_channels: int) -> None:
        """Allocate and initialise ``weight`` / ``bias`` for ``in_channels``."""
        Module.__setattr__(self, "in_channels", in_channels)
        if self._lazy_transposed:
            lead = (in_channels, self.out_channels // self.groups)
        else:
            lead = (self.out_channels, in_channels // self.groups)
        weight, bias = _conv_parameters(
            (*lead, *self._kernel()),
            self.out_channels,
            self._has_bias,
            self._device,
            self._dtype,
        )
        # ``fill`` and not assignment — the placeholder object is what
        # anything that read ``parameters()`` early is holding on to.
        _fill(self, "weight", weight)
        if bias is not None:
            _fill(self, "bias", bias)
        else:
            Module.__setattr__(self, "bias", None)

    def _materialize_from(self, x: Tensor) -> None:
        """Size ``in_channels`` from the first input ``x``."""
        if self.in_channels is None:
            self._initialize(_in_channels(x, self._lazy_rank))

    def _sized_by(self, weight: Tensor, prefix: str, error_msgs: list[str]) -> bool:
        """Size the layer from a checkpoint's ``weight``; ``False`` if it cannot be."""
        name = type(self).__name__
        if len(weight.shape) != self._lazy_rank + 2:
            error_msgs.append(
                f"{name} expected {self._lazy_rank + 2}-D weight at '{prefix}weight', "
                f"got {tuple(weight.shape)}"
            )
            return False
        out_axis, expected = (
            (1, self.out_channels // self.groups)
            if self._lazy_transposed
            else (0, self.out_channels)
        )
        if int(weight.shape[out_axis]) != expected:
            error_msgs.append(
                f"{name} out_channels mismatch at '{prefix}weight': "
                f"expected {expected}, got {int(weight.shape[out_axis])}"
            )
            return False
        self._dtype = self._dtype or weight.dtype
        self._device = self._device or weight.device
        in_channels = int(weight.shape[1 - out_axis])
        self._initialize(
            in_channels if self._lazy_transposed else in_channels * self.groups
        )
        return True

    @override
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
        """Size a still-lazy layer from the checkpoint's weight, then load it."""
        weight: Tensor | None = state_dict.get(f"{prefix}weight")
        if (
            self.in_channels is None
            and weight is not None
            and not self._sized_by(weight, prefix, error_msgs)
        ):
            return
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

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        padding = vars(self).get("_padding_str") or self.padding
        s = (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={padding}"
        )
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode!r}"
        return s
