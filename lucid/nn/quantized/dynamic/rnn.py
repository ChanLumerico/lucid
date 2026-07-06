"""Dynamically quantized ``LSTM``.

The recurrent weight matrices (``weight_ih_l*`` / ``weight_hh_l*``) are
stored as int8 (per-row symmetric); biases stay float.  A float ``nn.LSTM``
*shell* — held outside the module registry so it never enters the
``state_dict`` — is the compute engine: on each forward its weights are
overwritten with the dequantized values (which live on the same device as
the int8 buffers), so only the int8 form is the persistent, device-tracked
state.  This is weight-quantized dynamic inference; the per-timestep
activation quantization inside the cell is left to the real low-precision
kernel (Phase 6).
"""

from typing import TYPE_CHECKING, cast, override

import lucid
import lucid.nn as nn
from lucid.nn.parameter import Parameter
from lucid.quantization._functional import dequantize, quantize
from lucid.quantization._qscheme import QDtype, per_channel_symmetric, qint8

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


class LSTM(nn.Module):
    """int8-weight LSTM built from a float :class:`~lucid.nn.LSTM`.

    A dynamically quantized recurrent layer: every recurrent weight matrix
    (``weight_ih_l*`` / ``weight_hh_l*``) is stored int8 (per-row symmetric) while
    the biases stay float.  A float ``nn.LSTM`` *shell*, held outside the module
    registry so its float weights never enter the ``state_dict``, is the actual
    compute engine — on each forward its weights are overwritten with the values
    dequantized from the int8 buffers, so only the int8 form is the persistent,
    device-tracked state.  No calibration is required; the per-timestep activation
    quantization inside the cell is left to the real low-precision kernel.  Build
    instances with :func:`lucid.quantization.quantize_dynamic`, the recommended
    path for LSTM-heavy inference.

    Parameters
    ----------
    shell : lucid.nn.LSTM
        The float LSTM whose configuration and cell arithmetic are reused; it holds
        the recurrent weights that :meth:`from_float` quantizes to int8.

    Notes
    -----
    The shell is stored via ``object.__setattr__`` so it stays out of the module
    registry, keeping the serialized state limited to the int8 weight buffers, the
    per-row scales, and the float biases.
    """

    def __init__(self, shell: nn.LSTM) -> None:
        super().__init__()
        # The shell holds config + is the compute engine; keep it OUT of the
        # module registry so its float weights never reach ``state_dict``.
        object.__setattr__(self, "_shell", shell)
        self._weight_names: list[str] = [
            n for n, _ in shell.named_parameters() if n.startswith("weight")
        ]
        self._bias_names: list[str] = [
            n for n, _ in shell.named_parameters() if n.startswith("bias")
        ]

    @override
    def forward(  # type: ignore[override]  # LSTM-shaped (x, hx) signature
        self, x: Tensor, hx: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Inject dequantized weights into the shell and run it."""
        shell: nn.LSTM = object.__getattribute__(self, "_shell")
        for name in self._weight_names:
            codes = cast("Tensor", getattr(self, name + "_int8"))
            scale = cast("Tensor", getattr(self, name + "_scale"))
            deq = dequantize(codes, scale, lucid.tensor(0.0), ch_axis=0)
            shell._parameters[name] = Parameter(deq, requires_grad=False)
        for name in self._bias_names:
            bias = cast("Tensor", getattr(self, name))
            shell._parameters[name] = Parameter(bias, requires_grad=False)
        return shell.forward(x, hx)

    @classmethod
    def from_float(cls, mod: nn.Module, dtype: QDtype = qint8) -> LSTM:
        """Quantize a float :class:`~lucid.nn.LSTM`'s recurrent weights."""
        from lucid.quantization.observer import PerChannelMinMaxObserver

        shell = mod
        if not isinstance(shell, nn.LSTM):
            raise TypeError(
                f"dynamic LSTM.from_float expects an nn.LSTM, got {type(shell).__name__}"
            )
        qmod = cls(shell)
        for name, param in shell.named_parameters():
            if name.startswith("weight"):
                obs = PerChannelMinMaxObserver(
                    ch_axis=0, qscheme=per_channel_symmetric, qdtype=dtype
                )
                obs(param)
                scale, zero_point = obs.calculate_qparams()
                qmod.register_buffer(
                    name + "_int8", quantize(param, scale, zero_point, dtype, ch_axis=0)
                )
                qmod.register_buffer(name + "_scale", scale)
            else:
                qmod.register_buffer(name, param.detach())
        return qmod
