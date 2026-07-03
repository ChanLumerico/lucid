"""``lucid.nn.quantized.dynamic`` — dynamically quantized modules.

Weights are quantized to int8 ahead of time; **activations are quantized
per-forward** from their observed range (no calibration data needed).  This
is the go-to for Linear-heavy / Transformer inference, produced by
:func:`lucid.quantization.quantize_dynamic`.
"""

from lucid.nn.quantized.dynamic.linear import Linear
from lucid.nn.quantized.dynamic.rnn import LSTM

__all__ = ["Linear", "LSTM"]


def __dir__() -> list[str]:
    """Restrict introspection to the public modules."""
    return list(__all__)
