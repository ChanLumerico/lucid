"""``lucid.nn.qat`` — quantization-aware training modules.

Float-trainable layers that fake-quantize their weights and outputs (via the
straight-through estimator) so a network learns to be robust to int8
inference.  Produced by :func:`lucid.quantization.prepare_qat`; turned into
real quantized inference layers by :func:`lucid.quantization.convert`.
"""

from lucid.nn.qat.conv import Conv1d, Conv2d, Conv3d
from lucid.nn.qat.linear import Linear

__all__ = ["Linear", "Conv1d", "Conv2d", "Conv3d"]
