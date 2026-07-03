"""``lucid.nn.intrinsic.qat`` — fused QAT modules (Conv+BN fold-in-training)."""

from lucid.nn.intrinsic.qat.modules import ConvBnReLU2d

# The internal ``convert`` builder ``convbnrelu2d_to_quantized`` lives in
# ``.modules`` and is imported there directly by ``lucid.quantization._quantize``
# — deliberately not re-exported here (it is not part of the public surface).
__all__ = ["ConvBnReLU2d"]


def __dir__() -> list[str]:
    """Restrict introspection to the public module."""
    return list(__all__)
