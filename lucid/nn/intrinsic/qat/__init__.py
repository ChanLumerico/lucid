"""``lucid.nn.intrinsic.qat`` — fused QAT modules (Conv+BN fold-in-training)."""

from lucid.nn.intrinsic.qat.modules import ConvBnReLU2d, convbnrelu2d_to_quantized

__all__ = ["ConvBnReLU2d", "convbnrelu2d_to_quantized"]
