"""``lucid.nn.intrinsic`` — fused (intrinsic) float modules.

Produced by :func:`lucid.quantization.fuse_modules` and mapped to their
``nn.quantized`` counterparts during :func:`lucid.quantization.convert`.
"""

from lucid.nn.intrinsic.modules import ConvReLU1d, ConvReLU2d, ConvReLU3d, LinearReLU

__all__ = ["ConvReLU1d", "ConvReLU2d", "ConvReLU3d", "LinearReLU"]


def __dir__() -> list[str]:
    """Restrict introspection to the public surface (hide private helpers)."""
    return list(__all__)
