"""
lucid.nn.utils.fusion — inference-time layer fusion utilities.

The canonical entry point is ``fuse_conv_bn_eval(conv, bn)`` which
returns a new Conv2d-like module with the BatchNorm's running
statistics + affine parameters absorbed into the conv's weight and
bias.  The result computes exactly the same thing as ``bn(conv(x))``
*at eval time* but with one fewer kernel launch.

Only valid in evaluation mode (when BN uses its frozen running
statistics rather than batch statistics).  Calling this on a training
graph silently produces incorrect results — guard with ``module.eval()``.
"""

import copy as _copy_module

import lucid as _lucid
from lucid.nn.modules.conv import Conv1d, Conv2d, Conv3d
from lucid.nn.modules.normalization import BatchNorm1d, BatchNorm2d, BatchNorm3d


_CONV_TYPES = (Conv1d, Conv2d, Conv3d)
_BN_TYPES = (BatchNorm1d, BatchNorm2d, BatchNorm3d)


def fuse_conv_bn_eval(conv: object, bn: object) -> object:
    """Fuse ``bn(conv(x))`` into an equivalent Conv module for inference.

    Algorithm: with BN's running mean μ, running var σ², ε, affine γ/β:

        scale = γ / sqrt(σ² + ε)         (or 1 / sqrt(σ² + ε) if no affine)
        new_weight = conv.weight * scale.reshape(out, 1, ...)
        new_bias   = (conv.bias - μ) * scale + β
                     ((-μ) * scale + β if conv had no bias)

    Returns a deep-copied conv module with fused weight / bias.  The
    original ``conv`` and ``bn`` are not mutated.

    Parameters
    ----------
    conv : Conv1d | Conv2d | Conv3d
        Convolution module to absorb the BN into.
    bn : BatchNorm1d | BatchNorm2d | BatchNorm3d
        BatchNorm module providing running statistics + affine params.

    Returns
    -------
    Conv : a clone of ``conv`` with weight/bias updated.

    Raises
    ------
    TypeError
        If ``conv`` is not one of the supported Conv types or ``bn`` is
        not the matching BatchNorm rank.
    """
    if not isinstance(conv, _CONV_TYPES):
        raise TypeError(
            f"fuse_conv_bn_eval: conv must be Conv1d/2d/3d, got {type(conv).__name__}"
        )
    if not isinstance(bn, _BN_TYPES):
        raise TypeError(
            f"fuse_conv_bn_eval: bn must be BatchNorm1d/2d/3d, got {type(bn).__name__}"
        )

    fused = _copy_module.deepcopy(conv)

    # Pull out the relevant tensors as detached leaves on the conv's device.
    eps = float(bn.eps)
    running_mean = bn.running_mean.detach()
    running_var = bn.running_var.detach()
    gamma = bn.weight.detach() if bn.affine and bn.weight is not None else None
    beta = bn.bias.detach() if bn.affine and bn.bias is not None else None

    # scale = γ / sqrt(σ² + ε)  (γ defaults to 1 if affine is off)
    inv_std = (running_var + eps).rsqrt()
    scale = inv_std if gamma is None else gamma * inv_std

    # Broadcast scale to weight shape: (out_channels, 1, 1, ...).
    weight = fused.weight.detach()
    out_channels = int(weight.shape[0])
    scale_view = scale.reshape(out_channels, *([1] * (weight.ndim - 1)))
    new_weight = weight * scale_view

    # bias: start from (-running_mean) if conv had no bias.
    if fused.bias is not None:
        old_bias = fused.bias.detach()
    else:
        old_bias = _lucid.zeros(out_channels, dtype=weight.dtype, device=weight.device)
    new_bias = (old_bias - running_mean) * scale
    if beta is not None:
        new_bias = new_bias + beta

    # Write back as parameters on the fused module.
    from lucid.nn.parameter import Parameter  # noqa: PLC0415 — avoid cycle at module load
    fused.weight = Parameter(new_weight)
    if fused.bias is None:
        fused.bias = Parameter(new_bias)
    else:
        fused.bias = Parameter(new_bias)
    return fused
