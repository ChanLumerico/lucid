import math

import lucid

from lucid._tensor import Tensor
from lucid.types import _DeviceType, Numeric

from lucid.nn._kernel.embedding import embedding_kernel


def embedding(
    input_: Tensor,
    weight: Tensor,
    padding_idx: int | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
) -> Tensor:
    num_embeddings = int(weight.shape[0])
    if padding_idx is None:
        pad = -1
    else:
        pad = int(padding_idx)
        if pad < 0:
            pad += num_embeddings
        if pad < 0 or pad >= num_embeddings:
            raise IndexError("padding_idx out of range.")

    indices = input_.astype(lucid.Int)
    has_negative_idx = (indices < 0).sum().item() > 0
    has_oob_idx = (indices >= num_embeddings).sum().item() > 0
    if has_negative_idx or has_oob_idx:
        raise IndexError("embedding indices out of range.")

    if max_norm is not None:
        if norm_type <= 0:
            raise ValueError("norm_type must be positive.")

    op = embedding_kernel(padding_idx=pad, max_norm=max_norm, norm_type=norm_type)
    return op(indices, weight)


def sinusoidal_pos_embedding(
    seq_len: int,
    embed_dim: int,
    device: _DeviceType = "cpu",
    dtype: Numeric | None = None,
) -> Tensor:
    position = lucid.arange(seq_len, device=device).unsqueeze(axis=1)
    div_term = lucid.exp(
        lucid.arange(0, embed_dim, 2, device=device) * (-math.log(1e4) / embed_dim)
    )
    angle = position * div_term

    pos = lucid.zeros(
        (seq_len, embed_dim),
        device=device,
        dtype=lucid.Float32 if dtype is None else dtype,
    )
    pos[:, 0::2] = lucid.sin(angle)
    pos[:, 1::2] = lucid.cos(angle[:, : embed_dim // 2])

    return pos


def rotary_pos_embedding(
    input_: Tensor, position_ids: Tensor | None = None, interleaved: bool = True
) -> Tensor:
    seq_len, embed_dim = input_.shape[-2:]
    device = input_.device

    if embed_dim % 2 != 0:
        raise ValueError(f"Expected even input embedding dimension, got '{embed_dim}'.")

    theta = lucid.exp(
        -2
        * lucid.arange(embed_dim // 2, device=device, dtype=lucid.Double)
        * (math.log(10000.0) / embed_dim)
    )
    if position_ids is None:
        indices = lucid.arange(seq_len, device=device, dtype=lucid.Double)
    else:
        if position_ids.ndim != 1 or position_ids.shape[0] != seq_len:
            raise ValueError(
                "position_ids must be 1-D with length equal to input_.shape[-2]."
            )
        indices = position_ids.to(device).astype(lucid.Double)

    freq_half = indices.unsqueeze(-1) @ theta.unsqueeze(0)
    if interleaved:
        freq = freq_half.repeat(2, axis=-1)
    else:
        freq = lucid.concatenate([freq_half, freq_half], axis=-1)

    x = input_.astype(lucid.Double)
    input_rot = lucid.zeros_like(x)
    if interleaved:
        input_rot[..., 0::2] = -x[..., 1::2]
        input_rot[..., 1::2] = x[..., 0::2]
    else:
        half = embed_dim // 2
        input_rot[..., :half] = -x[..., half:]
        input_rot[..., half:] = x[..., :half]

    out = x * lucid.cos(freq) + input_rot * lucid.sin(freq)
    return out.astype(input_.dtype)
