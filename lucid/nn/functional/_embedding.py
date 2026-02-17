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
    seq_len: int, embed_dim: int, device: _DeviceType, dtype: Numeric | None = None
) -> Tensor:
    if seq_len <= 0 or embed_dim <= 0:
        raise ValueError("seq_len and embed_dim must be positive.")

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
