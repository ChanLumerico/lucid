"""
lucid.nn.functional._embedding — embedding gather + positional embeddings.

All three routes are 1:1 to fused C++ ops in `_C_nn`. `embedding` is a
gather + scatter-add backward; the positional embeddings produce constant
tables (sinusoidal) or fused rotation (RoPE) without any Python composition.
"""

from __future__ import annotations

from lucid._C.engine import nn as _C_nn
from lucid._tensor import Tensor
from lucid._bridge import impl_of, to_engine_dtype, to_engine_device
from lucid.types import _DeviceType, Numeric


def embedding(
    input_: Tensor,
    weight: Tensor,
    padding_idx: int | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
) -> Tensor:
    if max_norm is not None and norm_type <= 0:
        raise ValueError("norm_type must be positive.")
    num_embeddings = int(weight.shape[0])
    if padding_idx is None:
        pad = -1
    else:
        pad = int(padding_idx)
        if pad < 0:
            pad += num_embeddings
        if pad < 0 or pad >= num_embeddings:
            raise IndexError("padding_idx out of range.")
    return Tensor._wrap(_C_nn.embedding(
        impl_of(weight), impl_of(input_), pad))


def sinusoidal_pos_embedding(
    seq_len: int,
    embed_dim: int,
    device: _DeviceType = "cpu",
    dtype: Numeric | None = None,
) -> Tensor:
    from lucid.types import Float32
    eng_dtype = to_engine_dtype(dtype if dtype is not None else Float32)
    eng_device = to_engine_device(device)
    return Tensor._wrap(_C_nn.sinusoidal_pos_embedding(
        int(seq_len), int(embed_dim), eng_dtype, eng_device))


def rotary_pos_embedding(
    input_: Tensor,
    position_ids: Tensor | None = None,
    interleaved: bool = True,
) -> Tensor:
    seq_len, embed_dim = input_.shape[-2:]
    if embed_dim % 2 != 0:
        raise ValueError(f"Expected even input embedding dimension, got {embed_dim}.")
    if position_ids is not None:
        if position_ids.ndim != 1 or position_ids.shape[0] != seq_len:
            raise ValueError(
                "position_ids must be 1-D with length equal to input_.shape[-2].")
    return Tensor._wrap(_C_nn.rotary_pos_embedding(
        impl_of(input_),
        impl_of(position_ids) if position_ids is not None else None,
        bool(interleaved),
    ))
