"""
lucid.nn.modules.sparse — embedding tables.
"""

from __future__ import annotations


import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


__all__ = ["Embedding"]


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        _weight: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        if padding_idx is None:
            self.padding_idx = None
        else:
            pad = int(padding_idx)
            if pad < 0:
                pad += num_embeddings
            if pad < 0 or pad >= num_embeddings:
                raise IndexError("padding_idx out of range.")
            self.padding_idx = pad

        self.max_norm = max_norm
        self.norm_type = norm_type

        if _weight is None:
            self.weight = nn.Parameter(self._make_init_weight())
        else:
            self.weight = nn.Parameter(_weight)

        if self.padding_idx is not None:
            self._zero_padding_row()

    def _make_init_weight(self) -> Tensor:
        from lucid.ops.random import uniform
        return uniform(self.num_embeddings, self.embedding_dim,
                        low=-0.1, high=0.1)

    def _zero_padding_row(self) -> None:
        # One-time initialization helper: zero the padding row in-place.
        # Done via host round-trip because Tensor does not expose
        # __setitem__ for slice assignment yet; this is *not* in any
        # forward/backward compute path.
        import numpy as np
        arr = self.weight.numpy().copy()
        arr[self.padding_idx] = 0
        new = Tensor(arr, dtype=self.weight.dtype, device=self.weight.device)
        self.weight._impl = new._impl

    def forward(self, input_: Tensor) -> Tensor:
        return F.embedding(
            input_, self.weight, self.padding_idx, self.max_norm, self.norm_type
        )

    def reset_parameters(self) -> None:
        new = self._make_init_weight()
        # Move to weight's device/dtype if needed.
        if new.dtype != self.weight.dtype:
            new = new.astype(self.weight.dtype)
        if new.device != self.weight.device:
            new = new.to(self.weight.device)
        self.weight._impl = new._impl
        if self.padding_idx is not None:
            self._zero_padding_row()
