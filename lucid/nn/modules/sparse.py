"""
Sparse / embedding modules.
"""

from typing import Any
import math
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._factories.creation import empty
import lucid.nn.init as init
from lucid.nn.functional.sparse import embedding


class Embedding(Module):
    """Learnable embedding lookup table."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = Parameter(
            empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        )
        init.normal_(self.weight)

    def forward(self, x: Any) -> Any:
        return embedding(x, self.weight, self.padding_idx)

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}, padding_idx={self.padding_idx}"


class EmbeddingBag(Module):
    """Embedding lookup with per-bag reduction (``'sum'`` / ``'mean'`` / ``'max'``).

    When *offsets* is ``None`` the input is expected to have shape ``(B, L)``
    and the reduction is applied over the length dimension *L*.
    When *offsets* is provided the input is a flat 1-D index tensor and
    *offsets* marks the start of each bag.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
        padding_idx: int | None = None,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.padding_idx = padding_idx
        self.weight = Parameter(
            empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        )
        init.normal_(self.weight)

    def forward(self, x: Any, offsets: Any = None) -> Any:
        from lucid._C import engine as _C_engine
        from lucid._dispatch import _unwrap, _wrap
        from lucid._ops import stack

        emb = embedding(x, self.weight, self.padding_idx)
        emb_impl = _unwrap(emb)

        if offsets is None:
            # x shape (B, L) → emb shape (B, L, D); reduce over dim=1
            if self.mode == "sum":
                return _wrap(_C_engine.sum(emb_impl, [-2], False))
            if self.mode == "max":
                return _wrap(_C_engine.max(emb_impl, [-2], False))
            return _wrap(_C_engine.mean(emb_impl, [-2], False))

        # Flat index mode with offsets
        import numpy as np
        x_np = np.array(_unwrap(x).data_as_python(), dtype=np.int64).ravel()
        offs_np = np.array(_unwrap(offsets).data_as_python(), dtype=np.int64).ravel()
        B = len(offs_np)
        results = []
        for i in range(B):
            start = int(offs_np[i])
            end = int(offs_np[i + 1]) if i + 1 < B else len(x_np)
            bag_idx = x_np[start:end].astype(np.int64)
            idx_impl = _C_engine.TensorImpl(bag_idx, _C_engine.CPU, False)
            bag_impl = _C_engine.gather(_unwrap(self.weight), 0, idx_impl)
            if self.mode == "sum":
                r = _C_engine.sum(bag_impl, [0], False)
            elif self.mode == "max":
                r = _C_engine.max(bag_impl, [0], False)
            else:
                r = _C_engine.mean(bag_impl, [0], False)
            results.append(_wrap(r))
        return stack(results, 0)

    def extra_repr(self) -> str:
        return (
            f"{self.num_embeddings}, {self.embedding_dim}, "
            f"mode={self.mode!r}, padding_idx={self.padding_idx}"
        )
