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
