"""
Sparse / embedding modules.
"""

import math
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike
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
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = Parameter(
            empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        )
        init.normal_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
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
        device: DeviceLike = None,
        dtype: DTypeLike = None,
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

    def forward(self, x: Tensor, offsets: Tensor | None = None) -> Tensor:
        from lucid.nn.functional.sampling import embedding_bag as _eb

        _mode_map = {"sum": "sum", "mean": "mean", "max": "max"}
        return _eb(
            x,
            self.weight,
            offsets=offsets,
            mode=_mode_map.get(self.mode, "mean"),
            padding_idx=self.padding_idx,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.num_embeddings}, {self.embedding_dim}, "
            f"mode={self.mode!r}, padding_idx={self.padding_idx}"
        )
