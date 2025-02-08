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
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type

        if _weight is None:
            self.weight = nn.Parameter(
                lucid.random.uniform(-0.1, 0.1, (num_embeddings, embedding_dim))
            )
        else:
            self.weight = nn.Parameter(_weight)

    def forward(self, input_: Tensor) -> Tensor:
        return F.embedding(
            input_, self.weight, self.padding_idx, self.max_norm, self.norm_type
        )

    def reset_parameters(self) -> None:
        self.weight.data = lucid.random.uniform(-0.1, 0.1, self.weight.shape)
