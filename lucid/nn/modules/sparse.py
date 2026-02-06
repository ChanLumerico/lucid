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
            self.weight = nn.Parameter(
                lucid.random.uniform(-0.1, 0.1, (num_embeddings, embedding_dim))
            )
        else:
            self.weight = nn.Parameter(_weight)

        if self.padding_idx is not None:
            self.weight.data[self.padding_idx] = 0

    def forward(self, input_: Tensor) -> Tensor:
        return F.embedding(
            input_, self.weight, self.padding_idx, self.max_norm, self.norm_type
        )

    def reset_parameters(self) -> None:
        self.weight.data = lucid.random.uniform(-0.1, 0.1, self.weight.shape)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx] = 0
