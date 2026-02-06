import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


class _BertEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        pad_token_id: int,
        max_position_embeddings: int,
        type_vocab_size: int,
        layer_norm_eps: float,
        hidden_dropout_prob: float,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id
        )
