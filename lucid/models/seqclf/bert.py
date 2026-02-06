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
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size)

        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.position_ids: nn.Buffer
        self.register_buffer(
            "position_ids", nn.Buffer(lucid.arange(max_position_embeddings))
        )
        # TODO: Implement `lucid.Tensor.expand`
