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
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.position_ids: nn.Buffer
        self.token_type_ids: nn.Buffer
        self.register_buffer(
            "position_ids",
            nn.Buffer(lucid.arange(max_position_embeddings).expand(1, -1)),
        )
        self.register_buffer(
            "token_type_uds",
            nn.Buffer(lucid.zeros(*self.position_ids.shape, dtype=lucid.Long)),
        )

    def forward(
        self,
        input_ids: lucid.LongTensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
        past_key_values_length: int = 0,
    ) -> Tensor:
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        batch_size, seq_length = input_shape
        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids.expand(
                    position_ids.shape[0], -1
                )
                buffered_token_type_ids = lucid.gather(
                    buffered_token_type_ids, dim=1, index=position_ids
                )
                token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
            else:
                token_type_ids = lucid.zeros(
                    *input_shape, dtype=lucid.Long, device=self.position_ids.device
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings

        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class _BertSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        is_decoder: bool,
        is_causal: bool = False,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} is not a multiple "
                f"of num_attention_heads {num_attention_heads}."
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        self.is_decoder = is_decoder
        self.is_causal = is_causal
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: lucid.FloatTensor | None = None,
        past_key_value=None,  # NOTE: Need to implement `Cache` class?
    ) -> tuple[Tensor]: ...
