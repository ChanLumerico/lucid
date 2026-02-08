from dataclasses import dataclass
from typing import Callable

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


# __all__ = ["BertConfig", "BERT"]  NOTE: Add final `BERT` class after final implementation.


@dataclass
class BertConfig:
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    intermediate_size: int

    hidden_act: Callable[[Tensor], Tensor]
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float

    max_position_embeddings: int
    tie_word_embedding: bool
    type_vocab_size: int

    initializer_range: float
    layer_norm_eps: float

    use_cache: bool
    is_decoder: bool
    add_cross_attention: bool
    chunk_size_feed_forward: int

    pad_token_id: int = 0
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    classifier_dropout: float | None = None


class _BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.position_ids: nn.Buffer
        self.token_type_ids: nn.Buffer
        self.register_buffer(
            "position_ids",
            nn.Buffer(lucid.arange(config.max_position_embeddings).expand(1, -1)),
        )
        self.register_buffer(
            "token_type_ids",
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
        config: BertConfig,
        /,
        is_causal: bool = False,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {config.hidden_size} is not a multiple "
                f"of num_attention_heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.is_decoder = config.is_decoder
        self.is_causal = is_causal
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: lucid.FloatTensor | None = None,
        past_key_values: nn.KVCache | nn.EncoderDecoderCache | None = None,
        cache_position: Tensor | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)

        query_layer = self.query(hidden_states).reshape(*hidden_shape).swapaxes(1, 2)
        key_layer = self.key(hidden_states).reshape(*hidden_shape).swapaxes(1, 2)
        value_layer = self.value(hidden_states).reshape(*hidden_shape).swapaxes(1, 2)

        if past_key_values is not None:
            if self.layer_idx is None:
                raise ValueError(f"past_key_values requires layer_idx")

            current_past_key_value = past_key_values
            if isinstance(past_key_values, nn.EncoderDecoderCache):
                current_past_key_value = past_key_values.self_attention_cache

            key_layer, value_layer = current_past_key_value.update(
                key_layer,
                value_layer,
                layer_idx=self.layer_idx,
                cache_position=cache_position,
            )

        attn_output, attn_weights = F.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=self.is_causal,
            scale=self.scaling,
            output_weight=True,
        )
        attn_output = attn_output.swapaxes(1, 2).reshape(
            *input_shape, self.all_head_size
        )
        return attn_output, attn_weights


class _BertCrossAttention(nn.Module):
    def __init__(
        self,
        config: BertConfig,
        /,
        is_causal: bool = False,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size '{config.hidden_size}' must be a "
                f"multiple of num_attention_heads '{config.num_attention_heads}'."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = config.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.is_causal = is_causal
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: lucid.FloatTensor | None = None,
        attention_mask: lucid.FloatTensor | None = None,
        past_key_values: nn.EncoderDecoderCache | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        bsz, tgt_len = hidden_states.shape[:-1]
        src_len = (
            encoder_hidden_states.shape[1]
            if encoder_hidden_states is not None
            else None
        )
        q_input_shape = (bsz, tgt_len, -1, self.attention_head_size)

        if past_key_values is not None and self.layer_idx is None:
            raise ValueError(f"past_key_values requires layer_idx")

        query_layer = self.query(hidden_states).reshape(*q_input_shape).swapaxes(1, 2)

        is_updated = (
            past_key_values.is_updated.get(self.layer_idx)
            if past_key_values is not None
            else False
        )
        if past_key_values is not None and is_updated:
            cached = past_key_values.cross_attention_cache.get(self.layer_idx)
            if cached is None:
                raise ValueError(
                    f"cross-attention cache is missing for layer_idx={self.layer_idx}."
                )
            key_layer, value_layer = cached
            src_len = key_layer.shape[-2]

        else:
            if encoder_hidden_states is None:
                raise ValueError(
                    "encoder_hidden_states is required when cache is empty."
                )
            src_len = encoder_hidden_states.shape[1]
            kv_input_shape = (bsz, src_len, -1, self.attention_head_size)

            key_layer = (
                self.key(encoder_hidden_states).reshape(*kv_input_shape).swapaxes(1, 2)
            )
            value_layer = (
                self.value(encoder_hidden_states)
                .reshape(*kv_input_shape)
                .swapaxes(1, 2)
            )
            if past_key_values is not None:
                key_layer, value_layer = past_key_values.update(
                    key_layer,
                    value_layer,
                    layer_idx=self.layer_idx,
                    is_cross_attention=True,
                )

        attn_output, attn_weight = F.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scaling,
            is_causal=self.is_causal,
            output_weight=True,
        )
        attn_output = attn_output.swapaxes(1, 2).reshape(
            bsz, tgt_len, self.all_head_size
        )
        return attn_output, attn_weight


class _BertSelfOutput(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_)

        return hidden_states


class _BertAttention(nn.Module):
    def __init__(
        self,
        config: BertConfig,
        /,
        is_causal: bool = False,
        layer_idx: int | None = None,
        is_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.is_cross_attention = is_cross_attention
        attention_class = (
            _BertCrossAttention if is_cross_attention else _BertSelfAttention
        )
        self.self = attention_class(config, is_causal=is_causal, layer_idx=layer_idx)
        self.output = _BertSelfOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: lucid.FloatTensor | None = None,
        encoder_hidden_states: lucid.FloatTensor | None = None,
        encoder_attention_mask: lucid.FloatTensor | None = None,
        past_key_values: nn.KVCache | nn.EncoderDecoderCache | None = None,
        cache_position: Tensor | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        attention_mask = (
            attention_mask if not self.is_cross_attention else encoder_attention_mask
        )
        if (
            self.is_cross_attention
            and past_key_values is not None
            and not isinstance(past_key_values, nn.EncoderDecoderCache)
        ):
            raise TypeError(
                "Cross-attention requires past_key_values to be "
                "nn.EncoderDecoderCache."
            )

        attn_output, attn_weights = self.self(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        attn_output = self.output(attn_output, hidden_states)
        return attn_output, attn_weights


class _BertIntermediate(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class _BertOutput(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_)

        return hidden_states


class _BertLayer(nn.Module):
    def __init__(self, config: BertConfig, layer_idx: int | None = None) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.attention = _BertAttention(
            config, is_causal=config.is_decoder, layer_idx=layer_idx
        )
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention

        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(
                    f"{self} should be used as a decoder if cross attention is added."
                )
            self.crossattention = _BertAttention(
                config, is_causal=False, layer_idx=layer_idx, is_cross_attention=True
            )

        self.intermediate = _BertIntermediate(config)
        self.output = _BertOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: lucid.FloatTensor | None = None,
        encoder_hidden_states: lucid.FloatTensor | None = None,
        encoder_attention_mask: lucid.FloatTensor | None = None,
        past_key_values: nn.KVCache | nn.EncoderDecoderCache | None = None,
        cache_position: Tensor | None = None,
        **kwargs,
    ) -> tuple[Tensor, ...]:
        self_attention_output, _ = self.attention(
            hidden_states,
            attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        attention_output = self_attention_output

        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise RuntimeError(
                    "If encoder_hidden_states is passed, "
                    "config.add_cross_attention must be set to True."
                )

            cross_attention_output, _ = self.crossattention(
                self_attention_output,
                attention_mask=None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
            attention_output = cross_attention_output

        layer_output = nn.utils.apply_chunking_to_forward(
            self.forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        return layer_output

    def forward_chunk(self, attention_output: Tensor) -> Tensor:
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output
