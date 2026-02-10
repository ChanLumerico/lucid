from dataclasses import dataclass
from typing import Callable

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


__all__ = ["BERTConfig", "BERT"]


@dataclass
class BERTConfig:
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
    add_pooling_layer: bool = True


class _BERTEmbeddings(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
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
        cache_position: Tensor | None = None,
    ) -> Tensor:
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        batch_size, seq_length = input_shape
        if position_ids is None:
            if cache_position is not None:
                if cache_position.ndim != 1 or cache_position.shape[0] != seq_length:
                    raise ValueError(
                        "cache_position must be 1-D with length == seq_length "
                        f"(got shape={cache_position.shape}, seq_length={seq_length})."
                    )
                position_ids = cache_position.reshape(1, seq_length)
            else:
                position_ids = self.position_ids[
                    :, past_key_values_length : seq_length + past_key_values_length
                ]
        elif (
            position_ids.ndim != 2
            or position_ids.shape[1] != seq_length
            or position_ids.shape[0] not in (1, batch_size)
        ):
            raise ValueError(
                "position_ids must be 2-D with shape "
                "[batch_or_1, seq_length] where batch_or_1 is 1 or batch_size "
                f"(got shape={position_ids.shape}, batch_size={batch_size}, "
                f"seq_length={seq_length})."
            )

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


class _BERTSelfAttention(nn.Module):
    def __init__(
        self,
        config: BERTConfig,
        /,
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
            is_causal=False,
            scale=self.scaling,
            output_weight=True,
        )
        attn_output = attn_output.swapaxes(1, 2).reshape(
            *input_shape, self.all_head_size
        )
        return attn_output, attn_weights


class _BERTCrossAttention(nn.Module):
    def __init__(
        self,
        config: BERTConfig,
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


class _BERTSelfOutput(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_)

        return hidden_states


class _BERTAttention(nn.Module):
    def __init__(
        self,
        config: BERTConfig,
        /,
        is_causal: bool = False,
        layer_idx: int | None = None,
        is_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.is_cross_attention = is_cross_attention
        if is_cross_attention:
            self.self = _BERTCrossAttention(
                config, is_causal=is_causal, layer_idx=layer_idx
            )
        else:
            self.self = _BERTSelfAttention(config, layer_idx=layer_idx)
        self.output = _BERTSelfOutput(config)

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


class _BERTIntermediate(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class _BERTOutput(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_)

        return hidden_states


class _BERTLayer(nn.Module):
    def __init__(self, config: BERTConfig, layer_idx: int | None = None) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.attention = _BERTAttention(
            config, is_causal=config.is_decoder, layer_idx=layer_idx
        )
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention

        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(
                    f"{self} should be used as a decoder if cross attention is added."
                )
            self.crossattention = _BERTAttention(
                config, is_causal=False, layer_idx=layer_idx, is_cross_attention=True
            )

        self.intermediate = _BERTIntermediate(config)
        self.output = _BERTOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: lucid.FloatTensor | None = None,
        encoder_hidden_states: lucid.FloatTensor | None = None,
        encoder_attention_mask: lucid.FloatTensor | None = None,
        past_key_values: nn.KVCache | nn.EncoderDecoderCache | None = None,
        cache_position: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
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


class _BERTEncoder(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [_BERTLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: lucid.FloatTensor | None = None,
        encoder_hidden_states: lucid.FloatTensor | None = None,
        encoder_attention_mask: lucid.FloatTensor | None = None,
        past_key_values: nn.KVCache | nn.EncoderDecoderCache | None = None,
        use_cache: bool | None = None,
        cache_position: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        effective_use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )
        if not self.config.is_decoder:
            effective_use_cache = False

        if (
            self.config.add_cross_attention
            and past_key_values is not None
            and not isinstance(past_key_values, nn.EncoderDecoderCache)
        ):
            raise TypeError(
                "When add_cross_attention=True, past_key_values must be "
                "nn.EncoderDecoderCache."
            )

        if not effective_use_cache:
            past_key_values = None
            cache_position = None
        elif past_key_values is None:
            raise ValueError(
                "use_cache=True requires a persistent past_key_values cache "
                "instance from the caller."
            )

        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        return hidden_states


class _BERTPooler(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class BERT(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.config = config

        self.embeddings = _BERTEmbeddings(config)
        self.encoder = _BERTEncoder(config)
        self.pooler = _BERTPooler(config) if config.add_pooling_layer else None
        self.output_embeddings: nn.Linear | None = None

        self.apply(self._init_weights)
        self.tie_weights()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.constant(module.bias, 0.0)

        elif isinstance(module, nn.Embedding):
            nn.init.normal(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx] = 0

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant(module.bias, 0.0)
            nn.init.constant(module.weight, 1.0)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embeddings.word_embeddings = value
        self.tie_weights()

    def get_output_embeddings(self) -> nn.Linear | None:
        return self.output_embeddings

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.output_embeddings = value
        self.tie_weights()

    def tie_weights(self) -> None:
        if not self.config.tie_word_embedding:
            return

        output_embeddings = self.get_output_embeddings()
        if output_embeddings is None:
            return

        output_embeddings.weight = self.get_input_embeddings().weight

    def _to_4d_attention_mask(
        self,
        attention_mask: Tensor | None,
        target_length: int,
        source_length: int,
        device: str,
    ) -> Tensor | None:
        if attention_mask is None:
            return None

        if attention_mask.device != device:
            attention_mask = attention_mask.to(device)

        if attention_mask.ndim == 2:
            if attention_mask.shape[1] != source_length:
                raise ValueError(
                    f"attention_mask has invalid source length: "
                    f"{attention_mask.shape[1]} != {source_length}"
                )
            attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.ndim == 3:
            if attention_mask.shape[-1] != source_length:
                raise ValueError(
                    f"attention_mask has invalid source length: "
                    f"{attention_mask.shape[-1]} != {source_length}"
                )
            attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.ndim == 4:
            if attention_mask.shape[-1] != source_length:
                raise ValueError(
                    f"attention_mask has invalid source length: "
                    f"{attention_mask.shape[-1]} != {source_length}"
                )
        else:
            raise ValueError(
                f"attention_mask must be 2-D, 3-D, or 4-D (got {attention_mask.ndim})."
            )

        if attention_mask.shape[-2] not in (1, target_length):
            raise ValueError(
                f"attention_mask has invalid target length axis: "
                f"{attention_mask.shape[-2]} not in (1, {target_length})"
            )

        attention_mask = attention_mask.astype(lucid.Float32)
        mask_min = float(lucid.min(attention_mask).item())
        mask_max = float(lucid.max(attention_mask).item())

        if mask_min >= 0.0 and mask_max <= 1.0:
            return (1.0 - attention_mask) * -1e12
        if mask_max <= 0.0:
            return attention_mask

        raise ValueError(
            "attention_mask values must be binary (0/1) or additive (<= 0 values)."
        )

    def _build_decoder_causal_mask(
        self,
        *,
        batch_size: int,
        target_length: int,
        source_length: int,
        device: str,
        past_key_values_length: int,
        position_ids: Tensor | None = None,
        cache_position: Tensor | None = None,
    ) -> Tensor:
        if position_ids is not None:
            query_positions = position_ids

        elif cache_position is not None:
            if cache_position.ndim != 1 or cache_position.shape[0] != target_length:
                raise ValueError(
                    "cache_position must be 1-D with length == target_length "
                    f"(got shape={cache_position.shape}, target_length={target_length})."
                )
            query_positions = cache_position.reshape(1, target_length)

        else:
            query_positions = lucid.arange(
                past_key_values_length,
                past_key_values_length + target_length,
                device=device,
            )
            query_positions = query_positions.reshape(1, target_length)

        if query_positions.ndim != 2 or query_positions.shape[1] != target_length:
            raise ValueError(
                "query positions must be 2-D with shape [batch_or_1, target_length] "
                f"(got shape={query_positions.shape}, target_length={target_length})."
            )

        if query_positions.shape[0] == 1:
            query_positions = query_positions.expand(batch_size, target_length)
        elif query_positions.shape[0] != batch_size:
            raise ValueError(
                "query positions batch axis must be 1 or batch_size "
                f"(got {query_positions.shape[0]} vs {batch_size})."
            )

        if query_positions.device != device:
            query_positions = query_positions.to(device)

        query_positions = query_positions.reshape(batch_size, target_length, 1)
        key_positions = lucid.arange(source_length, device=device).reshape(1, 1, -1)

        causal_mask = (key_positions > query_positions).astype(lucid.Float32) * -1e12
        return causal_mask[:, None, :, :]

    def forward(
        self,
        input_ids: lucid.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: lucid.LongTensor | None = None,
        position_ids: lucid.LongTensor | None = None,
        inputs_embeds: lucid.FloatTensor | None = None,
        encoder_hidden_states: lucid.FloatTensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        past_key_values: nn.KVCache | nn.EncoderDecoderCache | None = None,
        use_cache: bool | None = None,
        cache_position: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Only one of input_ids or inputs_embeds can be provided.")

        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        if input_ids is not None:
            input_shape = input_ids.shape
            device = input_ids.device
        else:
            input_shape = inputs_embeds.shape[:-1]
            device = inputs_embeds.device

        if len(input_shape) != 2:
            raise ValueError(
                f"BERT expects 2-D token inputs [batch, seq] (got {input_shape})."
            )

        effective_use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )
        if not self.config.is_decoder:
            effective_use_cache = False

        if not effective_use_cache:
            past_key_values = None
            cache_position = None

        past_key_values_length = 0
        if effective_use_cache and past_key_values is not None:
            if isinstance(past_key_values, nn.EncoderDecoderCache):
                self_cache = past_key_values.self_attention_cache
                if len(self_cache.key_cache) > 0:
                    past_key_values_length = self_cache.get_seq_length()
            else:
                if len(past_key_values.key_cache) > 0:
                    past_key_values_length = past_key_values.get_seq_length()

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            cache_position=cache_position,
        )

        source_length = input_shape[1] + past_key_values_length
        if attention_mask is None:
            attention_mask = lucid.ones((input_shape[0], source_length), device=device)

        elif attention_mask.ndim == 2 and attention_mask.shape[1] == input_shape[1]:
            if past_key_values_length > 0:
                past_attention = lucid.ones(
                    (input_shape[0], past_key_values_length), device=device
                )
                attention_mask = lucid.concatenate(
                    [past_attention, attention_mask], axis=1
                )

        extended_attention_mask = self._to_4d_attention_mask(
            attention_mask=attention_mask,
            target_length=input_shape[1],
            source_length=source_length,
            device=device,
        )
        if self.config.is_decoder:
            causal_attention_mask = self._build_decoder_causal_mask(
                batch_size=input_shape[0],
                target_length=input_shape[1],
                source_length=source_length,
                device=device,
                past_key_values_length=past_key_values_length,
                position_ids=position_ids,
                cache_position=cache_position,
            )
            if extended_attention_mask is None:
                extended_attention_mask = causal_attention_mask
            else:
                extended_attention_mask = (
                    extended_attention_mask + causal_attention_mask
                )

        if encoder_hidden_states is None:
            if encoder_attention_mask is not None:
                raise ValueError(
                    "encoder_attention_mask requires encoder_hidden_states."
                )
            extended_encoder_attention_mask = None
        else:
            extended_encoder_attention_mask = self._to_4d_attention_mask(
                attention_mask=encoder_attention_mask,
                target_length=input_shape[1],
                source_length=encoder_hidden_states.shape[1],
                device=device,
            )

        sequence_output = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=extended_encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        return sequence_output, pooled_output
