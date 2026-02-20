from dataclasses import dataclass

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.models.text.bert import _model as bert


__all__ = ["RoFormerConfig", "RoFormer"]


@dataclass
class RoFormerConfig(bert.BERTConfig):
    rotary_value: bool = False
    rope_interleaved: bool = True


class _RoFormerEmbeddings(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.token_type_ids: nn.Buffer
        self.register_buffer(
            "token_type_ids",
            nn.Buffer(lucid.zeros(1, config.max_position_embeddings), dtype=lucid.Long),
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
        del position_ids, past_key_values_length, cache_position

        if input_ids is not None:
            input_shape = input_ids.shape
            device = input_ids.device
        else:
            input_shape = inputs_embeds.shape[:-1]
            device = inputs_embeds.device

        batch_size, seq_length = input_shape
        if token_type_ids is None:
            if self.token_type_ids.shape[1] >= seq_length:
                token_type_ids = self.token_type_ids[:, :seq_length].expand(
                    batch_size, seq_length
                )
            else:
                token_type_ids = lucid.zeros(
                    *input_shape, dtype=lucid.Long, device=device
                )

        elif token_type_ids.shape != input_shape:
            raise ValueError(
                "'token_type_ids' must match input token shape "
                f"(expected {input_shape}, got {token_type_ids.shape})"
            )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class _RoFormerSelfAttention(bert._BERTSelfAttention):
    def __init__(self, config: RoFormerConfig, /, layer_idx: int | None = None) -> None:
        super().__init__(config, layer_idx)
        self.rotary_value = config.rotary_value

        self.rope_qk = nn.RotaryPosEmbedding(
            embed_dim=self.attention_head_size,
            max_seq_len=config.max_position_embeddings,
            interleaved=config.rope_interleaved,
        )
        self.rope_v = (
            nn.RotaryPosEmbedding(
                embed_dim=self.attention_head_size,
                max_seq_len=config.max_position_embeddings,
                interleaved=config.rope_interleaved,
            )
            if config.rotary_value
            else None
        )

    def _resolve_rope_position_ids(
        self,
        *,
        hidden_states: Tensor,
        position_ids: Tensor | None,
        cache_position: Tensor | None,
        past_key_values: nn.KVCache | nn.EncoderDecoderCache | None,
    ) -> Tensor:
        batch_size, seq_length = hidden_states.shape[:2]
        device = hidden_states.device

        if position_ids is not None:
            if position_ids.ndim == 1:
                if position_ids.shape[0] != seq_length:
                    raise ValueError(
                        "1-D position_ids must have length == seq_length "
                        f"(got {position_ids.shape[0]} vs {seq_length})."
                    )
                return position_ids.to(device).long()

            if (
                position_ids.ndim != 2
                or position_ids.shape[1] != seq_length
                or position_ids.shape[0] not in (1, batch_size)
            ):
                raise ValueError(
                    "position_ids must be 1-D [seq] or 2-D [batch_or_1, seq] "
                    f"(got {position_ids.shape})."
                )

            if position_ids.shape[0] > 1:
                mismatch = (position_ids != position_ids[:1]).sum().long().item()
                if mismatch != 0:
                    raise ValueError(
                        "RoPE currently expects identical position_ids across batch."
                    )

            return position_ids[0].to(device).long()

        if cache_position is not None:
            if cache_position.ndim != 1 or cache_position.shape[0] != seq_length:
                raise ValueError(
                    "cache_position must be 1-D with length == seq_length "
                    f"(got {cache_position.shape}, seq_length={seq_length})."
                )
            return cache_position.to(device).long()

        past_key_values_length = 0
        if past_key_values is not None:
            current_past_key_value = past_key_values
            if isinstance(past_key_values, nn.EncoderDecoderCache):
                current_past_key_value = past_key_values.self_attention_cache

            if len(current_past_key_value.key_cache) > 0:
                past_key_values_length = current_past_key_value.get_seq_length()

        return lucid.arange(
            past_key_values_length,
            past_key_values_length + seq_length,
            device=device,
            dtype=lucid.Long,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: lucid.FloatTensor | None = None,
        past_key_values: nn.KVCache | nn.EncoderDecoderCache | None = None,
        cache_position: Tensor | None = None,
        position_ids: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        del kwargs

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)

        query_layer = self.query(hidden_states).reshape(*hidden_shape).swapaxes(1, 2)
        key_layer = self.key(hidden_states).reshape(*hidden_shape).swapaxes(1, 2)
        value_layer = self.value(hidden_states).reshape(*hidden_shape).swapaxes(1, 2)

        rope_position_ids = self._resolve_rope_position_ids(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )

        query_layer = self.rope_qk(query_layer, rope_position_ids)
        key_layer = self.rope_qk(key_layer, rope_position_ids)

        if self.rope_v is not None:
            value_layer = self.rope_v(value_layer, rope_position_ids)

        if past_key_values is not None:
            if self.layer_idx is None:
                raise ValueError("past_key_values requires layer_idx")

            current_past_key_value = past_key_values
            if isinstance(past_key_values, nn.EncoderDecoderCache):
                current_past_key_value = past_key_values.self_attention_cache

            key_layer, value_layer = current_past_key_value.update(
                key_layer,
                value_layer,
                layer_idx=self.layer_idx,
                cache_position=cache_position,
            )

        attn_output = F.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
            scale=self.scaling,
        )
        attn_output = attn_output.swapaxes(1, 2).reshape(
            *input_shape, self.all_head_size
        )
        return attn_output


class _RoFormerAttention(nn.Module):
    def __init__(
        self,
        config: RoFormerConfig,
        /,
        is_causal: bool = False,
        layer_idx: int | None = None,
        is_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.is_cross_attention = is_cross_attention
        if is_cross_attention:
            self.self = bert._BERTCrossAttention(
                config, is_causal=is_causal, layer_idx=layer_idx
            )
        else:
            self.self = _RoFormerSelfAttention(config, layer_idx=layer_idx)
        self.output = bert._BERTSelfOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: lucid.FloatTensor | None = None,
        encoder_hidden_states: lucid.FloatTensor | None = None,
        encoder_attention_mask: lucid.FloatTensor | None = None,
        past_key_values: nn.KVCache | nn.EncoderDecoderCache | None = None,
        cache_position: Tensor | None = None,
        position_ids: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        del kwargs

        attention_mask = (
            attention_mask if not self.is_cross_attention else encoder_attention_mask
        )
        if (
            self.is_cross_attention
            and past_key_values is not None
            and not isinstance(past_key_values, nn.EncoderDecoderCache)
        ):
            raise TypeError(
                "Cross-attention requires past_key_values to be nn.EncoderDecoderCache."
            )

        attn_output = self.self(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_ids=position_ids,
        )
        attn_output = self.output(attn_output, hidden_states)
        return attn_output


class _RoFormerLayer(nn.Module):
    def __init__(self, config: RoFormerConfig, layer_idx: int | None = None) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.attention = _RoFormerAttention(
            config, is_causal=config.is_decoder, layer_idx=layer_idx
        )
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention

        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(
                    f"{self} should be used as a decoder if cross attention is added."
                )
            self.crossattention = _RoFormerAttention(
                config, is_causal=False, layer_idx=layer_idx, is_cross_attention=True
            )

        self.intermediate = bert._BERTIntermediate(config)
        self.output = bert._BERTOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: lucid.FloatTensor | None = None,
        encoder_hidden_states: lucid.FloatTensor | None = None,
        encoder_attention_mask: lucid.FloatTensor | None = None,
        past_key_values: nn.KVCache | nn.EncoderDecoderCache | None = None,
        cache_position: Tensor | None = None,
        position_ids: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        self_attention_output = self.attention(
            hidden_states,
            attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_ids=position_ids,
            **kwargs,
        )
        attention_output = self_attention_output

        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise RuntimeError(
                    "If encoder_hidden_states is passed, "
                    "config.add_cross_attention must be set to True."
                )

            cross_attention_output = self.crossattention(
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


class _RoFormerEncoder(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [
                _RoFormerLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
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
        position_ids: Tensor | None = None,
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
                "use_cache=True requires a persistent past_key_values cache instance "
                "from the caller."
            )

        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_ids=position_ids,
                **kwargs,
            )

        return hidden_states


class RoFormer(bert.BERT):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__(config)
        self.config = config

        self.embeddings = _RoFormerEmbeddings(config)
        self.encoder = _RoFormerEncoder(config)
        self.pooler = bert._BERTPooler(config) if config.add_pooling_layer else None
        self.output_embeddings: nn.Linear | None = None

        self.apply(self._init_weights)
        self.tie_weights()

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
                f"RoFormer expects 2-D token inputs [batch, seq] (got {input_shape})."
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
            attention_mask,
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
            use_cache=effective_use_cache,
            cache_position=cache_position,
            position_ids=position_ids,
        )
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )
        return sequence_output, pooled_output
