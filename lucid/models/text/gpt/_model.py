from dataclasses import dataclass

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.models.base import PreTrainedModelMixin

__all__ = ["GPTConfig", "GPT"]


@dataclass
class GPTConfig:
    vocab_size: int = 40478
    max_position_embeddings: int = 512

    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12

    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1

    attention_prob_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02

    use_cache: bool = True
    bos_token_id: int = 40476
    eos_token_id: int = 40477
    pad_token_id: int | None = None

    def __post_init__(self) -> None:
        assert (
            self.hidden_size % self.num_attention_heads == 0
        ), "hidden_size must be divisible by num_attention_heads"
        assert self.hidden_size > 0 and self.num_hidden_layers > 0

    @classmethod
    def base(cls, **kwargs) -> GPTConfig:
        defaults = dict(
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=12,
            intermediate_size=3072,
        )
        defaults.update(kwargs)
        return cls(**defaults)


class _GPTEmbedding(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.tokens_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.positions_embed = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        self.drop = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids",
            lucid.arange(config.max_position_embeddings).unsqueeze(axis=0),
        )
        self.position_ids: nn.Buffer

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        past_length: int = 0,
    ) -> Tensor:
        seq_len = input_ids.shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_length : past_length + seq_len]
        return self.drop(
            self.tokens_embed(input_ids) + self.positions_embed(position_ids)
        )


class _GPTAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.scale = self.head_dim**-0.5

        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_drop = nn.Dropout(config.attention_prob_dropout_prob)
        self.resid_drop = nn.Dropout(config.hidden_dropout_prob)

        max_len = config.max_position_embeddings
        self.register_buffer(
            "causal_mask",
            lucid.tril(lucid.ones(max_len, max_len)).reshape(1, 1, max_len, max_len),
        )
        self.causal_mask: nn.Buffer

    def forward(
        self,
        hidden_state: Tensor,
        attention_mask: Tensor | None = None,
        past_key_value: nn.KVCache | None = None,
        cache_position: Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, nn.KVCache | None]:
        B, T, C = hidden_state.shape
        q, k, v = self.c_attn(hidden_state).split(self.hidden_size, axis=-1)

        q = q.reshape(B, T, self.num_heads, self.head_dim).swapaxes(1, 2)
        k = k.reshape(B, T, self.num_heads, self.head_dim).swapaxes(1, 2)
        v = v.reshape(B, T, self.num_heads, self.head_dim).swapaxes(1, 2)

        if past_key_value is not None:
            past_len = past_key_value.get_seq_length(self.layer_idx)
            k, v = past_key_value.update(k, v, self.layer_idx, cache_position)
        else:
            past_len = 0

        kv_len = k.shape[-2]
        mask = self.causal_mask[:, :, past_len : past_len + T, :kv_len]
        if attention_mask is not None:
            mask = mask + attention_mask

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
        )
        attn_out = attn_out.swapaxes(1, 2).reshape(B, T, C)
        attn_out = self.resid_drop(self.c_proj(attn_out))

        return attn_out, past_key_value if use_cache else None


class _GPTMLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, config.intermediate_size)
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size)

        self.drop = nn.Dropout(config.hidden_dropout_prob)
        self.act = nn.utils.get_activation_module_from_name(config.hidden_act)
        if self.act is None:
            raise ValueError(f"Invalid activation name: '{config.hidden_act}'")

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.c_proj(self.act(self.c_fc(x))))


class _GPTBlock(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = _GPTAttention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = _GPTMLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        past_key_value: nn.KVCache | None = None,
        cache_position: Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, nn.KVCache | None]:
        attn_out, present = self.attn(
            self.ln_1(hidden_states),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))

        return hidden_states, present


class _GPTDecoder(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.h = nn.ModuleList(
            [_GPTBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        past_key_values: list[nn.KVCache] | None = None,
        cache_position: Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, list[nn.KVCache] | None]:
        presents: list[nn.KVCache] | None = [] if use_cache else None
        for i, block in enumerate(self.h):
            pkv = past_key_values[i] if past_key_values is not None else None
            hidden_states, present = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=pkv,
                cache_position=cache_position,
                use_cache=use_cache,
            )
            if use_cache:
                presents.append(present)

        return hidden_states, presents


class GPT(PreTrainedModelMixin, nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = _GPTEmbedding(config)
        self.decoder = _GPTDecoder(config)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.constant(module.bias, 0.0)

        elif isinstance(module, nn.Embedding):
            nn.init.normal(module.weight, mean=0.0, std=self.config.initializer_range)

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant(module.bias, 0.0)
            nn.init.constant(module.weight, 1.0)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.tokens_embed

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embeddings.tokens_embed = value

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: list[nn.KVCache] | None = None,
        cache_position: Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, list[nn.KVCache] | None]:
        past_length = (
            past_key_values[0].get_seq_length() if past_key_values is not None else 0
        )
        hidden_states = self.embeddings(input_ids, position_ids, past_length)
        hidden_states, presents = self.decoder(
            hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
        )
        return hidden_states, presents
