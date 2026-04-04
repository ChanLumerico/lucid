from dataclasses import dataclass

import lucid.nn as nn

from lucid._tensor import Tensor
from lucid import register_model
from lucid.models.text.gpt import GPT, GPTConfig

__all__ = [
    "GPT2Config",
    "GPT2",
    "gpt2_small",
    "gpt2_medium",
    "gpt2_large",
    "gpt2_xlarge",
]


@dataclass
class GPT2Config(GPTConfig):
    vocab_size: int = 50257
    max_position_embeddings: int = 1024
    hidden_act: str = "gelu"

    @classmethod
    def small(cls, **kwargs) -> GPT2Config:
        defaults = dict(
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=12,
            intermediate_size=3072,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def medium(cls, **kwargs) -> GPT2Config:
        defaults = dict(
            hidden_size=1024,
            num_attention_heads=16,
            num_hidden_layers=24,
            intermediate_size=4096,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def large(cls, **kwargs) -> GPT2Config:
        defaults = dict(
            hidden_size=1280,
            num_attention_heads=20,
            num_hidden_layers=36,
            intermediate_size=5120,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def xl(cls, **kwargs) -> GPT2Config:
        defaults = dict(
            hidden_size=1600,
            num_attention_heads=25,
            num_hidden_layers=48,
            intermediate_size=6400,
        )
        defaults.update(kwargs)
        return cls(**defaults)


class GPT2(GPT):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__(config)
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        nn.init.constant(self.ln_f.bias, 0.0)
        nn.init.constant(self.ln_f.weight, 1.0)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: list[nn.KVCache] | None = None,
        cache_position: Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, list[nn.KVCache] | None]:
        hidden_states, presents = super().forward(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            cache_position,
            use_cache,
        )
        return self.ln_f(hidden_states), presents


@register_model
def gpt2_small(**kwargs) -> GPT2:
    return GPT2(GPT2Config.small(**kwargs))


@register_model
def gpt2_medium(**kwargs) -> GPT2:
    return GPT2(GPT2Config.medium(**kwargs))


@register_model
def gpt2_large(**kwargs) -> GPT2:
    return GPT2(GPT2Config.large(**kwargs))


@register_model
def gpt2_xlarge(**kwargs) -> GPT2:
    return GPT2(GPT2Config.xl(**kwargs))
