from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor

__all__ = [
    "GPTConfig",
    "GPT",
    "gpt2",
    "gpt2_medium",
    "gpt2_large",
    "gpt2_xl",
]


@dataclass
class GPTConfig:
    """Configuration for GPT style language models."""

    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    bias: bool = True

    def validate(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive.")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive.")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)

        mask = lucid.ones((config.block_size, config.block_size), device="cpu")
        mask = mask.tril().reshape(1, 1, config.block_size, config.block_size)
        self.register_buffer("_causal_mask", mask)

        # Store custom init scale for projection layer similar to GPT-2.
        scale = 1 / math.sqrt(2 * config.n_layer)
        self.c_proj.setattr_raw("_init_std", 0.02 * scale)

        self.n_head = config.n_head
        self.head_dim = head_dim

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block size {self.config.block_size}."
            )

        qkv = self.c_attn(x)
        qkv = qkv.reshape(B, T, self.n_head, 3 * self.head_dim)
        qkv = qkv.transpose((0, 2, 1, 3))
        q, k, v = qkv.chunk(3, axis=-1)

        att = q @ k.transpose((0, 1, 3, 2))
        mask = self._causal_mask[:, :, :T, :T]
        att = att.masked_fill(mask == 0, -1e12)
        att = att / math.sqrt(self.head_dim)
        att = F.softmax(att, axis=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose((0, 2, 1, 3)).reshape(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        inner_dim = 4 * config.n_embd
        self.fc = nn.Linear(config.n_embd, inner_dim, bias=config.bias)
        self.act = nn.GELU()
        self.proj = nn.Linear(inner_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.resid_dropout)

        scale = 1 / math.sqrt(2 * config.n_layer)
        self.proj.setattr_raw("_init_std", 0.02 * scale)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = self.act(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(
            config.n_embd, eps=config.layer_norm_epsilon, bias=config.bias
        )
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(
            config.n_embd, eps=config.layer_norm_epsilon, bias=config.bias
        )
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT-2 style Transformer language model."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(
            config.n_embd, eps=config.layer_norm_epsilon, bias=config.bias
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.lm_head.weight = self.wte.weight

    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        if not isinstance(idx, Tensor):
            raise TypeError("Input idx must be a Tensor.")

        if idx.device != self.device:
            idx = idx.to(self.device)

        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block size {self.config.block_size}."
            )

        token_embeddings = self.wte(idx)
        position_ids = (
            lucid.arange(T, device=idx.device).astype(lucid.Int64).unsqueeze(0)
        )
        position_ids = position_ids.broadcast_to((B, T))
        position_embeddings = self.wpe(position_ids)

        x = token_embeddings + position_embeddings
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            if not isinstance(targets, Tensor):
                raise TypeError("Targets must be provided as a Tensor.")
            if targets.device != self.device:
                targets = targets.to(self.device)
            loss = self.get_loss(logits, targets)

        return logits, loss

    def get_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        if logits.shape[:2] != targets.shape:
            raise ValueError("Targets shape must match logits batch and time dimensions.")
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1).astype(lucid.Int64)
        return F.cross_entropy(logits_flat, targets_flat)

    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        do_sample: bool = True,
        eos_token_id: int | None = None,
    ) -> Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")

        if not isinstance(idx, Tensor):
            idx = lucid.tensor(idx, dtype=lucid.Int64, device=self.device)
        else:
            if idx.device != self.device:
                idx = idx.to(self.device)
            idx = idx.astype(lucid.Int64)

        generated = idx
        finished = np.zeros(idx.shape[0], dtype=bool)

        with lucid.no_grad():
            for _ in range(max_new_tokens):
                if generated.shape[1] > self.config.block_size:
                    idx_cond = generated[:, -self.config.block_size :]
                else:
                    idx_cond = generated

                logits, _ = self.forward(idx_cond)
                logits = logits[:, -1, :]
                logits = logits / max(temperature, 1e-5)

                logits_np = logits.numpy()
                if top_k is not None and top_k > 0:
                    k = min(top_k, logits_np.shape[-1])
                    kth_values = np.partition(logits_np, -k, axis=-1)[:, -k]
                    mask = logits_np < kth_values[:, None]
                    logits_np = np.where(mask, -1e9, logits_np)

                if top_p is not None and 0 < top_p < 1.0:
                    sorted_idx = np.argsort(-logits_np, axis=-1)
                    sorted_logits = np.take_along_axis(logits_np, sorted_idx, axis=-1)
                    probs = _softmax_np(sorted_logits)
                    cumulative_probs = np.cumsum(probs, axis=-1)
                    mask = cumulative_probs > top_p
                    mask[:, 1:] = mask[:, :-1]
                    mask[:, 0] = False
                    sorted_logits = np.where(mask, -1e9, sorted_logits)
                    inv_idx = np.argsort(sorted_idx, axis=-1)
                    logits_np = np.take_along_axis(sorted_logits, inv_idx, axis=-1)

                probs = _softmax_np(logits_np)
                if do_sample:
                    next_tokens = [
                        np.random.choice(probs.shape[-1], p=probs[i])
                        for i in range(probs.shape[0])
                    ]
                    next_tokens = np.array(next_tokens, dtype=np.int64)
                else:
                    next_tokens = np.argmax(probs, axis=-1)

                if eos_token_id is not None:
                    next_tokens = np.where(finished, eos_token_id, next_tokens)

                next_tensor = lucid.tensor(
                    next_tokens, dtype=lucid.Int64, device=self.device
                ).unsqueeze(1)
                generated = lucid.concatenate((generated, next_tensor), axis=1)

                if eos_token_id is not None:
                    finished = finished | (next_tokens == eos_token_id)
                    if finished.all():
                        break

        return generated


    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = getattr(module, "_init_std", 0.02)
            nn.init.normal(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                module.bias.zero()
        elif isinstance(module, nn.Embedding):
            nn.init.normal(module.weight, mean=0.0, std=0.02)


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


@register_model
def gpt2(**kwargs) -> GPT:
    config = GPTConfig(**kwargs)
    return GPT(config)


@register_model
def gpt2_medium(**kwargs) -> GPT:
    config = GPTConfig(n_layer=24, n_head=16, n_embd=1024, **kwargs)
    return GPT(config)


@register_model
def gpt2_large(**kwargs) -> GPT:
    config = GPTConfig(n_layer=36, n_head=20, n_embd=1280, **kwargs)
    return GPT(config)


@register_model
def gpt2_xl(**kwargs) -> GPT:
    config = GPTConfig(n_layer=48, n_head=25, n_embd=1600, **kwargs)
    return GPT(config)
