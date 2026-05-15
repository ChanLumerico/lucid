"""GPT-2 model (Radford et al., 2019) — pre-LN decoder-only Transformer.

Differences from :mod:`lucid.models.text.gpt._model` (which implements GPT-1):

    * **Pre-LN block** — LayerNorm is applied **before** the attention / MLP
      sub-layer, with the residual flow remaining un-normalised.  This is the
      change that made >24-layer stacks trainable without warmup tricks.
    * **Final ``ln_f``** — one extra LayerNorm at the top of the trunk, after
      the last block.
    * **Bigger vocab + longer context** — knob change only.
    * **Init scaling** — output projections in attention / MLP have their
      weight std multiplied by ``1 / sqrt(2N)``.  We apply this only when
      ``config.scale_residual_init`` is True (matches HF default).

HuggingFace parity (state-dict keys):
    transformer.wte                — token embedding   (was ``tokens_embed``)
    transformer.wpe                — position embedding (was ``positions_embed``)
    transformer.h.{i}.ln_1
    transformer.h.{i}.attn.{c_attn, c_proj}
    transformer.h.{i}.ln_2
    transformer.h.{i}.mlp.{c_fc, c_proj}
    transformer.ln_f               — final LayerNorm
    lm_head.weight                 — tied to wte.weight
"""

import math
from dataclasses import dataclass
from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import GenerationMixin
from lucid.models._output import (
    BaseModelOutput,
    CausalLMOutput,
    MaskedLMOutput,
    ModelOutput,
)
from lucid.models._utils._text import extended_attention_mask, text_activation
from lucid.models.text.gpt2._config import GPT2Config

# ─────────────────────────────────────────────────────────────────────────────
# Multi-head causal self-attention (fused QKV) — identical shape to GPT-1
# ─────────────────────────────────────────────────────────────────────────────


class _GPT2SelfAttention(nn.Module):
    causal_mask: Tensor

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = math.sqrt(self.head_dim)

        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.attn_dropout = nn.Dropout(p=config.attention_dropout)
        self.resid_dropout = nn.Dropout(p=config.hidden_dropout)

        T = config.max_position_embeddings
        mask = lucid.tril(lucid.ones((T, T))).reshape(1, 1, T, T)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(  # type: ignore[override]
        self,
        hidden: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        B, T, _ = hidden.shape
        H, D = self.num_heads, self.head_dim

        qkv = cast(Tensor, self.c_attn(hidden))
        qkv = qkv.reshape(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores: Tensor = q @ k.permute(0, 1, 3, 2) / self.scale
        causal = self.causal_mask[:, :, :T, :T]
        scores = scores + (1.0 - causal) * -1e4
        if attention_mask is not None:
            scores = scores + attention_mask

        probs = F.softmax(scores, dim=-1)
        probs = cast(Tensor, self.attn_dropout(probs))

        out: Tensor = probs @ v
        out = out.permute(0, 2, 1, 3).reshape(B, T, H * D)
        out = cast(Tensor, self.c_proj(out))
        return cast(Tensor, self.resid_dropout(out))


# ─────────────────────────────────────────────────────────────────────────────
# Feed-forward block — same shape as GPT-1's, separate class for HF parity
# ─────────────────────────────────────────────────────────────────────────────


class _GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, config.intermediate_size)
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout)
        self._act_name = config.hidden_act

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        h = text_activation(self._act_name, cast(Tensor, self.c_fc(x)))
        h = cast(Tensor, self.c_proj(h))
        return cast(Tensor, self.dropout(h))


# ─────────────────────────────────────────────────────────────────────────────
# Pre-LN transformer block — the GPT-2 invention
# ─────────────────────────────────────────────────────────────────────────────


class _GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = _GPT2SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = _GPT2MLP(config)

    def forward(  # type: ignore[override]
        self,
        hidden: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        # x = x + Attn(LN(x))   — pre-LN: residual carries un-normalised state
        a = cast(
            Tensor,
            self.attn(cast(Tensor, self.ln_1(hidden)), attention_mask=attention_mask),
        )
        hidden = hidden + a
        m = cast(Tensor, self.mlp(cast(Tensor, self.ln_2(hidden))))
        hidden = hidden + m
        return hidden


# ─────────────────────────────────────────────────────────────────────────────
# Bare GPT-2 trunk
# ─────────────────────────────────────────────────────────────────────────────


def _scale_residual_init(model: nn.Module, num_layers: int) -> None:
    """Re-initialise the post-residual projection weights with std scaled by
    ``1 / sqrt(2 * N)``.

    Matches the trick from GPT-2 §2.3 ("Modified initialization") — keeps
    activation variance bounded as depth grows.  Targets every ``c_proj``
    inside attention / MLP sub-blocks.
    """
    factor = 1.0 / math.sqrt(2.0 * float(num_layers))
    for name, module in model.named_modules():
        if name.endswith(".attn.c_proj") or name.endswith(".mlp.c_proj"):
            assert isinstance(module, nn.Linear)
            # HF uses std=0.02 for every Linear by default; we already have
            # something Glorot-ish from nn.Linear's reset_parameters.  Re-init
            # ``c_proj`` weights to N(0, (0.02 * factor)²) so the residual
            # stream stays variance-stable at depth.
            nn.init.normal_(module.weight, mean=0.0, std=0.02 * factor)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class GPT2Model(PretrainedModel):
    """GPT-2 trunk — N pre-LN blocks topped by ``ln_f``."""

    config_class: ClassVar[type[GPT2Config]] = GPT2Config
    base_model_prefix: ClassVar[str] = "transformer"

    position_ids: Tensor

    def __init__(self, config: GPT2Config) -> None:
        super().__init__(config)
        self._max_pos = config.max_position_embeddings
        # HF naming: ``wte`` / ``wpe`` / ``h`` / ``ln_f`` — keep verbatim.
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(p=config.hidden_dropout)
        self.h = nn.ModuleList(
            [_GPT2Block(config) for _ in range(config.num_hidden_layers)]
        )
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        pos = lucid.arange(config.max_position_embeddings).long().unsqueeze(0)
        self.register_buffer("position_ids", pos, persistent=False)

        if config.scale_residual_init:
            _scale_residual_init(self, config.num_hidden_layers)

    def get_input_embeddings(self) -> nn.Module:
        return self.wte

    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, nn.Embedding):
            raise TypeError(
                f"GPT2Model input embeddings must be nn.Embedding, got {type(value).__name__}"
            )
        self.wte = value

    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> BaseModelOutput:
        B, T = int(input_ids.shape[0]), int(input_ids.shape[1])
        if T > self._max_pos:
            raise ValueError(
                f"Sequence length {T} exceeds max_position_embeddings {self._max_pos}"
            )

        pos_ids = self.position_ids[:, :T]
        tok_emb = cast(Tensor, self.wte(input_ids))
        pos_emb = cast(Tensor, self.wpe(pos_ids))
        hidden = cast(Tensor, self.drop(tok_emb + pos_emb))

        ext_mask = extended_attention_mask(attention_mask, (B, T))
        for block in self.h:
            hidden = cast(Tensor, block(hidden, attention_mask=ext_mask))

        hidden = cast(Tensor, self.ln_f(hidden))
        return BaseModelOutput(last_hidden_state=hidden)


# ─────────────────────────────────────────────────────────────────────────────
# Causal-LM head — GenerationMixin host
# ─────────────────────────────────────────────────────────────────────────────


class GPT2LMHeadModel(PretrainedModel, GenerationMixin):
    """GPT-2 + tied LM head.  Entry point for :meth:`GenerationMixin.generate`."""

    config_class: ClassVar[type[GPT2Config]] = GPT2Config
    base_model_prefix: ClassVar[str] = "transformer"

    def __init__(self, config: GPT2Config) -> None:
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self._tie_lm_head_to_input_embeddings()

    def _tie_lm_head_to_input_embeddings(self) -> None:
        self.lm_head.weight = self.transformer.wte.weight

    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> CausalLMOutput:
        outputs = cast(
            BaseModelOutput,
            self.transformer(input_ids, attention_mask=attention_mask),
        )
        logits = cast(Tensor, self.lm_head(outputs.last_hidden_state))

        loss: Tensor | None = None
        if labels is not None:
            B, T, V = logits.shape
            shift_logits = logits[:, :-1, :].reshape((B * (T - 1), V))
            shift_labels = labels[:, 1:].reshape((B * (T - 1),)).long()
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        return CausalLMOutput(logits=logits, loss=loss)


# ─────────────────────────────────────────────────────────────────────────────
# Sequence-classification — last-token pool (decoder semantics, like GPT-1)
# ─────────────────────────────────────────────────────────────────────────────


class GPT2ForSequenceClassification(PretrainedModel):
    config_class: ClassVar[type[GPT2Config]] = GPT2Config
    base_model_prefix: ClassVar[str] = "transformer"

    def __init__(self, config: GPT2Config) -> None:
        super().__init__(config)
        self.transformer = GPT2Model(config)
        drop = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout
        )
        self.dropout = nn.Dropout(p=drop)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=False)

    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> MaskedLMOutput:
        outputs = cast(
            BaseModelOutput,
            self.transformer(input_ids, attention_mask=attention_mask),
        )
        hidden = outputs.last_hidden_state
        B, T = int(hidden.shape[0]), int(hidden.shape[1])

        if attention_mask is None:
            last_idx = [T - 1] * B
        else:
            mask_f = attention_mask.float()
            last_idx = [int(mask_f[b].sum().item()) - 1 for b in range(B)]

        pooled = lucid.stack([hidden[b, last_idx[b], :] for b in range(B)], dim=0)
        pooled = cast(Tensor, self.dropout(pooled))
        logits = cast(Tensor, self.classifier(pooled))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.long())

        return MaskedLMOutput(logits=logits, loss=loss)


# ─────────────────────────────────────────────────────────────────────────────
# DoubleHeadsModel — LM head + multiple-choice classification head
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GPT2DoubleHeadsOutput(ModelOutput):
    """Joint LM + multiple-choice output for :class:`GPT2DoubleHeadsModel`."""

    lm_logits: Tensor
    mc_logits: Tensor
    loss: Tensor | None = None
    lm_loss: Tensor | None = None
    mc_loss: Tensor | None = None


class _GPT2MultipleChoiceHead(nn.Module):
    """Per-choice pooled-token classifier (analogous to GPT-1's variant)."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.summary = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

    def forward(self, hidden_states: Tensor, mc_token_ids: Tensor) -> Tensor:  # type: ignore[override]
        N, C, L, H = hidden_states.shape
        pooled_rows: list[list[list[float]]] = []
        for n in range(N):
            choices: list[list[float]] = []
            for c in range(C):
                t = int(mc_token_ids[n, c].item())
                choices.append(
                    [float(hidden_states[n, c, t, h].item()) for h in range(H)]
                )
            pooled_rows.append(choices)
        pooled = lucid.tensor(
            pooled_rows, device=hidden_states.device.type
        )  # (N, C, H)
        pooled = cast(Tensor, self.dropout(pooled))
        return cast(Tensor, self.summary(pooled)).reshape(N, C)


class GPT2DoubleHeadsModel(PretrainedModel):
    """GPT-2 + LM head + multiple-choice head.

    Identical contract to :class:`GPTDoubleHeadsModel` but with GPT-2's
    pre-LN trunk and HF-style ``transformer.wte`` parameter naming so
    checkpoints port directly.
    """

    config_class: ClassVar[type[GPT2Config]] = GPT2Config
    base_model_prefix: ClassVar[str] = "transformer"

    def __init__(self, config: GPT2Config) -> None:
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mc_head = _GPT2MultipleChoiceHead(config)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.wte.weight

    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        mc_token_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        mc_labels: Tensor | None = None,
    ) -> GPT2DoubleHeadsOutput:
        if input_ids.ndim != 3:
            raise ValueError(
                f"GPT2DoubleHeadsModel expects input_ids of shape (N, C, L), "
                f"got {tuple(input_ids.shape)}"
            )
        N = int(input_ids.shape[0])
        C = int(input_ids.shape[1])
        L = int(input_ids.shape[2])

        flat_ids = input_ids.reshape(N * C, L)
        flat_mask = (
            attention_mask.reshape(N * C, L) if attention_mask is not None else None
        )
        outputs = cast(
            BaseModelOutput,
            self.transformer(flat_ids, attention_mask=flat_mask),
        )
        hidden = outputs.last_hidden_state
        H = int(hidden.shape[-1])

        lm_logits_flat = cast(Tensor, self.lm_head(hidden))
        V = int(lm_logits_flat.shape[-1])
        lm_logits = lm_logits_flat.reshape(N, C, L, V)
        mc_logits = cast(Tensor, self.mc_head(hidden.reshape(N, C, L, H), mc_token_ids))

        lm_loss: Tensor | None = None
        mc_loss: Tensor | None = None
        if labels is not None:
            shift_logits = lm_logits[:, :, :-1, :].reshape(N * C * (L - 1), V)
            shift_labels = labels[:, :, 1:].reshape(N * C * (L - 1)).long()
            lm_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        if mc_labels is not None:
            mc_loss = F.cross_entropy(mc_logits, mc_labels.long())

        total_loss: Tensor | None = None
        if lm_loss is not None and mc_loss is not None:
            total_loss = lm_loss + mc_loss
        elif lm_loss is not None:
            total_loss = lm_loss
        elif mc_loss is not None:
            total_loss = mc_loss

        return GPT2DoubleHeadsOutput(
            lm_logits=lm_logits,
            mc_logits=mc_logits,
            loss=total_loss,
            lm_loss=lm_loss,
            mc_loss=mc_loss,
        )
