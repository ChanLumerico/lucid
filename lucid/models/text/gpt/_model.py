"""GPT-1 model (Radford et al., 2018) — decoder-only Transformer.

Module / parameter naming mirrors HuggingFace ``OpenAIGPTModel`` for state-
dict portability.  Layout:

    tokens_embed       (vocab → hidden)
    positions_embed    (max_pos → hidden)
    drop               (post-embedding Dropout)
    h.{i}.attn.{c_attn, c_proj}        (fused QKV + output projection)
    h.{i}.ln_1                         (pre-attention LayerNorm)
    h.{i}.mlp.{c_fc, c_proj}           (2-layer FFN, intermediate × 4)
    h.{i}.ln_2                         (pre-MLP LayerNorm)

The classifier / LM heads then sit on top of ``GPTModel``:

    GPTLMHeadModel             — tied LM head (GenerationMixin host)
    GPTForSequenceClassification — last-token classifier
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
from lucid.models.text.gpt._config import GPTConfig

# ─────────────────────────────────────────────────────────────────────────────
# Multi-head causal self-attention (fused QKV)
# ─────────────────────────────────────────────────────────────────────────────


class _GPTSelfAttention(nn.Module):
    """Fused QKV projection + causal masking.

    Parameter names match HF: ``c_attn`` is the ``(hidden → 3·hidden)`` fused
    projection, ``c_proj`` is the post-attention output projection.
    """

    causal_mask: Tensor

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = math.sqrt(self.head_dim)

        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.attn_dropout = nn.Dropout(p=config.attention_dropout)
        self.resid_dropout = nn.Dropout(p=config.hidden_dropout)

        # Precomputed lower-triangular causal mask: 1 on / below diagonal.
        # Shape (1, 1, T_max, T_max) so it broadcasts over (B, H).
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

        qkv = cast(Tensor, self.c_attn(hidden))  # (B, T, 3*hidden)
        qkv = qkv.reshape(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q: Tensor = qkv[0]
        k: Tensor = qkv[1]
        v: Tensor = qkv[2]

        scores: Tensor = q @ k.permute(0, 1, 3, 2) / self.scale  # (B, H, T, T)
        causal = self.causal_mask[:, :, :T, :T]
        # ``(1 - causal) * -1e4`` blocks attention to the upper triangle.
        scores = scores + (1.0 - causal) * -1e4
        if attention_mask is not None:
            scores = scores + attention_mask

        probs = F.softmax(scores, dim=-1)
        probs = cast(Tensor, self.attn_dropout(probs))

        out: Tensor = probs @ v  # (B, H, T, D)
        out = out.permute(0, 2, 1, 3).reshape(B, T, H * D)  # (B, T, hidden)
        out = cast(Tensor, self.c_proj(out))
        return cast(Tensor, self.resid_dropout(out))


# ─────────────────────────────────────────────────────────────────────────────
# Feed-forward block — Conv1d-style ``c_fc`` / ``c_proj`` per HF naming
# ─────────────────────────────────────────────────────────────────────────────


class _GPTMLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
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
# Transformer block — *post*-LN per GPT-1 (BERT-style), unlike GPT-2's pre-LN
# ─────────────────────────────────────────────────────────────────────────────


class _GPTBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attn = _GPTSelfAttention(config)
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = _GPTMLP(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(  # type: ignore[override]
        self,
        hidden: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        # Residual + post-LN, as in the original paper.
        a = cast(Tensor, self.attn(hidden, attention_mask=attention_mask))
        hidden = cast(Tensor, self.ln_1(hidden + a))
        m = cast(Tensor, self.mlp(hidden))
        hidden = cast(Tensor, self.ln_2(hidden + m))
        return hidden


# ─────────────────────────────────────────────────────────────────────────────
# Bare GPT trunk
# ─────────────────────────────────────────────────────────────────────────────


class GPTModel(PretrainedModel):
    """GPT-1 decoder trunk — returns the per-token hidden states."""

    config_class: ClassVar[type[GPTConfig]] = GPTConfig
    base_model_prefix: ClassVar[str] = "transformer"

    position_ids: Tensor

    def __init__(self, config: GPTConfig) -> None:
        super().__init__(config)
        self._max_pos = config.max_position_embeddings
        self.tokens_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.positions_embed = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.drop = nn.Dropout(p=config.hidden_dropout)
        self.h = nn.ModuleList(
            [_GPTBlock(config) for _ in range(config.num_hidden_layers)]
        )

        pos = lucid.arange(config.max_position_embeddings).long().unsqueeze(0)
        self.register_buffer("position_ids", pos, persistent=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.tokens_embed

    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, nn.Embedding):
            raise TypeError(
                "GPTModel input embeddings must be nn.Embedding, got "
                f"{type(value).__name__}"
            )
        self.tokens_embed = value

    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> BaseModelOutput:
        B, T = int(input_ids.shape[0]), int(input_ids.shape[1])
        if T > self._max_pos:
            raise ValueError(
                f"Sequence length {T} exceeds max_position_embeddings "
                f"{self._max_pos}"
            )

        pos_ids = self.position_ids[:, :T]
        tok_emb = cast(Tensor, self.tokens_embed(input_ids))
        pos_emb = cast(Tensor, self.positions_embed(pos_ids))
        hidden = cast(Tensor, self.drop(tok_emb + pos_emb))

        ext_mask = extended_attention_mask(attention_mask, (B, T))
        for block in self.h:
            hidden = cast(Tensor, block(hidden, attention_mask=ext_mask))

        return BaseModelOutput(last_hidden_state=hidden)


# ─────────────────────────────────────────────────────────────────────────────
# Causal-LM head — host for GenerationMixin
# ─────────────────────────────────────────────────────────────────────────────


class GPTLMHeadModel(PretrainedModel, GenerationMixin):
    """GPT-1 + tied LM head — the pre-training objective and the entry point
    for :meth:`GenerationMixin.generate`.

    HF parity:
        - ``transformer.*``  : trunk (matches HF ``transformer`` prefix)
        - ``lm_head.weight`` : tied to ``transformer.tokens_embed.weight``
    """

    config_class: ClassVar[type[GPTConfig]] = GPTConfig
    base_model_prefix: ClassVar[str] = "transformer"

    def __init__(self, config: GPTConfig) -> None:
        super().__init__(config)
        self.transformer = GPTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self._tie_lm_head_to_input_embeddings()

    def _tie_lm_head_to_input_embeddings(self) -> None:
        self.lm_head.weight = self.transformer.tokens_embed.weight

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
        hidden = outputs.last_hidden_state
        logits = cast(Tensor, self.lm_head(hidden))

        loss: Tensor | None = None
        if labels is not None:
            # Standard causal LM shift: predict token t+1 from positions [0..t].
            B, T, V = logits.shape
            shift_logits = logits[:, :-1, :].reshape((B * (T - 1), V))
            shift_labels = labels[:, 1:].reshape((B * (T - 1),)).long()
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        return CausalLMOutput(logits=logits, loss=loss)


# ─────────────────────────────────────────────────────────────────────────────
# Sequence classification — uses the LAST non-pad token's hidden state
# ─────────────────────────────────────────────────────────────────────────────


class GPTForSequenceClassification(PretrainedModel):
    """Decoder-style classifier — reads the final token's hidden vector.

    Unlike BERT (CLS-at-position-0), GPT-style classifiers pool the *last*
    real token, because every preceding position has attended to a strict
    prefix only.  When ``attention_mask`` is provided we honour it; otherwise
    the last position of every row is used.
    """

    config_class: ClassVar[type[GPTConfig]] = GPTConfig
    base_model_prefix: ClassVar[str] = "transformer"

    def __init__(self, config: GPTConfig) -> None:
        super().__init__(config)
        self.transformer = GPTModel(config)
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
        hidden = outputs.last_hidden_state  # (B, T, H)
        B, T = int(hidden.shape[0]), int(hidden.shape[1])

        if attention_mask is None:
            last_idx = [T - 1] * B
        else:
            # `.sum()` over int dtype isn't supported on the CPU backend, so
            # cast to float for the reduction and back to int for indexing.
            mask_f = attention_mask.float()
            last_idx = [int(mask_f[b].sum().item()) - 1 for b in range(B)]

        # Gather the last-token hidden state per row.
        pooled = lucid.stack(
            [hidden[b, last_idx[b], :] for b in range(B)], dim=0
        )  # (B, H)
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
class GPTDoubleHeadsOutput(ModelOutput):
    """Joint LM + multiple-choice output for :class:`GPTDoubleHeadsModel`."""

    lm_logits: Tensor
    mc_logits: Tensor
    loss: Tensor | None = None
    lm_loss: Tensor | None = None
    mc_loss: Tensor | None = None


class _GPTMultipleChoiceHead(nn.Module):
    """Pool the hidden state at ``mc_token_ids`` per choice → 1-way Linear.

    Maps ``(N, C, L, H) → (N, C)``: for each (batch, choice) pair, gather the
    hidden state at the choice-specific token (typically a CLS / final token)
    and project to a single logit.  The C-way softmax / argmax happens
    outside.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.summary = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

    def forward(self, hidden_states: Tensor, mc_token_ids: Tensor) -> Tensor:  # type: ignore[override]
        # hidden_states: (N, C, L, H); mc_token_ids: (N, C)
        N, C, L, H = hidden_states.shape
        # Gather the chosen position per (batch, choice).  Index-build in
        # Python — list comprehensions stay cheap because C is small (≤4-5).
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


class GPTDoubleHeadsModel(PretrainedModel):
    """GPT-1 + LM head + multiple-choice head — Radford 2018 fine-tuning recipe.

    Used for tasks like RACE / Story Cloze where each example presents a
    prompt followed by C candidate completions; the model is jointly trained
    with the standard LM objective on each choice and a C-way classification
    loss picking the correct completion.

    Inputs are 3-D: ``input_ids`` of shape ``(N, C, L)`` flattens to
    ``(N * C, L)`` for the trunk and reshapes back to ``(N, C, …)`` at the
    heads.  ``mc_token_ids`` of shape ``(N, C)`` indexes the per-choice
    pooling position (typically the trailing CLF token of each completion).
    """

    config_class: ClassVar[type[GPTConfig]] = GPTConfig
    base_model_prefix: ClassVar[str] = "transformer"

    def __init__(self, config: GPTConfig) -> None:
        super().__init__(config)
        self.transformer = GPTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mc_head = _GPTMultipleChoiceHead(config)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.tokens_embed.weight

    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        mc_token_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        mc_labels: Tensor | None = None,
    ) -> GPTDoubleHeadsOutput:
        if input_ids.ndim != 3:
            raise ValueError(
                f"GPTDoubleHeadsModel expects input_ids of shape (N, C, L), "
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
        hidden = outputs.last_hidden_state  # (N*C, L, H)
        H = int(hidden.shape[-1])

        lm_logits_flat = cast(Tensor, self.lm_head(hidden))  # (N*C, L, V)
        V = int(lm_logits_flat.shape[-1])
        lm_logits = lm_logits_flat.reshape(N, C, L, V)
        mc_logits = cast(
            Tensor, self.mc_head(hidden.reshape(N, C, L, H), mc_token_ids)
        )  # (N, C)

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

        return GPTDoubleHeadsOutput(
            lm_logits=lm_logits,
            mc_logits=mc_logits,
            loss=total_loss,
            lm_loss=lm_loss,
            mc_loss=mc_loss,
        )
