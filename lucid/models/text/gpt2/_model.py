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
from typing import ClassVar, cast, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import CausalLMMixin
from lucid.models._output import (
    BaseModelOutput,
    CausalLMOutput,
    MaskedLMOutput,
    ModelOutput,
)
from lucid.models._utils._text import extended_attention_mask, text_activation
from lucid.models.text.gpt2._config import GPT2Config
from lucid.utils.cache import Cache, DynamicCache, StaticCache

# ─────────────────────────────────────────────────────────────────────────────
# Multi-head causal self-attention (fused QKV) — identical shape to GPT-1
# ─────────────────────────────────────────────────────────────────────────────


@final
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

    @override
    def forward(  # type: ignore[override]
        self,
        hidden: Tensor,
        attention_mask: Tensor | None = None,
        *,
        past_key_value: Cache | None = None,
        layer_idx: int = 0,
        cache_position: Tensor | None = None,
        use_cache: bool = False,
    ) -> Tensor:
        B, T, _ = hidden.shape
        H, D = self.num_heads, self.head_dim

        qkv = cast(Tensor, self.c_attn(hidden))
        qkv = qkv.reshape(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, T, D)

        # KV cache: prepend the accumulated past and store the new tokens.  The
        # query stays length T (the new tokens); keys/values span the full
        # history so the new tokens attend over everything seen so far.
        past_len = 0
        if past_key_value is not None:
            past_len = past_key_value.get_seq_length(layer_idx)
            # cache_position is consumed by StaticCache (the in-place write
            # location); DynamicCache ignores it and simply appends.
            cache_kwargs: dict[str, object] = {"cache_position": cache_position}
            if isinstance(past_key_value, StaticCache):
                # Attend over only the filled prefix (``past_len + T``), not the
                # full max_cache_len buffer — the latter is max_cache_len/(filled+1)×
                # wasted q·kᵀ.  StaticCache.update narrows its returned view to it.
                cache_kwargs["read_len"] = past_len + T
            k, v = past_key_value.update(k, v, layer_idx, cache_kwargs=cache_kwargs)
        t_total = int(
            k.shape[2]
        )  # DynamicCache: past_len+T; StaticCache: max_cache_len

        scores: Tensor = q @ k.permute(0, 1, 3, 2) / self.scale  # (B, H, T, t_total)
        if isinstance(past_key_value, StaticCache) and cache_position is not None:
            # StaticCache returns the fixed max_len buffer, and the write
            # position is dynamic.  Row p of the lower-triangular causal_mask is
            # exactly the allowed-key pattern for a query at absolute position p
            # (keys j <= p), so gather the rows named by cache_position.  This
            # keeps the compiled graph position-agnostic and avoids ``arange``
            # (which has no MPSGraph emitter).  Bit-identical to the eager slice.
            causal = self.causal_mask.index_select(2, cache_position)[:, :, :, :t_total]
        else:
            # Causal slice for query positions [past_len, past_len+T) over keys
            # [0, t_total); with no cache (past_len=0, t_total=T) this is the
            # original [:, :, :T, :T] block.
            causal = self.causal_mask[:, :, past_len : past_len + T, :t_total]
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


@final
class _GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, config.intermediate_size)
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout)
        self._act_name = config.hidden_act

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        h = text_activation(self._act_name, cast(Tensor, self.c_fc(x)))
        h = cast(Tensor, self.c_proj(h))
        return cast(Tensor, self.dropout(h))


# ─────────────────────────────────────────────────────────────────────────────
# Pre-LN transformer block — the GPT-2 invention
# ─────────────────────────────────────────────────────────────────────────────


@final
class _GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = _GPT2SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = _GPT2MLP(config)

    @override
    def forward(  # type: ignore[override]
        self,
        hidden: Tensor,
        attention_mask: Tensor | None = None,
        *,
        past_key_value: Cache | None = None,
        layer_idx: int = 0,
        cache_position: Tensor | None = None,
        use_cache: bool = False,
    ) -> Tensor:
        # x = x + Attn(LN(x))   — pre-LN: residual carries un-normalised state
        a = cast(
            Tensor,
            self.attn(
                cast(Tensor, self.ln_1(hidden)),
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                layer_idx=layer_idx,
                cache_position=cache_position,
                use_cache=use_cache,
            ),
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
    r"""GPT-2 decoder trunk — N pre-LN transformer blocks topped by ``ln_f``.

    Implements the modified architecture of Radford, Wu, Child, Luan, Amodei,
    and Sutskever, 2019.  Differs from GPT-1 in three ways: (i) **pre-LN
    blocks** — LayerNorm is applied to the residual *input* of each
    sub-layer rather than after the residual add, which stabilises
    optimisation past depth :math:`L \geq 24`; (ii) a **final LayerNorm**
    (``ln_f``) caps the trunk; (iii) residual-output projection weights are
    initialised with an extra :math:`1/\sqrt{2N}` factor (Radford 2019
    §2.3) to keep activation variance roughly invariant to depth.

    Parameters
    ----------
    config : GPT2Config
        Hyperparameters controlling vocabulary, depth, width, head count,
        and the residual-init scaling toggle.

    Attributes
    ----------
    wte : nn.Embedding
        Word / token embedding table of shape ``(vocab_size, hidden_size)``.
        HuggingFace name retained verbatim for state-dict portability.
    wpe : nn.Embedding
        Learned absolute position embedding of shape
        ``(max_position_embeddings, hidden_size)``.
    drop : nn.Dropout
        Post-embedding dropout layer.
    h : nn.ModuleList
        Stack of ``config.num_hidden_layers`` pre-LN transformer blocks.
    ln_f : nn.LayerNorm
        Final LayerNorm applied to the trunk output.
    config_class : type[GPT2Config]
        Registry pointer for matching-config instantiation from disk.
    base_model_prefix : str
        State-dict prefix (``"transformer"``) under which the trunk is
        nested in task-head variants.

    Notes
    -----
    Reference: Radford, Wu, Child, Luan, Amodei, and Sutskever, *"Language
    Models are Unsupervised Multitask Learners"*, OpenAI Technical Report,
    2019.

    Each pre-LN block computes

    .. math::

        \begin{aligned}
            x &\leftarrow x + \mathrm{Attn}(\mathrm{LN}_1(x)) \\
            x &\leftarrow x + \mathrm{MLP}(\mathrm{LN}_2(x))
        \end{aligned}

    so the residual stream carries un-normalised state.  The trunk produces
    final hidden states :math:`H \in \mathbb{R}^{B \times T \times H}`
    which the LM head projects to vocabulary logits.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.gpt2 import GPT2Config, GPT2Model
    >>> cfg = GPT2Config(num_hidden_layers=2, hidden_size=128,
    ...                  num_attention_heads=2, intermediate_size=512)
    >>> model = GPT2Model(cfg).eval()
    >>> input_ids = lucid.tensor([[15496, 11, 995, 13]])   # "Hello, world."
    >>> out = model(input_ids)
    >>> out.last_hidden_state.shape   # (B=1, T=4, H=128)
    (1, 4, 128)
    """

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

    @override
    def get_input_embeddings(self) -> nn.Module:
        return self.wte

    @override
    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, nn.Embedding):
            raise TypeError(
                f"GPT2Model input embeddings must be nn.Embedding, got {type(value).__name__}"
            )
        self.wte = value

    @override
    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        cache_position: Tensor | None = None,
    ) -> BaseModelOutput:
        B, T = int(input_ids.shape[0]), int(input_ids.shape[1])
        past_len = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        if past_len + T > self._max_pos:
            raise ValueError(
                f"Sequence length {past_len + T} exceeds "
                f"max_position_embeddings {self._max_pos}"
            )

        # Absolute write positions (past_len, past_len+1, ...).  Defaulted here so
        # the position embedding and the cache write agree.
        if cache_position is None:
            cache_position = lucid.arange(
                past_len, past_len + T, device=input_ids.device.type
            ).long()
        # StaticCache drives the position embedding off the runtime
        # cache_position tensor (keeps the compiled graph position-agnostic);
        # otherwise slice the static position-id buffer by the Python offset.
        if isinstance(past_key_values, StaticCache):
            pos_ids = cache_position.reshape(1, T)
        else:
            pos_ids = self.position_ids[:, past_len : past_len + T]
        tok_emb = cast(Tensor, self.wte(input_ids))
        pos_emb = cast(Tensor, self.wpe(pos_ids))
        hidden = cast(Tensor, self.drop(tok_emb + pos_emb))

        ext_mask = extended_attention_mask(attention_mask, (B, T))
        for layer_idx, block in enumerate(self.h):
            hidden = cast(
                Tensor,
                block(
                    hidden,
                    attention_mask=ext_mask,
                    past_key_value=past_key_values,
                    layer_idx=layer_idx,
                    cache_position=cache_position,
                    use_cache=use_cache,
                ),
            )

        hidden = cast(Tensor, self.ln_f(hidden))
        return BaseModelOutput(last_hidden_state=hidden)


# ─────────────────────────────────────────────────────────────────────────────
# Causal-LM head — CausalLMMixin host
# ─────────────────────────────────────────────────────────────────────────────


class GPT2LMHeadModel(PretrainedModel, CausalLMMixin):
    r"""GPT-2 with a tied causal-language-modeling head.

    Wraps :class:`GPT2Model` with an output linear projection whose weight
    matrix is bound to the input ``wte`` embedding table when
    ``config.tie_word_embeddings`` is ``True`` — halving the parameter cost
    of the softmax layer.  This is the entry point for
    :meth:`lucid.models.CausalLMMixin.generate` and is the
    standard recipe for both pre-training and downstream generative use.

    Parameters
    ----------
    config : GPT2Config
        Hyperparameters.  ``config.tie_word_embeddings`` (default True)
        controls whether the LM-head weight is shared with the input
        embedding matrix.

    Attributes
    ----------
    transformer : GPT2Model
        Underlying decoder trunk (HF naming: ``transformer.*``).
    lm_head : nn.Linear
        Output projection of shape ``(hidden_size, vocab_size)`` mapping
        each token's hidden state to vocabulary logits.

    Notes
    -----
    Reference: Radford, Wu, Child, Luan, Amodei, and Sutskever, *"Language
    Models are Unsupervised Multitask Learners"*, OpenAI Technical Report,
    2019.

    The training loss (when ``labels`` is supplied) is the next-token
    shifted causal-LM cross-entropy:

    .. math::

        \mathcal{L}_{\mathrm{CLM}}
            = -\frac{1}{B(T-1)} \sum_{b, t}
              \log p_\theta(y_{b, t+1} \mid x_{b, \le t}),

    with positions labelled ``-100`` excluded.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.gpt2 import GPT2Config, GPT2LMHeadModel
    >>> cfg = GPT2Config(num_hidden_layers=2, hidden_size=128,
    ...                  num_attention_heads=2, intermediate_size=512)
    >>> model = GPT2LMHeadModel(cfg).eval()
    >>> input_ids = lucid.tensor([[15496, 11, 995, 13]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, T=4, V=50257)
    (1, 4, 50257)
    """

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

    @override
    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        *,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        cache_position: Tensor | None = None,
    ) -> CausalLMOutput:
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        outputs = cast(
            BaseModelOutput,
            self.transformer(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
            ),
        )
        logits = cast(Tensor, self.lm_head(outputs.last_hidden_state))

        loss: Tensor | None = None
        if labels is not None:
            B, T, V = logits.shape
            shift_logits = logits[:, :-1, :].reshape((B * (T - 1), V))
            shift_labels = labels[:, 1:].reshape((B * (T - 1),)).long()
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        return CausalLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=past_key_values if use_cache else None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sequence-classification — last-token pool (decoder semantics, like GPT-1)
# ─────────────────────────────────────────────────────────────────────────────


class GPT2ForSequenceClassification(PretrainedModel):
    r"""GPT-2 with a last-token sequence-classification head.

    Wraps :class:`GPT2Model` with a dropout-regularised linear classifier
    operating on the **rightmost non-pad token's** hidden state — the
    canonical decoder-style pooling.  Unlike BERT (CLS at position 0),
    decoder LMs only have a meaningful summary at the *last* attended
    position since every preceding position has attended to a strict
    prefix.

    Parameters
    ----------
    config : GPT2Config
        Hyperparameters.  ``config.num_labels`` sets the output dimension;
        ``config.classifier_dropout`` (falling back to ``hidden_dropout``)
        sets the dropout applied before the linear.

    Attributes
    ----------
    transformer : GPT2Model
        Underlying decoder trunk.
    dropout : nn.Dropout
        Dropout layer applied to the pooled last-token embedding.
    classifier : nn.Linear
        Final linear of shape ``(hidden_size, num_labels)`` producing
        per-class logits.

    Notes
    -----
    Reference: Radford et al., *"Language Models are Unsupervised
    Multitask Learners"*, OpenAI Technical Report, 2019.

    Pooling picks index :math:`t^\star = \sum_t \mathbf{1}[\mathrm{mask}_t]
    - 1` (the rightmost real token) per row, then computes

    .. math::

        z = W_{\mathrm{cls}}\, \mathrm{Dropout}(h_{t^\star}).

    When ``labels`` is supplied, cross-entropy over ``num_labels`` classes
    is computed and exposed as ``output.loss``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.gpt2 import GPT2Config, GPT2ForSequenceClassification
    >>> cfg = GPT2Config(num_labels=3, num_hidden_layers=2, hidden_size=128,
    ...                  num_attention_heads=2, intermediate_size=512)
    >>> model = GPT2ForSequenceClassification(cfg).eval()
    >>> input_ids = lucid.tensor([[15496, 11, 995, 13]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, num_labels=3)
    (1, 3)
    """

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

    @override
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


@dataclass(slots=True)
class GPT2DoubleHeadsOutput(ModelOutput):
    r"""Joint LM + multiple-choice output for :class:`GPT2DoubleHeadsModel`.

    Aggregates the per-choice causal-LM logits, the per-choice multiple-
    choice classification logits, and the optional per-objective and
    combined losses produced when training Radford-style multiple-choice
    fine-tunes (Story Cloze, RACE).

    Parameters
    ----------
    lm_logits : Tensor
        LM head logits of shape ``(N, C, L, vocab_size)`` — one
        distribution per token of every batch x choice pair.
    mc_logits : Tensor
        Multiple-choice head logits of shape ``(N, C)`` — one scalar per
        candidate completion.
    loss : Tensor or None, default=None
        Sum of ``lm_loss`` and ``mc_loss`` when both are available;
        otherwise the single available loss, or ``None``.
    lm_loss : Tensor or None, default=None
        Shift-cross-entropy on the LM head when ``labels`` is supplied.
    mc_loss : Tensor or None, default=None
        Cross-entropy on the multiple-choice head when ``mc_labels`` is
        supplied.

    Notes
    -----
    Reference: Radford et al., *"Language Models are Unsupervised
    Multitask Learners"*, OpenAI Technical Report, 2019.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.gpt2 import GPT2Config, GPT2DoubleHeadsModel
    >>> cfg = GPT2Config(num_hidden_layers=2, hidden_size=64,
    ...                  num_attention_heads=2, intermediate_size=256)
    >>> model = GPT2DoubleHeadsModel(cfg).eval()
    >>> input_ids = lucid.tensor([[[15496, 11, 995, 13]]])    # (N=1, C=1, L=4)
    >>> mc_ids = lucid.tensor([[3]]).long()
    >>> out = model(input_ids, mc_token_ids=mc_ids)
    >>> out.lm_logits.shape, out.mc_logits.shape
    ((1, 1, 4, 50257), (1, 1))
    """

    lm_logits: Tensor
    mc_logits: Tensor
    loss: Tensor | None = None
    lm_loss: Tensor | None = None
    mc_loss: Tensor | None = None


@final
class _GPT2MultipleChoiceHead(nn.Module):
    """Per-choice pooled-token classifier (analogous to GPT-1's variant)."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.summary = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

    @override
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
    r"""GPT-2 + LM head + multiple-choice classification head.

    Identical contract to :class:`GPTDoubleHeadsModel` but with GPT-2's
    pre-LN trunk and ``transformer.wte`` parameter naming so checkpoints
    port directly.  Used for tasks like RACE / Story Cloze where each
    example presents a prompt followed by ``C`` candidate completions;
    the model is jointly trained with the standard LM objective on each
    choice and a ``C``-way classification loss picking the correct one.

    Inputs are 3-D: ``input_ids`` of shape ``(N, C, L)`` flattens to
    ``(N * C, L)`` for the trunk and reshapes back to ``(N, C, ...)`` at
    the heads.  ``mc_token_ids`` of shape ``(N, C)`` indexes the per-choice
    pooling position (typically the trailing CLF token of each completion).

    Parameters
    ----------
    config : GPT2Config
        Hyperparameters.  ``config.tie_word_embeddings`` (default True)
        ties the LM head weight to the ``wte`` embedding matrix.

    Attributes
    ----------
    transformer : GPT2Model
        Underlying decoder trunk.
    lm_head : nn.Linear
        Tied output projection of shape ``(hidden_size, vocab_size)``.
    mc_head : nn.Module
        Per-choice pooled-token scalar projection of shape
        ``(hidden_size, 1)``.

    Notes
    -----
    Reference: Radford et al., *"Language Models are Unsupervised
    Multitask Learners"*, OpenAI Technical Report, 2019; multiple-choice
    head follows the GPT-1 fine-tuning recipe (Radford 2018 §3.3).

    The combined loss when both objectives are supplied is

    .. math::

        \mathcal{L}_{\mathrm{joint}}
            = \mathcal{L}_{\mathrm{CLM}} + \mathcal{L}_{\mathrm{MC}},

    where :math:`\mathcal{L}_{\mathrm{MC}}` is a softmax cross-entropy
    over the ``C`` candidate completions.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.gpt2 import GPT2Config, GPT2DoubleHeadsModel
    >>> cfg = GPT2Config(num_hidden_layers=2, hidden_size=64,
    ...                  num_attention_heads=2, intermediate_size=256)
    >>> model = GPT2DoubleHeadsModel(cfg).eval()
    >>> input_ids = lucid.tensor([[[15496, 11, 995, 13],
    ...                            [15496, 11, 1820, 13]]])    # (N=1, C=2, L=4)
    >>> mc_ids = lucid.tensor([[3, 3]]).long()
    >>> out = model(input_ids, mc_token_ids=mc_ids)
    >>> out.mc_logits.shape   # (N=1, C=2)
    (1, 2)
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

    @override
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
