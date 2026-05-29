"""Registry factories for RoFormer (Su et al., 2021).

The paper specifies a single architecture (``L=12, H=768, A=12`` — same
shape as BERT-base, with absolute PE replaced by RoPE).  We therefore
expose just ``roformer`` (plus task heads).  Test code that needs a
small variant should override config fields via
``create_model("roformer", hidden_size=..., num_hidden_layers=..., ...)``.
"""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.text.roformer._config import RoFormerConfig
from lucid.models.text.roformer._model import (
    RoFormerForMaskedLM,
    RoFormerForSequenceClassification,
    RoFormerForTokenClassification,
    RoFormerModel,
)
from lucid.models.text.roformer._weights import RoFormerMLMWeights, RoFormerWeights

_CFG_BASE = RoFormerConfig()  # paper default — the only published size


def _apply(cfg: RoFormerConfig, overrides: dict[str, object]) -> RoFormerConfig:
    return RoFormerConfig(**{**cfg.__dict__, **overrides}) if overrides else cfg


# ── Backbone ──────────────────────────────────────────────────────────────────


@register_model(  # type: ignore[arg-type]  # reason: roformer adds a typed weights= kwarg (per-model WeightsEnum); the ModelFactory protocol predates the weights system and names only pretrained + **overrides.
    task="base",
    family="roformer",
    model_type="roformer",
    model_class=RoFormerModel,
    default_config=_CFG_BASE,
)
def roformer(
    pretrained: bool | str = False,
    *,
    weights: RoFormerWeights | None = None,
    **overrides: object,
) -> RoFormerModel:
    r"""Construct a RoFormer encoder trunk.

    Canonical RoFormer architecture from Su, Lu, Pan, Murtadha, Wen, and
    Liu, 2024: :math:`L=12` transformer layers, :math:`H=768` hidden,
    :math:`A=12` attention heads, intermediate width 3072 — same shape as
    BERT-base, but with absolute learned position embeddings replaced by
    rotary position embedding (RoPE) applied to :math:`Q` and :math:`K`
    inside every self-attention layer.  Max sequence length is extended
    from 512 to 1536 to exercise RoPE's length-extrapolation property.

    Parameters
    ----------
    pretrained : bool, default=False
        Reserved for future weight registration; currently a no-op.
    **overrides : object
        Optional :class:`RoFormerConfig` field overrides (e.g.
        ``vocab_size=...``, ``rotary_base=...``,
        ``max_position_embeddings=...``) forwarded into the underlying
        config.

    Returns
    -------
    RoFormerModel
        Encoder trunk configured with the paper defaults plus any
        overrides.

    Notes
    -----
    Reference: Su, Lu, Pan, Murtadha, Wen, and Liu, *"RoFormer: Enhanced
    Transformer with Rotary Position Embedding"*, Neurocomputing,
    vol. 568, 2024, article 127063 (arXiv:2104.09864).

    Rotary rotation matrix per :math:`(\cos\theta_i, \sin\theta_i)` pair:

    .. math::

        R_\theta = \begin{pmatrix}
            \cos\theta & -\sin\theta \\
            \sin\theta & \cos\theta
        \end{pmatrix}.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.roformer import roformer
    >>> model = roformer().eval()
    >>> input_ids = lucid.tensor([[101, 7592, 2088, 102]])
    >>> out = model(input_ids)
    >>> out.last_hidden_state.shape, out.pooler_output.shape
    ((1, 4, 768), (1, 768))
    """
    entry = weights_mod.resolve_weights(RoFormerWeights, pretrained, weights)
    model = RoFormerModel(_apply(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="roformer")
    return model


# ── Task heads ────────────────────────────────────────────────────────────────


@register_model(  # type: ignore[arg-type]  # reason: roformer_mlm adds a typed weights= kwarg (per-model WeightsEnum); the ModelFactory protocol predates the weights system and names only pretrained + **overrides.
    task="masked-lm",
    family="roformer",
    model_type="roformer",
    model_class=RoFormerForMaskedLM,
    default_config=_CFG_BASE,
)
def roformer_mlm(
    pretrained: bool | str = False,
    *,
    weights: RoFormerMLMWeights | None = None,
    **overrides: object,
) -> RoFormerForMaskedLM:
    r"""Construct a RoFormer model with a tied masked-LM head.

    Same trunk as :func:`roformer` (L=12, H=768, A=12), augmented with the
    BERT-style two-layer projection mapping each hidden state to vocabulary
    logits via a decoder whose weight matrix is tied to the input
    ``word_embeddings`` table when ``config.tie_word_embeddings`` is True.
    Use for MLM pre-training when RoPE relative-position semantics are
    preferred to absolute position embeddings.

    Parameters
    ----------
    pretrained : bool, default=False
        Reserved for future weight registration; currently a no-op.
    **overrides : object
        Optional :class:`RoFormerConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    RoFormerForMaskedLM
        RoFormer trunk wrapped with the tied MLM head.

    Notes
    -----
    Reference: Su et al., *"RoFormer: Enhanced Transformer with Rotary
    Position Embedding"*, Neurocomputing, vol. 568, 2024 (arXiv:2104.09864).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.roformer import roformer_mlm
    >>> model = roformer_mlm().eval()
    >>> input_ids = lucid.tensor([[101, 7592, 103, 102]])   # [CLS] hello [MASK] [SEP]
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, 4, vocab=50000)
    (1, 4, 50000)
    """
    entry = weights_mod.resolve_weights(RoFormerMLMWeights, pretrained, weights)
    model = RoFormerForMaskedLM(_apply(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="roformer_mlm")
    return model


@register_model(
    task="sequence-classification",
    family="roformer",
    model_type="roformer",
    model_class=RoFormerForSequenceClassification,
    default_config=_CFG_BASE,
)
def roformer_cls(
    pretrained: bool = False, **overrides: object
) -> RoFormerForSequenceClassification:
    r"""Construct a RoFormer model with a sequence-classification head.

    Same trunk as :func:`roformer` (L=12, H=768, A=12), augmented with a
    dropout-regularised linear classifier on the pooled first-token
    embedding.  Use for GLUE-style fine-tunes when the RoPE relative
    position bias is preferred to absolute position embeddings.

    Parameters
    ----------
    pretrained : bool, default=False
        Reserved for future weight registration; currently a no-op.
    **overrides : object
        Optional :class:`RoFormerConfig` field overrides forwarded into the
        underlying config.  Pass ``num_labels=N`` to set the number of
        classes (default 2).

    Returns
    -------
    RoFormerForSequenceClassification
        RoFormer trunk wrapped with the pooled classifier head.

    Notes
    -----
    Reference: Su et al., *"RoFormer: Enhanced Transformer with Rotary
    Position Embedding"*, Neurocomputing, vol. 568, 2024 (arXiv:2104.09864).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.roformer import roformer_cls
    >>> model = roformer_cls(num_labels=3).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, num_labels=3)
    (1, 3)
    """
    return RoFormerForSequenceClassification(_apply(_CFG_BASE, overrides))


@register_model(
    task="token-classification",
    family="roformer",
    model_type="roformer",
    model_class=RoFormerForTokenClassification,
    default_config=_CFG_BASE,
)
def roformer_token_cls(
    pretrained: bool = False, **overrides: object
) -> RoFormerForTokenClassification:
    r"""Construct a RoFormer model with a per-token classification head.

    Same trunk as :func:`roformer` (L=12, H=768, A=12), augmented with a
    dropout-regularised per-position linear classifier of output width
    ``config.num_labels``.  Use for sequence-labelling tasks (NER, POS,
    chunking) when a RoPE-trained encoder is preferred.

    Parameters
    ----------
    pretrained : bool, default=False
        Reserved for future weight registration; currently a no-op.
    **overrides : object
        Optional :class:`RoFormerConfig` field overrides forwarded into the
        underlying config.  Pass ``num_labels=N`` to set the tag set size.

    Returns
    -------
    RoFormerForTokenClassification
        RoFormer trunk wrapped with the per-token classifier head.

    Notes
    -----
    Reference: Su et al., *"RoFormer: Enhanced Transformer with Rotary
    Position Embedding"*, Neurocomputing, vol. 568, 2024 (arXiv:2104.09864).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.roformer import roformer_token_cls
    >>> model = roformer_token_cls(num_labels=9).eval()
    >>> input_ids = lucid.tensor([[101, 2198, 7592, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, T=4, num_labels=9)
    (1, 4, 9)
    """
    return RoFormerForTokenClassification(_apply(_CFG_BASE, overrides))
