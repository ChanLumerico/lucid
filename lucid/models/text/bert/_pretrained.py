"""Registry factories for BERT variants.

Sizes follow Devlin et al. (base / large) and Turc et al. 2019
("Well-Read Students Learn Better") for the four pre-distilled sizes
(tiny / mini / small / medium).  All six base encoders and the two
masked-LM heads ship Wikipedia + BookCorpus pretrained weights through
:mod:`lucid.weights` (per-factory ``*Weights`` enums + ``weights=`` /
``pretrained=``, mirroring the ResNet wiring).  The fine-tune heads
(``*_cls`` / ``*_token_cls`` / ``*_qa``) carry no canonical *task* (GLUE /
SQuAD / NER) checkpoint, so their ``pretrained=`` selector loads the
matching pretrained **encoder** trunk (``bert_base`` / ``bert_large``)
into the ``.bert`` submodule and leaves the task head randomly
initialised — the standard fine-tuning starting point (mirrors the
reference ``AutoModelForX.from_pretrained(<encoder>)`` behaviour).
"""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.text.bert._config import BERTConfig
from lucid.models.text.bert._model import (
    BERTForMaskedLM,
    BERTForQuestionAnswering,
    BERTForSequenceClassification,
    BERTForTokenClassification,
    BERTModel,
)
from lucid.models.text.bert._weights import (
    BERTBaseMLMWeights,
    BERTBaseWeights,
    BERTLargeMLMWeights,
    BERTLargeWeights,
    BERTMediumWeights,
    BERTMiniWeights,
    BERTSmallWeights,
    BERTTinyWeights,
)

# Devlin et al. 2018 + Turc et al. 2019 size table.
_CFG_TINY = BERTConfig(
    hidden_size=128, num_hidden_layers=2, num_attention_heads=2, intermediate_size=512
)
_CFG_MINI = BERTConfig(
    hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024
)
_CFG_SMALL = BERTConfig(
    hidden_size=512, num_hidden_layers=4, num_attention_heads=8, intermediate_size=2048
)
_CFG_MEDIUM = BERTConfig(
    hidden_size=512, num_hidden_layers=8, num_attention_heads=8, intermediate_size=2048
)
_CFG_BASE = BERTConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
)
_CFG_LARGE = BERTConfig(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
)


def _apply(cfg: BERTConfig, overrides: dict[str, object]) -> BERTConfig:
    if not overrides:
        return cfg
    return BERTConfig(**{**cfg.__dict__, **overrides})


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(  # type: ignore[arg-type]  # reason: bert_tiny adds a typed weights= kwarg (per-model WeightsEnum); the ModelFactory protocol predates the weights system and names only pretrained + **overrides.
    task="base",
    family="bert",
    model_type="bert",
    model_class=BERTModel,
    default_config=_CFG_TINY,
)
def bert_tiny(
    pretrained: bool | str = False,
    *,
    weights: BERTTinyWeights | None = None,
    **overrides: object,
) -> BERTModel:
    r"""Construct a BERT-Tiny encoder.

    Smallest of the four distillation-targeted variants from Turc, Chang,
    Lee, and Toutanova, *"Well-Read Students Learn Better"*, 2019
    (arXiv:1908.08962): :math:`L=2` transformer layers, :math:`H=128` hidden,
    :math:`A=2` heads, intermediate width 512.  Roughly 4M parameters — use
    for CPU/edge inference, latency-bound demos, or unit tests of the BERT
    pipeline.

    Parameters
    ----------
    pretrained : bool, default=False
        Reserved for future weight registration.  No checkpoints are wired up
        yet, so the model is always returned with random initialisation.
    **overrides : object
        Optional :class:`BERTConfig` field overrides (e.g. ``vocab_size=...``,
        ``num_labels=...``).  Forwarded into a fresh ``BERTConfig`` whose
        defaults match the Tiny size table above.

    Returns
    -------
    BERTModel
        Encoder trunk configured with the BERT-Tiny size and any overrides.

    Notes
    -----
    Reference: Turc, Chang, Lee, and Toutanova, *"Well-Read Students Learn
    Better: On the Importance of Pre-training Compact Models"*, 2019
    (arXiv:1908.08962).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_tiny
    >>> model = bert_tiny().eval()
    >>> out = model(lucid.tensor([[101, 7592, 102]]))
    >>> out.last_hidden_state.shape   # (1, 3, 128)
    (1, 3, 128)
    """
    entry = weights_mod.resolve_weights(BERTTinyWeights, pretrained, weights)
    model = BERTModel(_apply(_CFG_TINY, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="bert_tiny")
    return model


@register_model(  # type: ignore[arg-type]  # reason: bert_mini adds a typed weights= kwarg (per-model WeightsEnum); the ModelFactory protocol predates the weights system and names only pretrained + **overrides.
    task="base",
    family="bert",
    model_type="bert",
    model_class=BERTModel,
    default_config=_CFG_MINI,
)
def bert_mini(
    pretrained: bool | str = False,
    *,
    weights: BERTMiniWeights | None = None,
    **overrides: object,
) -> BERTModel:
    r"""Construct a BERT-Mini encoder.

    Second-smallest of the Turc et al., 2019 distillation sizes:
    :math:`L=4` layers, :math:`H=256` hidden, :math:`A=4` heads, intermediate
    width 1024.  Roughly 11M parameters.

    Parameters
    ----------
    pretrained : bool, default=False
        Reserved for future weight registration; currently a no-op.
    **overrides : object
        Optional :class:`BERTConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    BERTModel
        Encoder trunk configured with the BERT-Mini size and any overrides.

    Notes
    -----
    Reference: Turc, Chang, Lee, and Toutanova, *"Well-Read Students Learn
    Better"*, 2019 (arXiv:1908.08962).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_mini
    >>> model = bert_mini().eval()
    >>> out = model(lucid.tensor([[101, 7592, 102]]))
    >>> out.last_hidden_state.shape   # (1, 3, 256)
    (1, 3, 256)
    """
    entry = weights_mod.resolve_weights(BERTMiniWeights, pretrained, weights)
    model = BERTModel(_apply(_CFG_MINI, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="bert_mini")
    return model


@register_model(  # type: ignore[arg-type]  # reason: bert_small adds a typed weights= kwarg (per-model WeightsEnum); the ModelFactory protocol predates the weights system and names only pretrained + **overrides.
    task="base",
    family="bert",
    model_type="bert",
    model_class=BERTModel,
    default_config=_CFG_SMALL,
)
def bert_small(
    pretrained: bool | str = False,
    *,
    weights: BERTSmallWeights | None = None,
    **overrides: object,
) -> BERTModel:
    r"""Construct a BERT-Small encoder.

    Distillation-targeted variant from Turc et al., 2019: :math:`L=4` layers,
    :math:`H=512` hidden, :math:`A=8` heads, intermediate width 2048.
    Roughly 29M parameters — a useful mid-point between Mini and Base.

    Parameters
    ----------
    pretrained : bool, default=False
        Reserved for future weight registration; currently a no-op.
    **overrides : object
        Optional :class:`BERTConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    BERTModel
        Encoder trunk configured with the BERT-Small size and any overrides.

    Notes
    -----
    Reference: Turc, Chang, Lee, and Toutanova, *"Well-Read Students Learn
    Better"*, 2019 (arXiv:1908.08962).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_small
    >>> model = bert_small().eval()
    >>> out = model(lucid.tensor([[101, 7592, 102]]))
    >>> out.last_hidden_state.shape   # (1, 3, 512)
    (1, 3, 512)
    """
    entry = weights_mod.resolve_weights(BERTSmallWeights, pretrained, weights)
    model = BERTModel(_apply(_CFG_SMALL, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="bert_small")
    return model


@register_model(  # type: ignore[arg-type]  # reason: bert_medium adds a typed weights= kwarg (per-model WeightsEnum); the ModelFactory protocol predates the weights system and names only pretrained + **overrides.
    task="base",
    family="bert",
    model_type="bert",
    model_class=BERTModel,
    default_config=_CFG_MEDIUM,
)
def bert_medium(
    pretrained: bool | str = False,
    *,
    weights: BERTMediumWeights | None = None,
    **overrides: object,
) -> BERTModel:
    r"""Construct a BERT-Medium encoder.

    Largest of the Turc et al., 2019 distillation sizes: :math:`L=8` layers,
    :math:`H=512` hidden, :math:`A=8` heads, intermediate width 2048.
    Roughly 41M parameters — closest in depth to Base while keeping the
    Small width.

    Parameters
    ----------
    pretrained : bool, default=False
        Reserved for future weight registration; currently a no-op.
    **overrides : object
        Optional :class:`BERTConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    BERTModel
        Encoder trunk configured with the BERT-Medium size and any overrides.

    Notes
    -----
    Reference: Turc, Chang, Lee, and Toutanova, *"Well-Read Students Learn
    Better"*, 2019 (arXiv:1908.08962).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_medium
    >>> model = bert_medium().eval()
    >>> out = model(lucid.tensor([[101, 7592, 102]]))
    >>> out.last_hidden_state.shape   # (1, 3, 512)
    (1, 3, 512)
    """
    entry = weights_mod.resolve_weights(BERTMediumWeights, pretrained, weights)
    model = BERTModel(_apply(_CFG_MEDIUM, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="bert_medium")
    return model


@register_model(  # type: ignore[arg-type]  # reason: bert_base adds a typed weights= kwarg (per-model WeightsEnum); the ModelFactory protocol predates the weights system and names only pretrained + **overrides.
    task="base",
    family="bert",
    model_type="bert",
    model_class=BERTModel,
    default_config=_CFG_BASE,
)
def bert_base(
    pretrained: bool | str = False,
    *,
    weights: BERTBaseWeights | None = None,
    **overrides: object,
) -> BERTModel:
    r"""Construct a BERT-Base encoder.

    Canonical mid-size variant from Devlin et al., 2018 Table 1:
    :math:`L=12` transformer layers, :math:`H=768` hidden, :math:`A=12`
    attention heads, intermediate width 3072 — roughly 110M parameters.
    Uses the 30,522-piece WordPiece vocabulary and a maximum sequence length
    of 512.  This is the default starting point for most BERT fine-tuning
    work.

    Parameters
    ----------
    pretrained : bool, default=False
        Reserved for future weight registration.  Returns a randomly
        initialised model today.
    **overrides : object
        Optional :class:`BERTConfig` field overrides (e.g. ``vocab_size=...``,
        ``max_position_embeddings=...``) forwarded into the underlying
        config.

    Returns
    -------
    BERTModel
        Encoder trunk configured with the BERT-Base size and any overrides.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805), Table 1 (BERT\ :sub:`BASE`).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_base
    >>> model = bert_base().eval()
    >>> input_ids = lucid.tensor([[101, 7592, 2088, 102]])   # [CLS] hello world [SEP]
    >>> out = model(input_ids)
    >>> out.last_hidden_state.shape, out.pooler_output.shape
    ((1, 4, 768), (1, 768))
    """
    entry = weights_mod.resolve_weights(BERTBaseWeights, pretrained, weights)
    model = BERTModel(_apply(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="bert_base")
    return model


@register_model(  # type: ignore[arg-type]  # reason: bert_large adds a typed weights= kwarg (per-model WeightsEnum); the ModelFactory protocol predates the weights system and names only pretrained + **overrides.
    task="base",
    family="bert",
    model_type="bert",
    model_class=BERTModel,
    default_config=_CFG_LARGE,
)
def bert_large(
    pretrained: bool | str = False,
    *,
    weights: BERTLargeWeights | None = None,
    **overrides: object,
) -> BERTModel:
    r"""Construct a BERT-Large encoder.

    Larger of the two original variants from Devlin et al., 2018 Table 1:
    :math:`L=24` transformer layers, :math:`H=1024` hidden, :math:`A=16`
    attention heads, intermediate width 4096 — roughly 340M parameters.
    Uses the 30,522-piece WordPiece vocabulary and a maximum sequence length
    of 512.  Achieved SOTA on every GLUE task and SQuAD v1.1/v2.0 at release.

    Parameters
    ----------
    pretrained : bool, default=False
        Reserved for future weight registration.  Returns a randomly
        initialised model today.
    **overrides : object
        Optional :class:`BERTConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    BERTModel
        Encoder trunk configured with the BERT-Large size and any overrides.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805), Table 1 (BERT\ :sub:`LARGE`).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_large
    >>> model = bert_large().eval()
    >>> input_ids = lucid.tensor([[101, 7592, 2088, 102]])   # [CLS] hello world [SEP]
    >>> out = model(input_ids)
    >>> out.last_hidden_state.shape, out.pooler_output.shape
    ((1, 4, 1024), (1, 1024))
    """
    entry = weights_mod.resolve_weights(BERTLargeWeights, pretrained, weights)
    model = BERTModel(_apply(_CFG_LARGE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="bert_large")
    return model


# ── Masked-LM heads ───────────────────────────────────────────────────────────


@register_model(  # type: ignore[arg-type]  # reason: bert_base_mlm adds a typed weights= kwarg (per-model WeightsEnum); the ModelFactory protocol predates the weights system and names only pretrained + **overrides.
    task="masked-lm",
    family="bert",
    model_type="bert",
    model_class=BERTForMaskedLM,
    default_config=_CFG_BASE,
)
def bert_base_mlm(
    pretrained: bool | str = False,
    *,
    weights: BERTBaseMLMWeights | None = None,
    **overrides: object,
) -> BERTForMaskedLM:
    r"""Construct a BERT-Base model with a tied masked-LM head.

    Same trunk as :func:`bert_base` (L=12, H=768, A=12, ~110M parameters),
    augmented with the masked-language-modeling prediction head used at
    pre-training time.  Suitable for continued pre-training on domain
    corpora, mask-filling inference, and reproducing the MLM half of the
    Devlin et al. (2018) pre-training recipe.

    Parameters
    ----------
    pretrained : bool, default=False
        Reserved for future weight registration; currently a no-op.
    **overrides : object
        Optional :class:`BERTConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    BERTForMaskedLM
        BERT-Base wrapped with the tied MLM head.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805) §3.1 (task #1).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_base_mlm
    >>> model = bert_base_mlm().eval()
    >>> input_ids = lucid.tensor([[101, 7592, 103, 102]])   # [CLS] hello [MASK] [SEP]
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, 4, vocab_size=30522)
    (1, 4, 30522)
    """
    entry = weights_mod.resolve_weights(BERTBaseMLMWeights, pretrained, weights)
    model = BERTForMaskedLM(_apply(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="bert_base_mlm")
    return model


@register_model(  # type: ignore[arg-type]  # reason: bert_large_mlm adds a typed weights= kwarg (per-model WeightsEnum); the ModelFactory protocol predates the weights system and names only pretrained + **overrides.
    task="masked-lm",
    family="bert",
    model_type="bert",
    model_class=BERTForMaskedLM,
    default_config=_CFG_LARGE,
)
def bert_large_mlm(
    pretrained: bool | str = False,
    *,
    weights: BERTLargeMLMWeights | None = None,
    **overrides: object,
) -> BERTForMaskedLM:
    r"""Construct a BERT-Large model with a tied masked-LM head.

    Same trunk as :func:`bert_large` (L=24, H=1024, A=16, ~340M parameters),
    augmented with the masked-language-modeling prediction head used at
    pre-training time.

    Parameters
    ----------
    pretrained : bool, default=False
        Reserved for future weight registration; currently a no-op.
    **overrides : object
        Optional :class:`BERTConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    BERTForMaskedLM
        BERT-Large wrapped with the tied MLM head.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805) §3.1 (task #1).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_large_mlm
    >>> model = bert_large_mlm().eval()
    >>> input_ids = lucid.tensor([[101, 7592, 103, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, 4, 30522)
    (1, 4, 30522)
    """
    entry = weights_mod.resolve_weights(BERTLargeMLMWeights, pretrained, weights)
    model = BERTForMaskedLM(_apply(_CFG_LARGE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="bert_large_mlm")
    return model


# ── Sequence / token / QA classification heads ────────────────────────────────


@register_model(  # type: ignore[arg-type]  # reason: bert_base_cls adds a typed weights= kwarg (the encoder BERTBaseWeights); the ModelFactory protocol predates the weights system and names only pretrained + **overrides.
    task="sequence-classification",
    family="bert",
    model_type="bert",
    model_class=BERTForSequenceClassification,
    default_config=_CFG_BASE,
)
def bert_base_cls(
    pretrained: bool | str = False,
    *,
    weights: BERTBaseWeights | None = None,
    **overrides: object,
) -> BERTForSequenceClassification:
    r"""Construct a BERT-Base model with a sequence-classification head.

    Same trunk as :func:`bert_base` (L=12, H=768, A=12), augmented with a
    pooled-CLS linear classifier of output width ``config.num_labels``.  The
    canonical fine-tuning recipe for GLUE-style single-sentence and
    sentence-pair tasks (SST-2, MNLI, QQP, RTE, MRPC, STS-B).

    Parameters
    ----------
    pretrained : bool or str, default=False
        Encoder-weight selector.  ``False`` → fully random init; ``True``
        → loads the pretrained :func:`bert_base` encoder
        (:attr:`BERTBaseWeights.DEFAULT`) into the ``.bert`` trunk; a tag
        string selects a specific encoder checkpoint.  **The classification
        head is always randomly initialised** — this is the standard
        fine-tuning starting point (no GLUE-fine-tuned head ships), so
        train on a downstream task before inference.
    weights : BERTBaseWeights, optional, keyword-only
        Explicit encoder-weights enum member; takes precedence over
        ``pretrained``.
    **overrides : object
        Optional :class:`BERTConfig` field overrides forwarded into the
        underlying config.  Pass ``num_labels=N`` to set the number of
        classes (default 2).  Overrides that change the encoder shape are
        incompatible with loading pretrained encoder weights.

    Returns
    -------
    BERTForSequenceClassification
        BERT-Base wrapped with the pooled classifier head (encoder
        pretrained when requested; head random).

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805) §4.1.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_base_cls
    >>> model = bert_base_cls(num_labels=3).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, num_labels=3)
    (1, 3)
    """
    entry = weights_mod.resolve_weights(BERTBaseWeights, pretrained, weights)
    model = BERTForSequenceClassification(_apply(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model.bert, entry, name="bert_base")
    return model


@register_model(  # type: ignore[arg-type]  # reason: bert_large_cls adds a typed weights= kwarg (the encoder BERTLargeWeights); the ModelFactory protocol predates the weights system and names only pretrained + **overrides.
    task="sequence-classification",
    family="bert",
    model_type="bert",
    model_class=BERTForSequenceClassification,
    default_config=_CFG_LARGE,
)
def bert_large_cls(
    pretrained: bool | str = False,
    *,
    weights: BERTLargeWeights | None = None,
    **overrides: object,
) -> BERTForSequenceClassification:
    r"""Construct a BERT-Large model with a sequence-classification head.

    Same trunk as :func:`bert_large` (L=24, H=1024, A=16, ~340M parameters),
    augmented with a pooled-CLS linear classifier of output width
    ``config.num_labels``.  Use for high-accuracy GLUE-style fine-tunes when
    compute permits.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Encoder-weight selector.  ``False`` → fully random init; ``True``
        → loads the pretrained :func:`bert_large` encoder
        (:attr:`BERTLargeWeights.DEFAULT`) into the ``.bert`` trunk; a tag
        string selects a specific encoder checkpoint.  **The classification
        head is always randomly initialised** (fine-tuning starting point).
    weights : BERTLargeWeights, optional, keyword-only
        Explicit encoder-weights enum member; takes precedence over
        ``pretrained``.
    **overrides : object
        Optional :class:`BERTConfig` field overrides forwarded into the
        underlying config.  Pass ``num_labels=N`` to set the number of
        classes (default 2).  Overrides that change the encoder shape are
        incompatible with loading pretrained encoder weights.

    Returns
    -------
    BERTForSequenceClassification
        BERT-Large wrapped with the pooled classifier head (encoder
        pretrained when requested; head random).

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805) §4.1.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_large_cls
    >>> model = bert_large_cls(num_labels=2).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, 2)
    (1, 2)
    """
    entry = weights_mod.resolve_weights(BERTLargeWeights, pretrained, weights)
    model = BERTForSequenceClassification(_apply(_CFG_LARGE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model.bert, entry, name="bert_large")
    return model


@register_model(  # type: ignore[arg-type]  # reason: bert_base_token_cls adds a typed weights= kwarg (the encoder BERTBaseWeights); the ModelFactory protocol predates the weights system and names only pretrained + **overrides.
    task="token-classification",
    family="bert",
    model_type="bert",
    model_class=BERTForTokenClassification,
    default_config=_CFG_BASE,
)
def bert_base_token_cls(
    pretrained: bool | str = False,
    *,
    weights: BERTBaseWeights | None = None,
    **overrides: object,
) -> BERTForTokenClassification:
    r"""Construct a BERT-Base model with a per-token classification head.

    Same trunk as :func:`bert_base` (L=12, H=768, A=12), augmented with a
    per-position linear classifier of output width ``config.num_labels``.
    The canonical fine-tuning recipe for sequence-labelling tasks: named-
    entity recognition (CoNLL-2003), part-of-speech tagging, chunking.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Encoder-weight selector.  ``False`` → fully random init; ``True``
        → loads the pretrained :func:`bert_base` encoder
        (:attr:`BERTBaseWeights.DEFAULT`) into the ``.bert`` trunk; a tag
        string selects a specific encoder checkpoint.  **The per-token
        classifier head is always randomly initialised** (fine-tuning
        starting point — no NER/POS-fine-tuned head ships).
    weights : BERTBaseWeights, optional, keyword-only
        Explicit encoder-weights enum member; takes precedence over
        ``pretrained``.
    **overrides : object
        Optional :class:`BERTConfig` field overrides forwarded into the
        underlying config.  Pass ``num_labels=N`` to set the tag set size.
        Overrides that change the encoder shape are incompatible with
        loading pretrained encoder weights.

    Returns
    -------
    BERTForTokenClassification
        BERT-Base wrapped with the per-token classifier head (encoder
        pretrained when requested; head random).

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805) §4.3.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_base_token_cls
    >>> model = bert_base_token_cls(num_labels=9).eval()   # e.g. CoNLL-2003 BIO tag set
    >>> input_ids = lucid.tensor([[101, 2198, 7592, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, T=4, num_labels=9)
    (1, 4, 9)
    """
    entry = weights_mod.resolve_weights(BERTBaseWeights, pretrained, weights)
    model = BERTForTokenClassification(_apply(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model.bert, entry, name="bert_base")
    return model


@register_model(  # type: ignore[arg-type]  # reason: bert_base_qa adds a typed weights= kwarg (the encoder BERTBaseWeights); the ModelFactory protocol predates the weights system and names only pretrained + **overrides.
    task="question-answering",
    family="bert",
    model_type="bert",
    model_class=BERTForQuestionAnswering,
    default_config=_CFG_BASE,
)
def bert_base_qa(
    pretrained: bool | str = False,
    *,
    weights: BERTBaseWeights | None = None,
    **overrides: object,
) -> BERTForQuestionAnswering:
    r"""Construct a BERT-Base model with an extractive-QA span head.

    Same trunk as :func:`bert_base` (L=12, H=768, A=12), augmented with a
    2-way linear head producing start- and end-position logits over each
    input token.  The canonical fine-tuning recipe for SQuAD v1.1 and v2.0.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Encoder-weight selector.  ``False`` → fully random init; ``True``
        → loads the pretrained :func:`bert_base` encoder
        (:attr:`BERTBaseWeights.DEFAULT`) into the ``.bert`` trunk; a tag
        string selects a specific encoder checkpoint.  **The span-prediction
        head is always randomly initialised** (fine-tuning starting point —
        no SQuAD-fine-tuned head ships).
    weights : BERTBaseWeights, optional, keyword-only
        Explicit encoder-weights enum member; takes precedence over
        ``pretrained``.
    **overrides : object
        Optional :class:`BERTConfig` field overrides forwarded into the
        underlying config.  ``num_labels`` is ignored by this head.
        Overrides that change the encoder shape are incompatible with
        loading pretrained encoder weights.

    Returns
    -------
    BERTForQuestionAnswering
        BERT-Base wrapped with the span-prediction head (encoder pretrained
        when requested; head random).

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805) §4.2.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_base_qa
    >>> model = bert_base_qa().eval()
    >>> # question + context (CLS q [SEP] c [SEP]) ids
    >>> input_ids = lucid.tensor([[101, 2040, 102, 1045, 2572, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, T=6, 2) — last dim is (start, end)
    (1, 6, 2)
    """
    entry = weights_mod.resolve_weights(BERTBaseWeights, pretrained, weights)
    model = BERTForQuestionAnswering(_apply(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model.bert, entry, name="bert_base")
    return model
