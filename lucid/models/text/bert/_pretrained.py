"""Registry factories for BERT variants.

Sizes follow Devlin et al. (base / large) and Turc et al. 2019
("Well-Read Students Learn Better") for the four pre-distilled sizes
(tiny / mini / small / medium).  No pretrained weight URLs are registered yet
— follow-up work wires Hugging Face checkpoints through
:mod:`lucid.weights` (a ``BertBaseWeights`` enum + ``weights=`` /
``pretrained=`` on the factories, mirroring the ResNet wiring).
"""

from lucid.models._registry import register_model
from lucid.models.text.bert._config import BertConfig
from lucid.models.text.bert._model import (
    BertForMaskedLM,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertModel,
)

# Devlin et al. 2018 + Turc et al. 2019 size table.
_CFG_TINY = BertConfig(
    hidden_size=128, num_hidden_layers=2, num_attention_heads=2, intermediate_size=512
)
_CFG_MINI = BertConfig(
    hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024
)
_CFG_SMALL = BertConfig(
    hidden_size=512, num_hidden_layers=4, num_attention_heads=8, intermediate_size=2048
)
_CFG_MEDIUM = BertConfig(
    hidden_size=512, num_hidden_layers=8, num_attention_heads=8, intermediate_size=2048
)
_CFG_BASE = BertConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
)
_CFG_LARGE = BertConfig(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
)


def _apply(cfg: BertConfig, overrides: dict[str, object]) -> BertConfig:
    if not overrides:
        return cfg
    return BertConfig(**{**cfg.__dict__, **overrides})


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="bert",
    model_type="bert",
    model_class=BertModel,
    default_config=_CFG_TINY,
)
def bert_tiny(pretrained: bool = False, **overrides: object) -> BertModel:
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
        Optional :class:`BertConfig` field overrides (e.g. ``vocab_size=...``,
        ``num_labels=...``).  Forwarded into a fresh ``BertConfig`` whose
        defaults match the Tiny size table above.

    Returns
    -------
    BertModel
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
    return BertModel(_apply(_CFG_TINY, overrides))


@register_model(
    task="base",
    family="bert",
    model_type="bert",
    model_class=BertModel,
    default_config=_CFG_MINI,
)
def bert_mini(pretrained: bool = False, **overrides: object) -> BertModel:
    r"""Construct a BERT-Mini encoder.

    Second-smallest of the Turc et al., 2019 distillation sizes:
    :math:`L=4` layers, :math:`H=256` hidden, :math:`A=4` heads, intermediate
    width 1024.  Roughly 11M parameters.

    Parameters
    ----------
    pretrained : bool, default=False
        Reserved for future weight registration; currently a no-op.
    **overrides : object
        Optional :class:`BertConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    BertModel
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
    return BertModel(_apply(_CFG_MINI, overrides))


@register_model(
    task="base",
    family="bert",
    model_type="bert",
    model_class=BertModel,
    default_config=_CFG_SMALL,
)
def bert_small(pretrained: bool = False, **overrides: object) -> BertModel:
    r"""Construct a BERT-Small encoder.

    Distillation-targeted variant from Turc et al., 2019: :math:`L=4` layers,
    :math:`H=512` hidden, :math:`A=8` heads, intermediate width 2048.
    Roughly 29M parameters — a useful mid-point between Mini and Base.

    Parameters
    ----------
    pretrained : bool, default=False
        Reserved for future weight registration; currently a no-op.
    **overrides : object
        Optional :class:`BertConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    BertModel
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
    return BertModel(_apply(_CFG_SMALL, overrides))


@register_model(
    task="base",
    family="bert",
    model_type="bert",
    model_class=BertModel,
    default_config=_CFG_MEDIUM,
)
def bert_medium(pretrained: bool = False, **overrides: object) -> BertModel:
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
        Optional :class:`BertConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    BertModel
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
    return BertModel(_apply(_CFG_MEDIUM, overrides))


@register_model(
    task="base",
    family="bert",
    model_type="bert",
    model_class=BertModel,
    default_config=_CFG_BASE,
)
def bert_base(pretrained: bool = False, **overrides: object) -> BertModel:
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
        Optional :class:`BertConfig` field overrides (e.g. ``vocab_size=...``,
        ``max_position_embeddings=...``) forwarded into the underlying
        config.

    Returns
    -------
    BertModel
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
    return BertModel(_apply(_CFG_BASE, overrides))


@register_model(
    task="base",
    family="bert",
    model_type="bert",
    model_class=BertModel,
    default_config=_CFG_LARGE,
)
def bert_large(pretrained: bool = False, **overrides: object) -> BertModel:
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
        Optional :class:`BertConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    BertModel
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
    return BertModel(_apply(_CFG_LARGE, overrides))


# ── Masked-LM heads ───────────────────────────────────────────────────────────


@register_model(
    task="masked-lm",
    family="bert",
    model_type="bert",
    model_class=BertForMaskedLM,
    default_config=_CFG_BASE,
)
def bert_base_mlm(pretrained: bool = False, **overrides: object) -> BertForMaskedLM:
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
        Optional :class:`BertConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    BertForMaskedLM
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
    return BertForMaskedLM(_apply(_CFG_BASE, overrides))


@register_model(
    task="masked-lm",
    family="bert",
    model_type="bert",
    model_class=BertForMaskedLM,
    default_config=_CFG_LARGE,
)
def bert_large_mlm(pretrained: bool = False, **overrides: object) -> BertForMaskedLM:
    r"""Construct a BERT-Large model with a tied masked-LM head.

    Same trunk as :func:`bert_large` (L=24, H=1024, A=16, ~340M parameters),
    augmented with the masked-language-modeling prediction head used at
    pre-training time.

    Parameters
    ----------
    pretrained : bool, default=False
        Reserved for future weight registration; currently a no-op.
    **overrides : object
        Optional :class:`BertConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    BertForMaskedLM
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
    return BertForMaskedLM(_apply(_CFG_LARGE, overrides))


# ── Sequence / token / QA classification heads ────────────────────────────────


@register_model(
    task="sequence-classification",
    family="bert",
    model_type="bert",
    model_class=BertForSequenceClassification,
    default_config=_CFG_BASE,
)
def bert_base_cls(
    pretrained: bool = False, **overrides: object
) -> BertForSequenceClassification:
    r"""Construct a BERT-Base model with a sequence-classification head.

    Same trunk as :func:`bert_base` (L=12, H=768, A=12), augmented with a
    pooled-CLS linear classifier of output width ``config.num_labels``.  The
    canonical fine-tuning recipe for GLUE-style single-sentence and
    sentence-pair tasks (SST-2, MNLI, QQP, RTE, MRPC, STS-B).

    Parameters
    ----------
    pretrained : bool, default=False
        Reserved for future weight registration; currently a no-op.
    **overrides : object
        Optional :class:`BertConfig` field overrides forwarded into the
        underlying config.  Pass ``num_labels=N`` to set the number of
        classes (default 2).

    Returns
    -------
    BertForSequenceClassification
        BERT-Base wrapped with the pooled classifier head.

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
    return BertForSequenceClassification(_apply(_CFG_BASE, overrides))


@register_model(
    task="sequence-classification",
    family="bert",
    model_type="bert",
    model_class=BertForSequenceClassification,
    default_config=_CFG_LARGE,
)
def bert_large_cls(
    pretrained: bool = False, **overrides: object
) -> BertForSequenceClassification:
    r"""Construct a BERT-Large model with a sequence-classification head.

    Same trunk as :func:`bert_large` (L=24, H=1024, A=16, ~340M parameters),
    augmented with a pooled-CLS linear classifier of output width
    ``config.num_labels``.  Use for high-accuracy GLUE-style fine-tunes when
    compute permits.

    Parameters
    ----------
    pretrained : bool, default=False
        Reserved for future weight registration; currently a no-op.
    **overrides : object
        Optional :class:`BertConfig` field overrides forwarded into the
        underlying config.  Pass ``num_labels=N`` to set the number of
        classes (default 2).

    Returns
    -------
    BertForSequenceClassification
        BERT-Large wrapped with the pooled classifier head.

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
    return BertForSequenceClassification(_apply(_CFG_LARGE, overrides))


@register_model(
    task="token-classification",
    family="bert",
    model_type="bert",
    model_class=BertForTokenClassification,
    default_config=_CFG_BASE,
)
def bert_base_token_cls(
    pretrained: bool = False, **overrides: object
) -> BertForTokenClassification:
    r"""Construct a BERT-Base model with a per-token classification head.

    Same trunk as :func:`bert_base` (L=12, H=768, A=12), augmented with a
    per-position linear classifier of output width ``config.num_labels``.
    The canonical fine-tuning recipe for sequence-labelling tasks: named-
    entity recognition (CoNLL-2003), part-of-speech tagging, chunking.

    Parameters
    ----------
    pretrained : bool, default=False
        Reserved for future weight registration; currently a no-op.
    **overrides : object
        Optional :class:`BertConfig` field overrides forwarded into the
        underlying config.  Pass ``num_labels=N`` to set the tag set size.

    Returns
    -------
    BertForTokenClassification
        BERT-Base wrapped with the per-token classifier head.

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
    return BertForTokenClassification(_apply(_CFG_BASE, overrides))


@register_model(
    task="question-answering",
    family="bert",
    model_type="bert",
    model_class=BertForQuestionAnswering,
    default_config=_CFG_BASE,
)
def bert_base_qa(
    pretrained: bool = False, **overrides: object
) -> BertForQuestionAnswering:
    r"""Construct a BERT-Base model with an extractive-QA span head.

    Same trunk as :func:`bert_base` (L=12, H=768, A=12), augmented with a
    2-way linear head producing start- and end-position logits over each
    input token.  The canonical fine-tuning recipe for SQuAD v1.1 and v2.0.

    Parameters
    ----------
    pretrained : bool, default=False
        Reserved for future weight registration; currently a no-op.
    **overrides : object
        Optional :class:`BertConfig` field overrides forwarded into the
        underlying config.  ``num_labels`` is ignored by this head.

    Returns
    -------
    BertForQuestionAnswering
        BERT-Base wrapped with the span-prediction head.

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
    return BertForQuestionAnswering(_apply(_CFG_BASE, overrides))
