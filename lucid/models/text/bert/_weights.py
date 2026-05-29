"""Pretrained-weight declarations for the BERT family.

Eight checkpoints ship through the :mod:`lucid.weights` system, all
converted from the upstream Hugging Face checkpoints (a pure identity
parameter map — Lucid mirrors the ``BERTModel`` naming one-for-one):

* **Base encoders** (:class:`BERTTinyWeights` … :class:`BERTLargeWeights`)
  — the bare encoder trunk.  The four miniatures are the Turc et al. 2019
  "Well-Read Students" pre-distilled sizes; ``base`` / ``large`` are the
  original Devlin et al. 2018 ``bert-*-uncased`` checkpoints.
* **Masked-LM heads** (:class:`BERTBaseMLMWeights`,
  :class:`BERTLargeMLMWeights`) — encoder + the ``cls.predictions`` head
  used for the pre-training objective, re-tied to the input embedding.

All checkpoints were pre-trained on the Wikipedia + BookCorpus corpus
(uncased WordPiece, 30 522-token vocabulary).  Text models consume token
ids directly, so the entries carry a no-op preprocessing transform; tokenize
inputs with the matching :class:`lucid.models.text.bert.BERTTokenizer`.
"""

from lucid.utils.transforms import Compose
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

# Text preprocessing is tokenization (a lucid.utils.tokenizer concern), not a
# tensor transform — entries carry an explicit no-op so `.transforms()` is safe.
_NOOP = Compose([])

_TAG = "WIKIPEDIA_BOOKSCORPUS"
_LICENSE = "apache-2.0"


def _url(slug: str) -> str:
    return f"{HUB_BASE}/{slug}/resolve/main/{_TAG}/model.safetensors"


@register_weights("bert_tiny")
class BERTTinyWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.bert_tiny`."""

    WIKIPEDIA_BOOKSCORPUS = WeightEntry(
        url=_url("bert-tiny"),
        sha256="3572b8818d3e4e9e46507b327224348f32d838bd7e75bff140dc5728d4ff6cc2",
        num_classes=128,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/google/bert_uncased_L-2_H-128_A-2",
            "license": _LICENSE,
            "num_params": 4_385_920,
            "file_size_mb": 16.74,
        },
    )
    DEFAULT = WIKIPEDIA_BOOKSCORPUS


@register_weights("bert_mini")
class BERTMiniWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.bert_mini`."""

    WIKIPEDIA_BOOKSCORPUS = WeightEntry(
        url=_url("bert-mini"),
        sha256="c644db652fbc91ae2e2d55a9efe8fb0b5f582448420093ba5c65cff7bcb7b5d9",
        num_classes=256,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/google/bert_uncased_L-4_H-256_A-4",
            "license": _LICENSE,
            "num_params": 11_171_328,
            "file_size_mb": 42.62,
        },
    )
    DEFAULT = WIKIPEDIA_BOOKSCORPUS


@register_weights("bert_small")
class BERTSmallWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.bert_small`."""

    WIKIPEDIA_BOOKSCORPUS = WeightEntry(
        url=_url("bert-small"),
        sha256="ff63fc91d94ae3cb1240f7ad147db7d6f45958c544ee00c1c9e264cb45675979",
        num_classes=512,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/google/bert_uncased_L-4_H-512_A-8",
            "license": _LICENSE,
            "num_params": 28_763_648,
            "file_size_mb": 109.73,
        },
    )
    DEFAULT = WIKIPEDIA_BOOKSCORPUS


@register_weights("bert_medium")
class BERTMediumWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.bert_medium`."""

    WIKIPEDIA_BOOKSCORPUS = WeightEntry(
        url=_url("bert-medium"),
        sha256="2edd0b9805b731de1c625308449f6e8cb8487c1982bbff22cf2f0a07abbdf886",
        num_classes=512,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/google/bert_uncased_L-8_H-512_A-8",
            "license": _LICENSE,
            "num_params": 41_373_184,
            "file_size_mb": 157.84,
        },
    )
    DEFAULT = WIKIPEDIA_BOOKSCORPUS


@register_weights("bert_base")
class BERTBaseWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.bert_base`."""

    WIKIPEDIA_BOOKSCORPUS = WeightEntry(
        url=_url("bert-base"),
        sha256="a4faff5f2aab76df5bc433bef9b3f4e7e8628855885bf54256614095fa814a53",
        num_classes=768,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/google-bert/bert-base-uncased",
            "license": _LICENSE,
            "num_params": 109_482_240,
            "file_size_mb": 417.66,
        },
    )
    DEFAULT = WIKIPEDIA_BOOKSCORPUS


@register_weights("bert_large")
class BERTLargeWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.bert_large`."""

    WIKIPEDIA_BOOKSCORPUS = WeightEntry(
        url=_url("bert-large"),
        sha256="e15f6d8cc7bb0babc43228c83cde9a3e41ccdb3c066e78b8b071b1a16dacebb2",
        num_classes=1024,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/google-bert/bert-large-uncased",
            "license": _LICENSE,
            "num_params": 335_141_888,
            "file_size_mb": 1278.51,
        },
    )
    DEFAULT = WIKIPEDIA_BOOKSCORPUS


@register_weights("bert_base_mlm")
class BERTBaseMLMWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.bert_base_mlm`."""

    WIKIPEDIA_BOOKSCORPUS = WeightEntry(
        url=_url("bert-base-mlm"),
        sha256="76c4910e80b5c31bee8cab879dc647a3976052ebe32b592cd522103a88b7b282",
        num_classes=30_522,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/google-bert/bert-base-uncased",
            "license": _LICENSE,
            "num_params": 109_514_298,
            "file_size_mb": 509.46,
        },
    )
    DEFAULT = WIKIPEDIA_BOOKSCORPUS


@register_weights("bert_large_mlm")
class BERTLargeMLMWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.bert_large_mlm`."""

    WIKIPEDIA_BOOKSCORPUS = WeightEntry(
        url=_url("bert-large-mlm"),
        sha256="22b8affd371ec095d2b93fb25b4b16380fa7cd6395a7033aa07ca485df7d5a78",
        num_classes=30_522,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/google-bert/bert-large-uncased",
            "license": _LICENSE,
            "num_params": 335_174_586,
            "file_size_mb": 1401.87,
        },
    )
    DEFAULT = WIKIPEDIA_BOOKSCORPUS
