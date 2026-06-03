"""Pretrained-weight declarations for the RoFormer family.

Two checkpoints ship through :mod:`lucid.weights`, converted from the only
published canonical RoFormer release — ``junnyu/roformer_chinese_base``
(Su et al. 2021; Chinese, CLUECorpusSmall):

* :class:`RoFormerWeights` — the bare ``RoFormerModel`` encoder.
* :class:`RoFormerMLMWeights` — encoder + the ``cls.predictions`` masked-LM
  head (decoder tied to the word embedding).

Lucid's RoFormer uses the **interleaved** rotary convention matching the
upstream checkpoint (the original RoPE paper's pairing), so conversion is an
identity key map (split q/k/v, no transpose); the fixed sinusoidal RoPE table
and the tied duplicate decoder bias are dropped, and the untrained pooler is
filled from init.  Forward parity vs the reference framework is ≤6.4e-6.
Chinese WordPiece, 50 000-token vocab; tokenize with
:class:`lucid.models.text.roformer.RoFormerTokenizer`.
"""

from lucid.utils.transforms import Compose
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

_NOOP = Compose([])
_TAG = "CLUECORPUSSMALL"
_LICENSE = "apache-2.0"


@register_weights("roformer")
class RoFormerWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.roformer`.

    Su et al. 2021 12-layer rotary encoder trunk (``H=768``, 123.6 M params).

    Attributes
    ----------
    CLUECORPUSSMALL : WeightEntry
        CLUECorpusSmall Chinese pre-training checkpoint sourced from
        ``transformers/junnyu/roformer_chinese_base``.
    DEFAULT : WeightEntry
        Alias for :attr:`CLUECORPUSSMALL`.

    Notes
    -----
    Reference: Su, Lu, Pan, Murtadha, Wen, Liu, *"RoFormer: Enhanced
    Transformer with Rotary Position Embedding"*, 2021 (arXiv:2104.09864).

    Examples
    --------
    >>> from lucid.models import roformer
    >>> model = roformer(pretrained=True).eval()
    """

    CLUECORPUSSMALL = WeightEntry(
        url=f"{HUB_BASE}/roformer-chinese-base/resolve/main/{_TAG}/model.safetensors",
        sha256="acf2ec4f3850839bca64495cececa883cfedc21283072e07180c2e0c3841afb5",
        num_classes=768,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/junnyu/roformer_chinese_base",
            "license": _LICENSE,
            "num_params": 123_555_840,
            "file_size_mb": 473.23,
        },
    )
    DEFAULT = CLUECORPUSSMALL


@register_weights("roformer_mlm")
class RoFormerMLMWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.roformer_mlm`.

    RoFormer encoder + tied MLM head over the 50 000-token Chinese vocab.

    Attributes
    ----------
    CLUECORPUSSMALL : WeightEntry
        CLUECorpusSmall Chinese pre-training checkpoint (encoder + MLM
        head) sourced from ``transformers/junnyu/roformer_chinese_base``.
    DEFAULT : WeightEntry
        Alias for :attr:`CLUECORPUSSMALL`.

    Notes
    -----
    Reference: Su, Lu, Pan, Murtadha, Wen, Liu, *"RoFormer: Enhanced
    Transformer with Rotary Position Embedding"*, 2021 (arXiv:2104.09864).

    Examples
    --------
    >>> from lucid.models import roformer_mlm
    >>> model = roformer_mlm(pretrained=True).eval()
    """

    CLUECORPUSSMALL = WeightEntry(
        url=f"{HUB_BASE}/roformer-chinese-base-mlm/resolve/main/{_TAG}/model.safetensors",
        sha256="baf02be90f0e73bb1f8952504decb6d2b55bc974e060d8c02445a1a20b0c885b",
        num_classes=50_000,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/junnyu/roformer_chinese_base",
            "license": _LICENSE,
            "num_params": 123_555_840,
            "file_size_mb": 622.16,
        },
    )
    DEFAULT = CLUECORPUSSMALL
