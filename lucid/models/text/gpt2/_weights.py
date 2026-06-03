"""Pretrained-weight declarations for the GPT-2 family.

Eight checkpoints ship through :mod:`lucid.weights`, converted from the
OpenAI GPT-2 release (Radford et al. 2019) re-hosted on the Hub:

* **Transformers** (:class:`GPT2SmallWeights` … :class:`GPT2XLargeWeights`)
  — the bare ``GPT2Model`` trunk.
* **Causal-LM heads** (:class:`GPT2SmallLMWeights` …
  :class:`GPT2XLargeLMWeights`) — trunk + ``lm_head`` tied to ``wte``.

Conversion is an identity key map plus the Conv1D→Linear weight transpose
(HF stores the attention/MLP projection weights transposed); forward parity
vs the reference framework is ≤2e-4.  All checkpoints were trained on
WebText (byte-level BPE, 50 257-token vocab); tokenize with
:class:`lucid.models.text.gpt2.GPT2Tokenizer`.
"""

from lucid.utils.transforms import Compose
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

# Text preprocessing is tokenization (a lucid.utils.tokenizer concern), not a
# tensor transform — entries carry an explicit no-op so `.transforms()` is safe.
_NOOP = Compose([])

_TAG = "WEBTEXT"
_LICENSE = "mit"


def _url(slug: str) -> str:
    return f"{HUB_BASE}/{slug}/resolve/main/{_TAG}/model.safetensors"


@register_weights("gpt2_small")
class GPT2SmallWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.gpt2_small`.

    Radford et al. 2019 12-layer trunk (``H=768, A=12``, 124 M params).

    Attributes
    ----------
    WEBTEXT : WeightEntry
        WebText pre-training checkpoint sourced from ``transformers/gpt2``.
    DEFAULT : WeightEntry
        Alias for :attr:`WEBTEXT`.

    Notes
    -----
    Reference: Radford, Wu, Child, Luan, Amodei, Sutskever, *"Language
    Models are Unsupervised Multitask Learners"*, OpenAI 2019.

    Examples
    --------
    >>> from lucid.models import gpt2_small
    >>> model = gpt2_small(pretrained=True).eval()
    """

    WEBTEXT = WeightEntry(
        url=_url("gpt2-small"),
        sha256="b5b049cdf4a2cb055d8b9fd98ae99f66ffb500975e3b415511890c9ca0322f56",
        num_classes=768,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/gpt2",
            "license": _LICENSE,
            "num_params": 124_439_808,
            "file_size_mb": 474.71,
        },
    )
    DEFAULT = WEBTEXT


@register_weights("gpt2_medium")
class GPT2MediumWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.gpt2_medium`.

    Radford et al. 2019 24-layer trunk (``H=1024, A=16``, 355 M params).

    Attributes
    ----------
    WEBTEXT : WeightEntry
        WebText pre-training checkpoint sourced from
        ``transformers/gpt2-medium``.
    DEFAULT : WeightEntry
        Alias for :attr:`WEBTEXT`.

    Notes
    -----
    Reference: Radford, Wu, Child, Luan, Amodei, Sutskever, *"Language
    Models are Unsupervised Multitask Learners"*, OpenAI 2019.

    Examples
    --------
    >>> from lucid.models import gpt2_medium
    >>> model = gpt2_medium(pretrained=True).eval()
    """

    WEBTEXT = WeightEntry(
        url=_url("gpt2-medium"),
        sha256="ed6e74ece94bf1c83d433c78e1e8c05fcb7e69ae4f1f2b78b92f8a5851079caf",
        num_classes=1024,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/gpt2-medium",
            "license": _LICENSE,
            "num_params": 354_823_168,
            "file_size_mb": 1353.57,
        },
    )
    DEFAULT = WEBTEXT


@register_weights("gpt2_large")
class GPT2LargeWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.gpt2_large`.

    Radford et al. 2019 36-layer trunk (``H=1280, A=20``, 774 M params).

    Attributes
    ----------
    WEBTEXT : WeightEntry
        WebText pre-training checkpoint sourced from
        ``transformers/gpt2-large``.
    DEFAULT : WeightEntry
        Alias for :attr:`WEBTEXT`.

    Notes
    -----
    Reference: Radford, Wu, Child, Luan, Amodei, Sutskever, *"Language
    Models are Unsupervised Multitask Learners"*, OpenAI 2019.

    Examples
    --------
    >>> from lucid.models import gpt2_large
    >>> model = gpt2_large(pretrained=True).eval()
    """

    WEBTEXT = WeightEntry(
        url=_url("gpt2-large"),
        sha256="34ba8e6c21e540475a7cd96d8d6c2219a2e375b04e3ad88aa0a124cc80ff32c9",
        num_classes=1280,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/gpt2-large",
            "license": _LICENSE,
            "num_params": 774_030_080,
            "file_size_mb": 2952.73,
        },
    )
    DEFAULT = WEBTEXT


@register_weights("gpt2_xlarge")
class GPT2XLargeWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.gpt2_xlarge`.

    Radford et al. 2019 48-layer trunk (``H=1600, A=25``, 1.56 B params).

    Attributes
    ----------
    WEBTEXT : WeightEntry
        WebText pre-training checkpoint sourced from
        ``transformers/gpt2-xl``.
    DEFAULT : WeightEntry
        Alias for :attr:`WEBTEXT`.

    Notes
    -----
    Reference: Radford, Wu, Child, Luan, Amodei, Sutskever, *"Language
    Models are Unsupervised Multitask Learners"*, OpenAI 2019.

    Examples
    --------
    >>> from lucid.models import gpt2_xlarge
    >>> model = gpt2_xlarge(pretrained=True).eval()
    """

    WEBTEXT = WeightEntry(
        url=_url("gpt2-xlarge"),
        sha256="c3ced2dc7d6a6e04f7ecfc8c79aa5864a5a74059f8990b94bcacadcd02e6321e",
        num_classes=1600,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/gpt2-xl",
            "license": _LICENSE,
            "num_params": 1_557_611_200,
            "file_size_mb": 5941.87,
        },
    )
    DEFAULT = WEBTEXT


@register_weights("gpt2_small_lm")
class GPT2SmallLMWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.gpt2_small_lm`.

    GPT-2 Small trunk + tied ``lm_head`` over the 50 257-token BPE vocab.

    Attributes
    ----------
    WEBTEXT : WeightEntry
        WebText pre-training checkpoint (trunk + LM head) sourced from
        ``transformers/gpt2``.
    DEFAULT : WeightEntry
        Alias for :attr:`WEBTEXT`.

    Notes
    -----
    Reference: Radford, Wu, Child, Luan, Amodei, Sutskever, *"Language
    Models are Unsupervised Multitask Learners"*, OpenAI 2019.

    Examples
    --------
    >>> from lucid.models import gpt2_small_lm
    >>> model = gpt2_small_lm(pretrained=True).eval()
    """

    WEBTEXT = WeightEntry(
        url=_url("gpt2-small-lm"),
        sha256="68a979997a5d078bb9ccc6dab74000f2643e2c53beb091d2f6b6be819ecf8998",
        num_classes=50_257,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/gpt2",
            "license": _LICENSE,
            "num_params": 124_439_808,
            "file_size_mb": 621.95,
        },
    )
    DEFAULT = WEBTEXT


@register_weights("gpt2_medium_lm")
class GPT2MediumLMWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.gpt2_medium_lm`.

    GPT-2 Medium trunk + tied ``lm_head`` over the 50 257-token BPE vocab.

    Attributes
    ----------
    WEBTEXT : WeightEntry
        WebText pre-training checkpoint (trunk + LM head) sourced from
        ``transformers/gpt2-medium``.
    DEFAULT : WeightEntry
        Alias for :attr:`WEBTEXT`.

    Notes
    -----
    Reference: Radford, Wu, Child, Luan, Amodei, Sutskever, *"Language
    Models are Unsupervised Multitask Learners"*, OpenAI 2019.

    Examples
    --------
    >>> from lucid.models import gpt2_medium_lm
    >>> model = gpt2_medium_lm(pretrained=True).eval()
    """

    WEBTEXT = WeightEntry(
        url=_url("gpt2-medium-lm"),
        sha256="7a027cf3dae8605ed88f8fad9bceb331f80629752c0682c2c41fdf253978affb",
        num_classes=50_257,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/gpt2-medium",
            "license": _LICENSE,
            "num_params": 354_823_168,
            "file_size_mb": 1353.57,
        },
    )
    DEFAULT = WEBTEXT


@register_weights("gpt2_large_lm")
class GPT2LargeLMWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.gpt2_large_lm`.

    GPT-2 Large trunk + tied ``lm_head`` over the 50 257-token BPE vocab.

    Attributes
    ----------
    WEBTEXT : WeightEntry
        WebText pre-training checkpoint (trunk + LM head) sourced from
        ``transformers/gpt2-large``.
    DEFAULT : WeightEntry
        Alias for :attr:`WEBTEXT`.

    Notes
    -----
    Reference: Radford, Wu, Child, Luan, Amodei, Sutskever, *"Language
    Models are Unsupervised Multitask Learners"*, OpenAI 2019.

    Examples
    --------
    >>> from lucid.models import gpt2_large_lm
    >>> model = gpt2_large_lm(pretrained=True).eval()
    """

    WEBTEXT = WeightEntry(
        url=_url("gpt2-large-lm"),
        sha256="45e0d48011dcf8c6c82cf8046453bc4dc783f67814615f342d1f17f17906d892",
        num_classes=50_257,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/gpt2-large",
            "license": _LICENSE,
            "num_params": 774_030_080,
            "file_size_mb": 2952.73,
        },
    )
    DEFAULT = WEBTEXT


@register_weights("gpt2_xlarge_lm")
class GPT2XLargeLMWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.gpt2_xlarge_lm`.

    GPT-2 XLarge trunk + tied ``lm_head`` over the 50 257-token BPE vocab.

    Attributes
    ----------
    WEBTEXT : WeightEntry
        WebText pre-training checkpoint (trunk + LM head) sourced from
        ``transformers/gpt2-xl``.
    DEFAULT : WeightEntry
        Alias for :attr:`WEBTEXT`.

    Notes
    -----
    Reference: Radford, Wu, Child, Luan, Amodei, Sutskever, *"Language
    Models are Unsupervised Multitask Learners"*, OpenAI 2019.

    Examples
    --------
    >>> from lucid.models import gpt2_xlarge_lm
    >>> model = gpt2_xlarge_lm(pretrained=True).eval()
    """

    WEBTEXT = WeightEntry(
        url=_url("gpt2-xlarge-lm"),
        sha256="f1216dd4b93b3d005cf4c4fd07110c2cb8acc66931b61f35776b99735be6791b",
        num_classes=50_257,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/gpt2-xl",
            "license": _LICENSE,
            "num_params": 1_557_611_200,
            "file_size_mb": 5941.87,
        },
    )
    DEFAULT = WEBTEXT
