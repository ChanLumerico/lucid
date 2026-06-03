"""Pretrained-weight declarations for the GPT-1 family.

Two checkpoints ship through :mod:`lucid.weights`, converted from the OpenAI
GPT release (Radford et al. 2018) re-hosted on the Hub:

* :class:`GPTWeights` — the bare ``GPTModel`` decoder trunk.
* :class:`GPTLMWeights` — trunk + ``lm_head`` tied to ``tokens_embed``.

Conversion is an identity key map plus the Conv1D→Linear weight transpose
(``c_attn`` / ``c_proj`` / ``c_fc``); forward parity vs the reference
framework is 1.3e-5.  Pre-trained on BookCorpus (byte-pair-encoded,
40 478-token vocab); tokenize with
:class:`lucid.models.text.gpt.GPTTokenizer`.
"""

from lucid.utils.transforms import Compose
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

_NOOP = Compose([])
_TAG = "BOOKCORPUS"
_LICENSE = "mit"


@register_weights("gpt")
class GPTWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.gpt`.

    Radford et al. 2018 12-layer decoder trunk (``H=768, A=12``, 116.5 M params).

    Attributes
    ----------
    BOOKCORPUS : WeightEntry
        BookCorpus pre-training checkpoint sourced from
        ``transformers/openai-community/openai-gpt``.
    DEFAULT : WeightEntry
        Alias for :attr:`BOOKCORPUS`.

    Notes
    -----
    Reference: Radford, Narasimhan, Salimans, Sutskever, *"Improving
    Language Understanding by Generative Pre-Training"*, OpenAI 2018.

    Examples
    --------
    >>> from lucid.models import gpt
    >>> model = gpt(pretrained=True).eval()
    """

    BOOKCORPUS = WeightEntry(
        url=f"{HUB_BASE}/gpt/resolve/main/{_TAG}/model.safetensors",
        sha256="4172e94751f2a64076b09c41ea74966f9bc18465acea6ca4bcf3f43c3be661f7",
        num_classes=768,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/openai-community/openai-gpt",
            "license": _LICENSE,
            "num_params": 116_534_784,
            "file_size_mb": 444.56,
        },
    )
    DEFAULT = BOOKCORPUS


@register_weights("gpt_lm")
class GPTLMWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.gpt_lm`.

    GPT-1 decoder + tied ``lm_head`` over the 40 478-token BPE vocab.

    Attributes
    ----------
    BOOKCORPUS : WeightEntry
        BookCorpus pre-training checkpoint (decoder + LM head) sourced
        from ``transformers/openai-community/openai-gpt``.
    DEFAULT : WeightEntry
        Alias for :attr:`BOOKCORPUS`.

    Notes
    -----
    Reference: Radford, Narasimhan, Salimans, Sutskever, *"Improving
    Language Understanding by Generative Pre-Training"*, OpenAI 2018.

    Examples
    --------
    >>> from lucid.models import gpt_lm
    >>> model = gpt_lm(pretrained=True).eval()
    """

    BOOKCORPUS = WeightEntry(
        url=f"{HUB_BASE}/gpt-lm/resolve/main/{_TAG}/model.safetensors",
        sha256="5427d53997be3425ed560c2849ab4f5b9284f3563f5c56b24fbd6e69a785834b",
        num_classes=40_478,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/openai-community/openai-gpt",
            "license": _LICENSE,
            "num_params": 116_534_784,
            "file_size_mb": 563.15,
        },
    )
    DEFAULT = BOOKCORPUS
