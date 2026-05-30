r"""Pretrained-weight declarations for DDPM (Ho et al., 2020).

Full UNet checkpoints converted from the official ``google/ddpm-*`` diffusers
releases (Apache-2.0).  Unlike a classifier, ``pretrained=True`` here is an
*inference-ready generator*: the :class:`DDPMForImageGeneration` wrapper draws
samples via ancestral DDPM sampling, and the bare :class:`DDPMModel` is the
trained noise predictor :math:`\epsilon_\theta(x_t, t)`.

The generation input is Gaussian noise, so the entries carry a no-op
preprocessing transform; sampled outputs are in ``[-1, 1]`` (rescale to
``[0, 1]`` for display).
"""

from lucid.utils.transforms import Compose
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

_NOOP = Compose([])


def _url(slug: str, tag: str) -> str:
    return f"{HUB_BASE}/{slug}/resolve/main/{tag}/model.safetensors"


@register_weights("ddpm_cifar")
@register_weights("ddpm_cifar_gen")
class DDPMCifarWeights(WeightsEnum):
    r"""Pretrained DDPM weights for :func:`lucid.models.ddpm_cifar`."""

    CIFAR10 = WeightEntry(
        url=_url("ddpm-cifar10", "CIFAR10"),
        sha256="18973f01a0f99620fad9060d7db4d8d6255960132538dadc78aeb1d21078a084",
        num_classes=3,
        transforms=_NOOP,
        meta={
            "tag": "CIFAR10",
            "source": "diffusers/google/ddpm-cifar10-32",
            "license": "apache-2.0",
            "num_params": 35_746_307,
            "metrics": {"cifar10": {"fid": 3.17}},
        },
    )
    DEFAULT = CIFAR10


@register_weights("ddpm_lsun")
@register_weights("ddpm_lsun_gen")
class DDPMChurchWeights(WeightsEnum):
    r"""Pretrained DDPM weights for :func:`lucid.models.ddpm_lsun`.

    LSUN-Church 256×256 (``google/ddpm-church-256``).
    """

    LSUN_CHURCH = WeightEntry(
        url=_url("ddpm-church", "LSUN_CHURCH"),
        sha256="8e247117314a46e816c831573722affe02a92189a145928dac0693298d8ae723",
        num_classes=3,
        transforms=_NOOP,
        meta={
            "tag": "LSUN_CHURCH",
            "source": "diffusers/google/ddpm-church-256",
            "license": "apache-2.0",
            "num_params": 113_673_219,
            "metrics": {"lsun-churches": {"fid": 7.89}},
        },
    )
    DEFAULT = LSUN_CHURCH
