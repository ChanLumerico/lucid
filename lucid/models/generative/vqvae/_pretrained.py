"""Registry factories for VQ-VAE.

van den Oord et al., 2017 report a single image architecture rather than a
table of sized variants — 2 stride-2 convolutions, 2 residual blocks, 256
hidden units, a 512 x 256 codebook, beta = 0.25 — so this family follows
the project rule for paper-nominal models and exposes one trunk plus one
task head rather than ``_small`` / ``_base`` / ``_large`` siblings:

    * ``vqvae``     — bare encoder / codebook / decoder.
    * ``vqvae_gen`` — the same trunk with the three-term objective.

The paper's ImageNet setting is ``sample_size=128`` (a 32 x 32 latent
grid); the defaults here use the generative domain's 32 x 32 images (an
8 x 8 grid) and are rescaled by overriding config fields at
``create_model`` time.

No parameter count is registered: the paper states the architecture but
never a trainable-parameter total, and the docs site derives the true
count by introspection anyway.
"""

from dataclasses import replace
from typing import Any, cast

from lucid.models._registry import register_model
from lucid.models._utils._common import reject_unavailable_pretrained
from lucid.models.generative.vqvae._config import VQVAEConfig
from lucid.models.generative.vqvae._model import VQVAEForImageGeneration, VQVAEModel

# Paper defaults (Section 4.1) at the generative domain's 32 x 32 image
# size: two stride-2 stages, two residual blocks, 256 hidden units, a
# 512 x 256 codebook, and a commitment coefficient of 0.25.
_CFG_VQ_VAE = VQVAEConfig(
    sample_size=32,
    in_channels=3,
    out_channels=3,
    num_embeddings=512,
    embedding_dim=256,
    hidden_channels=256,
    num_downsample_layers=2,
    num_residual_layers=2,
    residual_hidden_channels=256,
    commitment_cost=0.25,
)


def _apply(cfg: VQVAEConfig, overrides: dict[str, object]) -> VQVAEConfig:
    return replace(cfg, **cast(dict[str, Any], overrides)) if overrides else cfg


# ── Bare encoder / codebook / decoder ────────────────────────────────────────


@register_model(
    task="base",
    family="vqvae",
    model_type="vqvae",
    model_class=VQVAEModel,
    default_config=_CFG_VQ_VAE,
)
def vqvae(pretrained: bool = False, **overrides: object) -> VQVAEModel:
    r"""Construct a VQ-VAE trunk — encoder, codebook, decoder, no loss.

    Discrete-latent auto-encoder following van den Oord, Vinyals, and
    Kavukcuoglu, 2017, Section 4.1: two stride-2 convolutions with a 4x4
    window, two pre-activation residual blocks, 256 hidden units
    throughout, and a ``512 x 256`` codebook.  At the default
    ``sample_size=32`` the latent grid is ``8 x 8``.

    Use :meth:`VQVAEModel.encode_indices` and
    :meth:`VQVAEModel.decode_indices` for the tokeniser interface — an
    image in, an integer code field out, and back.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises
        rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`VQVAEConfig` field overrides (e.g.
        ``sample_size=128`` for the paper's ImageNet setting,
        ``num_embeddings=...``, ``commitment_cost=...``) forwarded into
        the underlying config.

    Returns
    -------
    VQVAEModel
        VQ-VAE trunk configured with the paper defaults and any
        overrides.

    Notes
    -----
    Reference: van den Oord, Vinyals, and Kavukcuoglu, *"Neural Discrete
    Representation Learning"*, NeurIPS, 2017 (arXiv:1711.00937).

    Quantisation snaps each spatial position of the encoder output to its
    nearest codebook entry:

    .. math::

        z_q(x) = e_k,
        \qquad
        k = \arg\min_j \big\| z_e(x) - e_j \big\|_2 .

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.vqvae import vqvae
    >>> model = vqvae().eval()
    >>> x = lucid.randn((1, 3, 32, 32))
    >>> out = model(x)
    >>> out.sample.shape, out.indices.shape
    ((1, 3, 32, 32), (1, 8, 8))
    """
    if pretrained:
        reject_unavailable_pretrained("vqvae")
    return VQVAEModel(_apply(_CFG_VQ_VAE, overrides))


# ── Image-generation head ────────────────────────────────────────────────────


@register_model(
    task="image-generation",
    family="vqvae",
    model_type="vqvae",
    model_class=VQVAEForImageGeneration,
    default_config=_CFG_VQ_VAE,
)
def vqvae_gen(pretrained: bool = False, **overrides: object) -> VQVAEForImageGeneration:
    r"""Construct a VQ-VAE with the full training objective and a sampler.

    Same trunk as :func:`vqvae`, wrapped with the three-term objective of
    van den Oord et al., 2017 — reconstruction, codebook, and
    :math:`\beta`-weighted commitment — plus a convenience sampler over
    the uniform codebook prior.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises
        rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`VQVAEConfig` field overrides forwarded into the
        underlying config.  Pass ``recon_loss="bce"`` for Bernoulli
        likelihoods on ``[0, 1]`` data, or ``commitment_cost=...`` to
        retune :math:`\beta`.

    Returns
    -------
    VQVAEForImageGeneration
        VQ-VAE wrapped with the full objective and prior sampler.

    Notes
    -----
    Reference: van den Oord, Vinyals, and Kavukcuoglu, *"Neural Discrete
    Representation Learning"*, NeurIPS, 2017 (arXiv:1711.00937).

    Training objective:

    .. math::

        L = \log p(x \mid z_q(x))
            + \big\| \mathrm{sg}[z_e(x)] - e \big\|_2^2
            + \beta \big\| z_e(x) - \mathrm{sg}[e] \big\|_2^2 .

    ``generate`` samples the *uniform* prior the model was trained
    against, not a learned one — the paper fits a PixelCNN over the latent
    grid for its figures.  Samples from the uniform prior are expected to
    be incoherent; see :class:`VQVAEForImageGeneration`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.vqvae import vqvae_gen
    >>> model = vqvae_gen().eval()
    >>> x = lucid.randn((1, 3, 32, 32))
    >>> out = model(x)
    >>> out.sample.shape, out.perplexity.shape
    ((1, 3, 32, 32), ())
    """
    if pretrained:
        reject_unavailable_pretrained("vqvae_gen")
    return VQVAEForImageGeneration(_apply(_CFG_VQ_VAE, overrides))
