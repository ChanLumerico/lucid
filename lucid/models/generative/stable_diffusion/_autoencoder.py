r"""The first stage — a KL-regularised autoencoder over a *spatial* latent.

This is not the VAE this zoo already has.  :class:`lucid.models.VAEModel`
flattens to a vector through a ``Linear`` bottleneck, which is the right
shape for a generative model of whole images and the wrong one here: the
second stage is a convolutional U-Net, and it needs a latent that still
has height and width to convolve over.

So the bottleneck is a convolution, and the latent is
:math:`z \in \mathbb{R}^{h \times w \times c}` with
:math:`h = H/f`.  At the released settings :math:`f = 8` and
:math:`c = 4`, so a 512-pixel image becomes ``(4, 64, 64)`` — 48 times
fewer values, and every one of them still addressable by position.

**Why the KL penalty is "slight".**  The paper's phrasing is exact: it
imposes *a slight KL-penalty towards a standard normal*.  A full VAE
weight would pull the latent toward :math:`\mathcal{N}(0, I)` hard
enough to erase the spatial detail the U-Net is supposed to model.  The
penalty exists only to stop the scale drifting — the diffusion forward
process adds noise of a fixed variance, so a latent free to grow makes
that noise negligible and the model learns nothing.
"""

from dataclasses import dataclass
from typing import cast, final, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.models._output import ModelOutput
from lucid.models.generative.stable_diffusion._config import StableDiffusionConfig

__all__ = ["AutoencoderKL", "AutoencoderKLOutput", "DiagonalGaussian"]


@final
class DiagonalGaussian:
    r"""The posterior a spatial encoder emits, as a distribution.

    Parameters
    ----------
    mean, logvar : Tensor
        ``(B, C, H, W)`` each — the encoder splits its ``2C`` output
        channels down the middle.

    Notes
    -----
    Held as a small object rather than a pair of tensors because three
    call sites want three different things from it: training wants a
    sample and a KL, encoding for the U-Net wants a sample, and
    deterministic reconstruction wants the mean.  Returning a tuple
    makes the third indistinguishable from the second at the call site,
    which is how a "deterministic" path quietly keeps sampling.

    ``logvar`` is clamped.  An encoder early in training can emit a
    variance whose exponential overflows, and the resulting ``inf``
    reaches the loss as a ``nan`` several operations later.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.stable_diffusion import DiagonalGaussian
    >>> post = DiagonalGaussian(lucid.zeros((1, 4, 8, 8)), lucid.zeros((1, 4, 8, 8)))
    >>> post.mode().shape
    (1, 4, 8, 8)
    >>> float(post.kl().item())
    0.0
    """

    def __init__(self, mean: Tensor, logvar: Tensor) -> None:
        """Hold the posterior. See the class docstring for parameters."""
        self.mean = mean
        self.logvar = lucid.clip(logvar, -30.0, 20.0)
        self.std = lucid.exp(0.5 * self.logvar)

    def sample(self) -> Tensor:
        """Draw a reparameterised sample.

        Returns
        -------
        Tensor
            ``(B, C, H, W)``.
        """
        noise = lucid.randn(
            tuple(int(s) for s in self.mean.shape),
            device=self.mean.device.type,
            dtype=self.mean.dtype,
        )
        return self.mean + self.std * noise

    def mode(self) -> Tensor:
        """Return the distribution's mode, which for a Gaussian is its mean.

        Returns
        -------
        Tensor
            ``(B, C, H, W)``.
        """
        return self.mean

    def kl(self) -> Tensor:
        r"""KL against :math:`\mathcal{N}(0, I)`, averaged over the batch.

        Returns
        -------
        Tensor
            A scalar.

        Notes
        -----
        Summed over channels *and* both spatial axes, then averaged over
        the batch — the latent is one event per image, not one per
        position.  Averaging over positions instead would make the
        penalty depend on resolution, so the same weight would mean
        different things at 256 and 512 pixels.
        """
        per_element = self.mean**2 + lucid.exp(self.logvar) - 1.0 - self.logvar
        return 0.5 * per_element.sum(dim=(1, 2, 3)).mean()


@final
class _ResnetBlock(nn.Module):
    """Pre-norm residual block, the autoencoder's unit of depth.

    Parameters
    ----------
    in_channels, out_channels : int
        Widths.  A ``1 x 1`` shortcut appears only when they differ.
    groups : int
        GroupNorm groups.
    """

    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        """Initialise the block. See the class docstring for parameters."""
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.shortcut: nn.Module | None = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Two convolutions on the residual stream.

        Parameters
        ----------
        x : Tensor
            ``(B, in_channels, H, W)``.

        Returns
        -------
        Tensor
            ``(B, out_channels, H, W)``.
        """
        h = cast(Tensor, self.conv1(cast(Tensor, self.act(cast(Tensor, self.norm1(x))))))
        h = cast(Tensor, self.conv2(cast(Tensor, self.act(cast(Tensor, self.norm2(h))))))
        skip = x if self.shortcut is None else cast(Tensor, self.shortcut(x))
        return skip + h


@final
class _SelfAttention2d(nn.Module):
    """Single-head self-attention over a feature map's positions.

    Parameters
    ----------
    channels : int
        Feature width.
    groups : int
        GroupNorm groups.

    Notes
    -----
    Applied only at the lowest resolution, where the token count is
    small enough for the quadratic cost to be irrelevant.  This is the
    autoencoder's only non-local operation — everything else is a
    convolution, so without it the receptive field would never cover the
    frame.
    """

    def __init__(self, channels: int, groups: int) -> None:
        """Initialise the block. See the class docstring for parameters."""
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.attn = nn.MultiheadAttention(channels, 1, batch_first=True)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Attend across positions, then add back.

        Parameters
        ----------
        x : Tensor
            ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            ``(B, C, H, W)``.
        """
        b, c, h, w = (int(s) for s in x.shape)
        normed = cast(Tensor, self.norm(x)).reshape(b, c, h * w).swapaxes(1, 2)
        attended, _ = self.attn(normed, normed, normed, need_weights=False)
        attended = attended.swapaxes(1, 2).reshape(b, c, h, w)
        return x + cast(Tensor, self.proj(attended))


@dataclass(slots=True)
class AutoencoderKLOutput(ModelOutput):
    """What :class:`AutoencoderKL` returns.

    Attributes
    ----------
    reconstruction : Tensor
        ``(B, C, H, W)`` in the input's units.
    latent : Tensor
        ``(B, latent_channels, H/f, W/f)`` — whichever of the sample or
        the mode was asked for.
    kl : Tensor
        The posterior's divergence from a standard normal, a scalar.
    """

    reconstruction: Tensor
    latent: Tensor
    kl: Tensor


class AutoencoderKL(nn.Module):
    r"""The KL-regularised first stage of a latent diffusion model.

    Parameters
    ----------
    config : StableDiffusionConfig
        Read for the autoencoder fields.

    Notes
    -----
    Reference: Rombach et al., CVPR 2022 (arXiv:2112.10752), §3.1.

    The encoder ends in a ``2c``-channel convolution whose halves are
    the posterior's mean and log-variance, and a ``1 x 1``
    ``quant_conv`` after it; the decoder begins with the mirroring
    ``post_quant_conv``.  Those two look redundant and are not — they
    are where the released VQ- and KL-regularised first stages differ,
    so keeping them makes the two interchangeable behind one interface.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.stable_diffusion import (
    ...     AutoencoderKL, StableDiffusionConfig)
    >>> config = StableDiffusionConfig(sample_size=32, downsample_factor=4,
    ...                                vae_block_out_channels=(32, 64, 64),
    ...                                unet_block_out_channels=(32, 64),
    ...                                norm_num_groups=32)
    >>> vae = AutoencoderKL(config).eval()
    >>> out = vae(lucid.randn((1, 3, 32, 32)))
    >>> out.latent.shape, out.reconstruction.shape
    ((1, 4, 8, 8), (1, 3, 32, 32))
    """

    def __init__(self, config: StableDiffusionConfig) -> None:
        """Initialise the autoencoder. See the class docstring."""
        super().__init__()
        self.config = config
        widths = config.vae_block_out_channels
        groups = config.norm_num_groups
        latent = config.latent_channels

        self.conv_in = nn.Conv2d(
            config.in_channels, widths[0], kernel_size=3, padding=1
        )
        down: list[nn.Module] = []
        current = widths[0]
        for level, width in enumerate(widths):
            for _ in range(config.vae_layers_per_block):
                down.append(_ResnetBlock(current, width, groups))
                current = width
            if level != len(widths) - 1:
                down.append(
                    nn.Conv2d(current, current, kernel_size=3, stride=2, padding=1)
                )
        self.down = nn.ModuleList(down)

        self.mid_block_1 = _ResnetBlock(current, current, groups)
        self.mid_attn = _SelfAttention2d(current, groups)
        self.mid_block_2 = _ResnetBlock(current, current, groups)

        self.norm_out = nn.GroupNorm(groups, current)
        self.conv_out = nn.Conv2d(current, 2 * latent, kernel_size=3, padding=1)
        self.quant_conv = nn.Conv2d(2 * latent, 2 * latent, kernel_size=1)

        self.post_quant_conv = nn.Conv2d(latent, latent, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(latent, current, kernel_size=3, padding=1)
        self.decoder_mid_block_1 = _ResnetBlock(current, current, groups)
        self.decoder_mid_attn = _SelfAttention2d(current, groups)
        self.decoder_mid_block_2 = _ResnetBlock(current, current, groups)

        up: list[nn.Module] = []
        for level, width in enumerate(reversed(widths)):
            for _ in range(config.vae_layers_per_block + 1):
                up.append(_ResnetBlock(current, width, groups))
                current = width
            if level != len(widths) - 1:
                up.append(nn.Upsample(scale_factor=2.0, mode="nearest"))
                up.append(nn.Conv2d(current, current, kernel_size=3, padding=1))
        self.up = nn.ModuleList(up)

        self.decoder_norm_out = nn.GroupNorm(groups, current)
        self.decoder_conv_out = nn.Conv2d(
            current, config.out_channels, kernel_size=3, padding=1
        )
        self.act = nn.SiLU()

    def encode(self, x: Tensor) -> DiagonalGaussian:
        """Map an image to its posterior over latents.

        Parameters
        ----------
        x : Tensor
            ``(B, in_channels, H, W)``.

        Returns
        -------
        DiagonalGaussian
            Mean and log-variance, each ``(B, latent_channels, H/f, W/f)``.
        """
        h = cast(Tensor, self.conv_in(x))
        for block in self.down:
            h = cast(Tensor, block(h))
        h = cast(Tensor, self.mid_block_1(h))
        h = cast(Tensor, self.mid_attn(h))
        h = cast(Tensor, self.mid_block_2(h))
        h = cast(Tensor, self.conv_out(cast(Tensor, self.act(cast(Tensor, self.norm_out(h))))))
        moments = cast(Tensor, self.quant_conv(h))
        channels = int(moments.shape[1]) // 2
        return DiagonalGaussian(moments[:, :channels], moments[:, channels:])

    def decode(self, z: Tensor) -> Tensor:
        """Map a latent back to an image.

        Parameters
        ----------
        z : Tensor
            ``(B, latent_channels, h, w)``.

        Returns
        -------
        Tensor
            ``(B, out_channels, h*f, w*f)``.
        """
        h = cast(Tensor, self.decoder_conv_in(cast(Tensor, self.post_quant_conv(z))))
        h = cast(Tensor, self.decoder_mid_block_1(h))
        h = cast(Tensor, self.decoder_mid_attn(h))
        h = cast(Tensor, self.decoder_mid_block_2(h))
        for block in self.up:
            h = cast(Tensor, block(h))
        return cast(
            Tensor,
            self.decoder_conv_out(cast(Tensor, self.act(cast(Tensor, self.decoder_norm_out(h))))),
        )

    @override
    def forward(  # type: ignore[override]
        self, x: Tensor, sample: bool = True
    ) -> AutoencoderKLOutput:
        """Encode, take a latent, and decode it back.

        Parameters
        ----------
        x : Tensor
            ``(B, in_channels, H, W)``.
        sample : bool, default=True
            Draw from the posterior, or take its mode.  Training draws;
            a deterministic reconstruction does not.

        Returns
        -------
        AutoencoderKLOutput
            Reconstruction, latent and KL.
        """
        posterior = self.encode(x)
        z = posterior.sample() if sample else posterior.mode()
        return AutoencoderKLOutput(
            reconstruction=self.decode(z), latent=z, kl=posterior.kl()
        )
