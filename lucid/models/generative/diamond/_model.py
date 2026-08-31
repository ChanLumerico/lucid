r"""DIAMOND — a world model that predicts frames instead of latents.

Three networks, trained separately, sharing nothing.

The **denoiser** answers the only question a dynamics model has to
answer: given the last :math:`L` frames and the actions taken, what does
the next frame look like?  It is a plain U-Net 2D wrapped in EDM's
preconditioners, so the thing that varies with the noise level is the
*target*, not the architecture.

The **reward/termination model** answers what an image cannot: scalar
reward and whether the episode ended.  Both are sequence predictions
under partial observability, so it is convolutional blocks feeding an
LSTM rather than a second diffusion model.

The **actor-critic** is trained entirely on frames the denoiser
imagined, never on real experience.

Keeping them apart is the paper's design, not an implementation
convenience: the diffusion loss is a reconstruction in pixel space, the
reward loss is cross-entropy on three classes, and the policy loss is
REINFORCE.  Nothing would be shared by fusing them but a checkpoint.
"""

import math
from dataclasses import dataclass
from typing import Callable, ClassVar, cast, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._output import ModelOutput
from lucid.models._tasks import WorldModelingModel
from lucid.models.generative._common._returns import lambda_return
from lucid.models.generative.diamond._config import DIAMONDConfig

__all__ = [
    "DIAMONDModel",
    "DIAMONDForWorldModeling",
    "DIAMONDOutput",
    "DIAMONDBehaviorOutput",
]


def _groups(channels: int) -> int:
    """Group count for a normalisation over ``channels``.

    ``max(1, channels // 32)`` — 64 channels normalise in two groups, 32
    in one.  ⚠️ The paper says only "a group normalization layer"; this
    rule is read off the released implementation, and it is the sort of
    choice that changes activations without changing a single shape.
    """
    return max(1, channels // 32)


class _GroupNorm(nn.Module):
    """Plain group norm, wrapped so its parameter path matches the norms
    that *are* conditioned.

    Notes
    -----
    Both kinds appear in the released checkpoint and they are not
    interchangeable: the residual blocks' norms take their scale and
    shift from the conditioning vector, while the output norm and the
    attention's norm carry their own affine.
    """

    def __init__(self, channels: int) -> None:
        """Initialise the layer.

        Parameters
        ----------
        channels : int
            Channels being normalised.
        """
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Normalise ``(B, C, H, W)``."""
        return cast(Tensor, self.norm(x))


class _AdaGroupNorm(nn.Module):
    """Group norm whose scale and shift are regressed from a vector.

    Parameters
    ----------
    channels : int
        Channels being normalised.
    cond_dim : int
        Width of the conditioning vector.

    Notes
    -----
    Reference: Zheng et al., *"Learning Semantic-Aware Normalization for
    Generative Adversarial Networks"*, NeurIPS, 2020 — the mechanism
    DIAMOND admits both the action history and the diffusion time
    through.

    The normalisation itself is affine-free: an affine that the
    conditioning immediately overrides would be two parameters per
    channel doing nothing.
    """

    def __init__(self, channels: int, cond_dim: int) -> None:
        """Initialise the layer. See the class docstring for parameters."""
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels, affine=False)
        self.linear = nn.Linear(cond_dim, 2 * channels)

    @override
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:  # type: ignore[override]
        """Normalise ``(B, C, H, W)`` and modulate it by ``(B, cond_dim)``."""
        params = cast(Tensor, self.linear(cond))
        channels = int(x.shape[1])
        scale = params[:, :channels].reshape(-1, channels, 1, 1)
        shift = params[:, channels:].reshape(-1, channels, 1, 1)
        return cast(Tensor, self.norm(x)) * (1.0 + scale) + shift


class _SelfAttention2d(nn.Module):
    """Multi-head self-attention over a feature map.

    Parameters
    ----------
    channels : int
        Feature width.
    head_dim : int, default=8
        Channels per head.

    Notes
    -----
    ⚠️ The paper's configuration table lists no attention, and the
    released Atari config sets ``attn_depths: [0,0,0,0]`` — yet the
    checkpoint carries attention weights in the U-Net's middle blocks
    and in the reward encoder's last stage.  Those depths govern the
    *resolution* stages; the middle is separate and always attends.
    """

    def __init__(self, channels: int, head_dim: int = 8) -> None:
        """Initialise the layer. See the class docstring for parameters."""
        super().__init__()
        self.heads = max(1, channels // head_dim)
        self.norm = _GroupNorm(channels)
        self.qkv_proj = nn.Conv2d(channels, channels * 3, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Attend over the spatial positions of ``(B, C, H, W)``."""
        batch, channels = int(x.shape[0]), int(x.shape[1])
        height, width = int(x.shape[2]), int(x.shape[3])
        qkv = cast(Tensor, self.qkv_proj(cast(Tensor, self.norm(x))))
        qkv = qkv.reshape(batch, 3, self.heads, channels // self.heads, height * width)
        query = qkv[:, 0].permute(0, 1, 3, 2)
        key = qkv[:, 1].permute(0, 1, 3, 2)
        value = qkv[:, 2].permute(0, 1, 3, 2)
        scale = float(channels // self.heads) ** -0.5
        scores = lucid.softmax(query @ key.permute(0, 1, 3, 2) * scale, dim=-1)
        out = (scores @ value).permute(0, 1, 3, 2)
        out = out.reshape(batch, channels, height, width)
        return x + cast(Tensor, self.out_proj(out))


class _ResBlock(nn.Module):
    """Two conditioned norm-activation-convolution stacks, plus attention.

    Parameters
    ----------
    in_channels, out_channels : int
        Channel widths.  When they differ a 1x1 ``proj`` brings the skip
        along — which is every decoder block, since each concatenates an
        encoder skip onto its input.
    cond_dim : int
        Width of the conditioning vector.
    attention : bool, default=False
        Whether the block attends after its convolutions.
    head_dim : int, default=8
        Channels per attention head.

    Notes
    -----
    Reference: Alonso et al., arXiv:2405.12399, Appendix D — "a group
    normalization layer, a SiLU activation, and a 3x3 convolution with
    stride 1 and padding 1".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        attention: bool = False,
        head_dim: int = 8,
    ) -> None:
        """Initialise the block. See the class docstring for parameters."""
        super().__init__()
        self.proj: nn.Module | None = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.norm1 = _AdaGroupNorm(in_channels, cond_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.norm2 = _AdaGroupNorm(out_channels, cond_dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.attn: nn.Module | None = (
            _SelfAttention2d(out_channels, head_dim) if attention else None
        )
        self.act = nn.SiLU()

    @override
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:  # type: ignore[override]
        """Apply the block to ``(B, C, H, W)`` under ``cond``."""
        residual = x if self.proj is None else cast(Tensor, self.proj(x))
        h = cast(Tensor, self.norm1(x, cond))
        h = cast(Tensor, self.conv1(cast(Tensor, self.act(h))))
        h = cast(Tensor, self.norm2(h, cond))
        h = cast(Tensor, self.conv2(cast(Tensor, self.act(h))))
        out = residual + h
        return out if self.attn is None else cast(Tensor, self.attn(out))


class _ResBlocks(nn.Module):
    """A run of residual blocks that all see the same conditioning."""

    def __init__(self, blocks: list[_ResBlock]) -> None:
        """Initialise the run.

        Parameters
        ----------
        blocks : list of _ResBlock
            The blocks, in order.
        """
        super().__init__()
        self.resblocks = nn.ModuleList(list(blocks))

    def each(self) -> list[_ResBlock]:
        """The blocks, typed, for a caller that manages skips itself."""
        return [cast(_ResBlock, b) for b in self.resblocks]

    @override
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:  # type: ignore[override]
        """Run every block in order."""
        for block in self.each():
            x = cast(Tensor, block(x, cond))
        return x


class _Downsample(nn.Module):
    """Halve the resolution with a stride-2 convolution."""

    def __init__(self, channels: int) -> None:
        """Initialise the layer.

        Parameters
        ----------
        channels : int
            Width, unchanged by the operation.
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Halve ``(B, C, H, W)``."""
        return cast(Tensor, self.conv(x))


class _Upsample(nn.Module):
    """Double the resolution, then convolve.

    Notes
    -----
    Nearest-neighbour interpolation followed by a 3x3 convolution, which
    is the usual way to avoid the checkerboard a transposed convolution
    leaves.

    The target size is passed in rather than assumed.  A stride-2
    convolution rounds up, so 15 becomes 8 going down and doubling would
    give 16 coming back — one row too many.  Interpolating to the skip's
    own size is exact at any resolution, and is what lets the CS:GO model
    work at 30x56.
    """

    def __init__(self, channels: int) -> None:
        """Initialise the layer.

        Parameters
        ----------
        channels : int
            Width, unchanged by the operation.
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=1, padding=1)

    @override
    def forward(self, x: Tensor, size: tuple[int, int]) -> Tensor:  # type: ignore[override]
        """Resize ``(B, C, H, W)`` to ``size``, then convolve."""
        up = F.interpolate(x, size=size, mode="nearest")
        return cast(Tensor, self.conv(up))


class _FourierFeatures(nn.Module):
    r"""Random-frequency embedding of a scalar.

    Parameters
    ----------
    width : int
        Output width; half of it is cosines and half sines, so the
        learned frequency vector is ``width // 2`` wide.

    Notes
    -----
    ⚠️ The paper does not say how the diffusion time is embedded.  The
    checkpoint's ``noise_emb.weight`` is ``(1, cond/2)``, which is
    Fourier features rather than the fixed sinusoidal ladder a
    transformer would use — the frequencies are **learned**.
    """

    def __init__(self, width: int) -> None:
        """Initialise the layer. See the class docstring for parameters."""
        super().__init__()
        self.weight = nn.Parameter(lucid.randn((1, width // 2)))

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Embed ``(B,)`` into ``(B, width)``."""
        args = x.reshape(-1, 1) * self.weight * (2.0 * math.pi)
        return lucid.cat([lucid.cos(args), lucid.sin(args)], dim=-1)


class _UNet(nn.Module):
    """The U-Net the denoiser's vector field runs on.

    Parameters
    ----------
    cond_dim : int
        Conditioning width.
    channels, layers : tuple of int
        Per-resolution widths and block counts.
    head_dim : int
        Channels per attention head in the middle blocks.

    Notes
    -----
    The decoder has one more block per resolution than the encoder,
    because the encoder pushes a skip for its input as well as for each
    of its blocks.  Getting that wrong builds a working network with
    visibly fewer parameters.
    """

    def __init__(
        self,
        cond_dim: int,
        channels: tuple[int, ...],
        layers: tuple[int, ...],
        head_dim: int,
        attn: tuple[int, ...] | None = None,
    ) -> None:
        """Initialise the network. See the class docstring for parameters."""
        super().__init__()
        attn = (0,) * len(channels) if attn is None else attn
        self.d_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        width = channels[0]
        for stage, (out_ch, count) in enumerate(zip(channels, layers)):
            self.downsamples.append(nn.Identity() if stage == 0 else _Downsample(width))
            self.d_blocks.append(
                _ResBlocks(
                    [
                        _ResBlock(
                            width if i == 0 else out_ch,
                            out_ch,
                            cond_dim,
                            bool(attn[stage]),
                            head_dim,
                        )
                        for i in range(count)
                    ]
                )
            )
            width = out_ch

        self.mid_blocks = _ResBlocks(
            [_ResBlock(width, width, cond_dim, True, head_dim) for _ in range(2)]
        )

        # The decoder concatenates whatever the encoder pushed, and at a
        # resolution boundary that is the *previous* stage's width rather
        # than this one's.  Recording the pushes is the only way to size
        # the blocks when the widths differ per stage, as CS:GO's do.
        pushes: list[int] = [channels[0]]
        width = channels[0]
        for stage, (out_ch, count) in enumerate(zip(channels, layers)):
            if stage > 0:
                pushes.append(width)
            pushes.extend([out_ch] * count)
            width = out_ch

        self.u_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        cursor = len(pushes)
        for stage in reversed(range(len(channels))):
            out_ch = channels[stage]
            self.upsamples.append(
                nn.Identity() if stage == len(channels) - 1 else _Upsample(width)
            )
            # The last block of each decoder resolution steps down to the
            # next one's width, so the upsample that follows already works
            # at the shallower size.  Holding the width for the whole
            # stage instead builds a wider network — 43M wider, at CS:GO's
            # channel counts — that still runs and still trains.
            shallower = channels[max(stage - 1, 0)]
            blocks = []
            count = layers[stage] + 1
            for index in range(count):
                cursor -= 1
                target = shallower if index == count - 1 else out_ch
                blocks.append(
                    _ResBlock(
                        width + pushes[cursor],
                        target,
                        cond_dim,
                        bool(attn[stage]),
                        head_dim,
                    )
                )
                width = target
            self.u_blocks.append(_ResBlocks(blocks))

    @override
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:  # type: ignore[override]
        """Map ``(B, C, H, W)`` through the U and back.

        The skips are one flat stack, not one list per resolution: the
        encoder pushes its input, then every block's output and every
        downsample's, and the decoder pops exactly one per block.  The
        counts have to agree — twelve either way at the paper's depths —
        and they only do because each decoder resolution has one block
        more than its encoder counterpart.
        """
        skips: list[Tensor] = [x]
        for stage, blocks in enumerate(self.d_blocks):
            if stage > 0:
                x = cast(Tensor, self.downsamples[stage](x))
                skips.append(x)
            for block in cast(_ResBlocks, blocks).each():
                x = cast(Tensor, block(x, cond))
                skips.append(x)

        x = cast(Tensor, self.mid_blocks(x, cond))

        for index, blocks in enumerate(self.u_blocks):
            if index > 0:
                target = skips[-1]
                resize = cast(_UpsampleCall, self.upsamples[index])
                x = resize(x, (int(target.shape[2]), int(target.shape[3])))
            for block in cast(_ResBlocks, blocks).each():
                x = lucid.cat([x, skips.pop()], dim=1)
                x = cast(Tensor, block(x, cond))
        return x


class _Denoiser(nn.Module):
    r"""The network EDM's preconditioners wrap.

    Parameters
    ----------
    config : DIAMONDConfig
        The variant to build.

    Notes
    -----
    Reference: Alonso et al., arXiv:2405.12399, Section 3.1 and Table 2,
    with the conditioning path taken from the released checkpoint — the
    paper says only that actions and diffusion time enter through
    adaptive group normalisation, not how they are embedded first.

    The past frames arrive already concatenated to the noised frame, so
    the image path is an ordinary U-Net that has no idea it is
    conditioned on anything.
    """

    def __init__(self, config: DIAMONDConfig) -> None:
        """Initialise the network. See the class docstring for parameters."""
        super().__init__()
        self.config = config
        cond_dim = config.cond_dim
        self.noise_emb = _FourierFeatures(cond_dim)
        # When the history is noised too, its level is a second thing the
        # network has to be told — the same frame at two different
        # degradations is two different conditionings.  Absent from the
        # Atari checkpoint, present in CS:GO's, exactly tracking
        # ``noise_previous_obs``.
        self.noise_cond_emb: nn.Module | None = (
            _FourierFeatures(cond_dim) if config.noise_previous_obs else None
        )
        # One embedding per conditioning step, flattened back to
        # ``cond_dim`` — so the width per action falls as the history
        # grows, rather than the conditioning vector growing with it.
        self.action_embed = nn.Embedding(
            config.num_actions, cond_dim // config.conditioning_frames
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )
        self.conv_in = nn.Conv2d(
            config.denoiser_in_channels, config.unet_channels[0], 3, padding=1
        )
        self.unet = _UNet(
            cond_dim,
            config.unet_channels,
            config.unet_layers,
            config.attention_head_dim,
            config.attn_depths,
        )
        self.norm_out = _GroupNorm(config.unet_channels[0])
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            config.unet_channels[0], config.out_channels, 3, padding=1
        )

    def conditioning(
        self, c_noise: Tensor, actions: Tensor, cond_noise: Tensor | None = None
    ) -> Tensor:
        """Fuse the noise level and the action history into one vector.

        Parameters
        ----------
        c_noise : Tensor
            ``(B,)`` preconditioned noise level of the frame being denoised.
        actions : Tensor
            ``(B, L)`` past action indices.
        cond_noise : Tensor or None, optional
            ``(B,)`` level the *history* was noised at, when it was.
            Ignored unless the configuration asks for it.

        Returns
        -------
        Tensor
            ``(B, cond_dim)``.
        """
        time = cast(Tensor, self.noise_emb(c_noise))
        embedded = cast(Tensor, self.action_embed(actions))
        flat = embedded.reshape(int(actions.shape[0]), -1)
        total = time + flat
        if self.noise_cond_emb is not None:
            level = lucid.zeros_like(c_noise) if cond_noise is None else cond_noise
            total = total + cast(Tensor, self.noise_cond_emb(level))
        return cast(Tensor, self.cond_proj(total))

    @override
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:  # type: ignore[override]
        """Map ``(B, (L+1)*C, H, W)`` to a single ``(B, C, H, W)`` frame."""
        h = cast(Tensor, self.conv_in(x))
        h = cast(Tensor, self.unet(h, cond))
        h = cast(Tensor, self.norm_out(h))
        return cast(Tensor, self.conv_out(cast(Tensor, self.act(h))))


class _Upsampler(nn.Module):
    r"""A second diffusion model that magnifies the denoiser's output.

    Parameters
    ----------
    config : DIAMONDConfig
        The variant to build; its ``upsampler_*`` fields describe this
        network and its ``upsampling_factor`` how far it magnifies.

    Notes
    -----
    ⚠️ Nowhere in the paper.  It exists only in the released CS:GO
    configuration, and it is why that experiment can afford a 3D scene:
    the world model diffuses at 30x56 and this brings the frame to
    150x280, rather than paying full-resolution diffusion for detail a
    cheaper network can add.

    It conditions on three images — the noised full-resolution frame, the
    low-resolution one scaled up to meet it, and the previous
    full-resolution frame — and on nothing else.  There is no action
    embedding: what the agent did is already in the frame it is being
    asked to sharpen.
    """

    def __init__(self, config: DIAMONDConfig) -> None:
        """Initialise the network. See the class docstring for parameters."""
        super().__init__()
        if config.upsampler_channels is None or config.upsampler_layers is None:
            raise ValueError("this configuration describes no upsampler")
        self.config = config
        cond_dim = config.cond_dim
        self.noise_emb = _FourierFeatures(cond_dim)
        self.noise_cond_emb = _FourierFeatures(cond_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )
        self.conv_in = nn.Conv2d(
            config.in_channels * 3, config.upsampler_channels[0], 3, padding=1
        )
        self.unet = _UNet(
            cond_dim,
            config.upsampler_channels,
            config.upsampler_layers,
            config.attention_head_dim,
            config.upsampler_attn_depths,
        )
        self.norm_out = _GroupNorm(config.upsampler_channels[0])
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            config.upsampler_channels[0], config.out_channels, 3, padding=1
        )

    def conditioning(self, c_noise: Tensor, cond_noise: Tensor | None = None) -> Tensor:
        """The two noise levels, embedded and summed.

        Parameters
        ----------
        c_noise : Tensor
            ``(B,)`` level of the frame being sharpened.
        cond_noise : Tensor or None, optional
            ``(B,)`` level its conditioning was noised at.

        Returns
        -------
        Tensor
            ``(B, cond_dim)``.
        """
        total = cast(Tensor, self.noise_emb(c_noise))
        level = lucid.zeros_like(c_noise) if cond_noise is None else cond_noise
        total = total + cast(Tensor, self.noise_cond_emb(level))
        return cast(Tensor, self.cond_proj(total))

    @override
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:  # type: ignore[override]
        """Map ``(B, 3*C, H, W)`` to one sharpened ``(B, C, H, W)`` frame."""
        h = cast(Tensor, self.conv_in(x))
        h = cast(Tensor, self.unet(h, cond))
        h = cast(Tensor, self.norm_out(h))
        return cast(Tensor, self.conv_out(cast(Tensor, self.act(h))))


class _Encoder(nn.Module):
    """Downsampling residual stack, with attention at the bottom.

    Parameters
    ----------
    in_channels : int
        Channels entering the stack.
    channels, layers : tuple of int
        Per-resolution widths and block counts.
    cond_dim : int
        Conditioning width.
    head_dim : int
        Channels per attention head in the final stage.

    Notes
    -----
    One stage more than ``channels`` names: the extra one sits at the
    bottom resolution and attends, mirroring the U-Net's middle blocks.
    Without it the reward model's LSTM would see a feature map from a
    stack that never looked across the frame.
    """

    def __init__(
        self,
        in_channels: int,
        channels: tuple[int, ...],
        layers: tuple[int, ...],
        cond_dim: int,
        head_dim: int,
    ) -> None:
        """Initialise the stack. See the class docstring for parameters."""
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        width = channels[0]
        for stage, (out_ch, count) in enumerate(zip(channels, layers)):
            self.downsamples.append(nn.Identity() if stage == 0 else _Downsample(width))
            self.blocks.append(
                _ResBlocks(
                    [
                        _ResBlock(width if i == 0 else out_ch, out_ch, cond_dim)
                        for i in range(count)
                    ]
                )
            )
            width = out_ch
        self.blocks.append(
            _ResBlocks(
                [
                    _ResBlock(width, width, cond_dim, True, head_dim)
                    for _ in range(layers[-1])
                ]
            )
        )
        self.out_channels = width

    @override
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:  # type: ignore[override]
        """Reduce ``(B, C, H, W)`` by a factor of two per downsample."""
        h = cast(Tensor, self.conv_in(x))
        for stage, blocks in enumerate(self.blocks):
            if 0 < stage < len(self.downsamples):
                h = cast(Tensor, self.downsamples[stage](h))
            h = cast(Tensor, cast(_ResBlocks, blocks)(h, cond))
        return h


class _RewardEndModel(nn.Module):
    r"""Reward and episode termination, over a sequence.

    Parameters
    ----------
    config : DIAMONDConfig
        The variant to build.

    Notes
    -----
    Reference: Alonso et al., arXiv:2405.12399, Appendix D and
    Algorithm 1, with three details from the released checkpoint that
    the paper does not state.

    It reads **two** frames, not one — the observation and the one the
    action led to.  A reward is a property of the transition, and a
    single frame cannot show one.

    Its head emits **five** numbers: three reward classes and two
    termination classes.  Algorithm 1's ``CE(r_hat, sign(r))`` explains
    the first three, since the environment clips reward to
    :math:`\{-1, 0, 1\}`; the checkpoint explains the rest.
    """

    def __init__(self, config: DIAMONDConfig) -> None:
        """Initialise the model. See the class docstring for parameters."""
        super().__init__()
        self.config = config
        cond_dim = config.reward_cond_dim
        self.action_embed = nn.Embedding(config.num_actions, cond_dim)
        self.encoder = _Encoder(
            config.in_channels * config.reward_frames,
            config.reward_channels,
            config.reward_layers,
            cond_dim,
            config.attention_head_dim,
        )
        side = _side(config) // (2 ** (len(config.reward_channels) - 1))
        flat = self.encoder.out_channels * side * side
        self.cell = nn.LSTMCell(flat, config.reward_lstm_dim)
        self.head = nn.Sequential(
            nn.Linear(config.reward_lstm_dim, config.reward_lstm_dim),
            nn.SiLU(),
            # No bias on the last layer — the released checkpoint has
            # none, and a bias here would be five parameters the port
            # could not fill.
            nn.Linear(config.reward_lstm_dim, 5, bias=False),
        )

    @override
    def forward(  # type: ignore[override]
        self,
        frames: Tensor,
        action: Tensor,
        state: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        """Advance one step.

        Parameters
        ----------
        frames : Tensor
            ``(B, reward_frames * C, H, W)`` — the transition, stacked.
        action : Tensor
            ``(B,)`` action indices taken from the first of them.
        state : tuple of Tensor or None, optional
            Carried LSTM ``(hidden, cell)``; ``None`` starts at zero,
            which is what the burn-in then overwrites.

        Returns
        -------
        (Tensor, Tensor, tuple of Tensor)
            Reward class logits ``(B, 3)``, termination class logits
            ``(B, 2)``, and the LSTM state to carry forward.
        """
        batch = int(frames.shape[0])
        cond = cast(Tensor, self.action_embed(action))
        feature = cast(Tensor, self.encoder(frames, cond))
        flat = feature.reshape(batch, -1)
        if state is None:
            zeros = lucid.zeros(
                (batch, self.config.reward_lstm_dim), device=frames.device.type
            )
            state = (zeros, zeros)
        step = cast(
            Callable[[Tensor, tuple[Tensor, Tensor]], tuple[Tensor, Tensor]], self.cell
        )
        hidden, cell = step(flat, state)
        out = cast(Tensor, self.head(hidden))
        return out[:, :3], out[:, 3:], (hidden, cell)


class _SimpleResBlock(nn.Module):
    """Norm, activation, one convolution, plus a projected skip.

    Notes
    -----
    The actor-critic's block is *not* the denoiser's: one convolution
    rather than two, an ordinary affine group norm rather than a
    conditioned one, and a 1x1 ``skip_projection`` when the width
    changes.  It has nothing to condition on — the policy sees a frame
    and nothing else.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialise the block.

        Parameters
        ----------
        in_channels, out_channels : int
            Channel widths.
        """
        super().__init__()
        self.f = nn.Sequential(
            _GroupNorm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
        )
        self.skip_projection: nn.Module | None = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Apply the block to ``(B, C, H, W)``."""
        residual = (
            x if self.skip_projection is None else cast(Tensor, self.skip_projection(x))
        )
        return residual + cast(Tensor, self.f(x))


class _ActorCritic(nn.Module):
    r"""Shared trunk, a policy head and a state-value head.

    Parameters
    ----------
    config : DIAMONDConfig
        The variant to build.

    Notes
    -----
    Reference: Alonso et al., arXiv:2405.12399, Appendix D — the weights
    "are shared except for the last layer".  :math:`V` is a state-value
    network rather than a critic proper, which is why the policy loss is
    REINFORCE with a baseline and not an advantage actor-critic.
    """

    def __init__(self, config: DIAMONDConfig) -> None:
        """Initialise the network. See the class docstring for parameters."""
        super().__init__()
        self.config = config
        layers: list[nn.Module] = [
            nn.Conv2d(config.in_channels, config.actor_channels[0], 3, padding=1)
        ]
        width = config.actor_channels[0]
        for out_ch, count in zip(config.actor_channels, config.actor_layers):
            for _ in range(count):
                layers.append(_SimpleResBlock(width, out_ch))
                width = out_ch
            layers.append(nn.MaxPool2d(2, stride=2))
        self.encoder = nn.Sequential(*layers)
        side = _side(config) // (2 ** len(config.actor_channels))
        self.cell = nn.LSTMCell(width * side * side, config.actor_lstm_dim)
        self.actor_linear = nn.Linear(config.actor_lstm_dim, config.num_actions)
        self.critic_linear = nn.Linear(config.actor_lstm_dim, 1)

    @override
    def forward(  # type: ignore[override]
        self, frame: Tensor, state: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        """Advance one step.

        Parameters
        ----------
        frame : Tensor
            ``(B, C, H, W)``.
        state : tuple of Tensor or None, optional
            Carried LSTM ``(hidden, cell)``.

        Returns
        -------
        (Tensor, Tensor, tuple of Tensor)
            Action logits ``(B, num_actions)``, state value ``(B,)``, and
            the LSTM state.
        """
        batch = int(frame.shape[0])
        feature = cast(Tensor, self.encoder(frame))
        flat = feature.reshape(batch, -1)
        if state is None:
            zeros = lucid.zeros(
                (batch, self.config.actor_lstm_dim), device=frame.device.type
            )
            state = (zeros, zeros)
        step = cast(
            Callable[[Tensor, tuple[Tensor, Tensor]], tuple[Tensor, Tensor]], self.cell
        )
        hidden, cell = step(flat, state)
        logits = cast(Tensor, self.actor_linear(hidden))
        value = cast(Tensor, self.critic_linear(hidden)).reshape(-1)
        return logits, value, (hidden, cell)


# ``Module.__call__`` is annotated for the common case of tensors in and a
# tensor out, which neither of the recurrent sub-models fits: they take a
# carried LSTM state and return one alongside their predictions.  Naming
# the two call shapes once keeps the casts off every call site.
_RewardEndCall = Callable[
    [Tensor, Tensor, tuple[Tensor, Tensor] | None],
    tuple[Tensor, Tensor, tuple[Tensor, Tensor]],
]
_ActorCriticCall = Callable[
    [Tensor, tuple[Tensor, Tensor] | None],
    tuple[Tensor, Tensor, tuple[Tensor, Tensor]],
]
# The decoder's upsample takes a target size rather than a second tensor,
# for the same reason: a stride-2 encoder does not halve cleanly.
_UpsampleCall = Callable[[Tensor, tuple[int, int]], Tensor]


def _side(config: DIAMONDConfig) -> int:
    """Frame side length, whichever way ``sample_size`` was written."""
    return (
        config.sample_size
        if isinstance(config.sample_size, int)
        else config.sample_size[0]
    )


@dataclass(slots=True)
class DIAMONDOutput(ModelOutput):
    """What the world model returns after denoising one step.

    Attributes
    ----------
    loss : Tensor
        The reconstruction loss of Algorithm 1, ``||D(x_noised) - x||^2``.
        Scalar.
    prediction : Tensor
        The denoised next frame, ``(B, C, H, W)``.
    sigma : Tensor
        The noise level each element was trained at, ``(B,)``.  Carried
        because the loss is an average over a *distribution* of noise
        levels, and a run that only ever drew easy ones would report a
        falling loss while learning nothing about the hard regime.
    """

    loss: Tensor
    prediction: Tensor
    sigma: Tensor


@dataclass(slots=True)
class DIAMONDBehaviorOutput(ModelOutput):
    """What a pass of imagination returns.

    Attributes
    ----------
    policy_loss, value_loss : Tensor
        REINFORCE with a baseline, and the squared error against
        :math:`\\lambda`-returns.  Scalars.
    entropy : Tensor
        Mean policy entropy over the imagined trajectory, before the
        weight :math:`\\eta` is applied.  Scalar.
    returns : Tensor
        The :math:`\\lambda`-returns the value head regressed to,
        ``(B, H)``.
    frames : Tensor
        Every frame the denoiser imagined, ``(B, H, C, H_img, W_img)``.
    history : Tensor
        The conditioning the rollout ended on, ``(B, L, C, H_img, W_img)``
        — what a continuation would start from.  Once the horizon
        reaches :math:`L` this is entirely imagined, which is the
        clearest statement of what "training in imagination" means.
    history_actions : Tensor
        The actions beside it, ``(B, L)``.
    """

    policy_loss: Tensor
    value_loss: Tensor
    entropy: Tensor
    returns: Tensor
    frames: Tensor
    history: Tensor
    history_actions: Tensor


class DIAMONDModel(PretrainedModel):
    r"""The three networks that make up a DIAMOND agent.

    Parameters
    ----------
    config : DIAMONDConfig
        Frozen configuration.

    Attributes
    ----------
    denoiser : _Denoiser
        The U-Net whose preconditioned wrapper predicts the next frame.
    reward_end : _RewardEndModel
        Reward class and termination, over a sequence.
    actor_critic : _ActorCritic
        The policy and its value baseline.

    Notes
    -----
    Reference: Alonso, Eloi, et al., *"Diffusion for World Modeling:
    Visual Details Matter in Atari"*, NeurIPS, 2024 (arXiv:2405.12399).

    The EDM preconditioners live on this class rather than inside the
    U-Net, because they are a property of the *diffusion*, not of the
    network: the same U-Net under DDPM's parameterisation is the
    comparison the paper's Section 5.1 runs, and it drifts.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.diamond import DIAMONDConfig, DIAMONDModel
    >>> config = DIAMONDConfig(
    ...     sample_size=16, unet_channels=(8, 8), unet_layers=(1, 1),
    ...     reward_channels=(8, 8), reward_layers=(1, 1),
    ...     actor_channels=(8, 8), actor_layers=(1, 1),
    ...     cond_dim=16, reward_cond_dim=8,
    ...     reward_lstm_dim=16, actor_lstm_dim=16, num_actions=4)
    >>> model = DIAMONDModel(config).eval()
    >>> frames = lucid.randn((2, 4, 3, 16, 16))
    >>> actions = lucid.tensor([[0, 1, 2, 3], [1, 1, 0, 2]], dtype=lucid.int64)
    >>> with lucid.no_grad():
    ...     nxt = model.imagine_frame(frames, actions)
    >>> nxt.shape
    (2, 3, 16, 16)
    """

    config_class: ClassVar[type[DIAMONDConfig]] = DIAMONDConfig

    def __init__(self, config: DIAMONDConfig) -> None:
        """Initialise the agent. See the class docstring for parameters."""
        super().__init__(config)
        self.config: DIAMONDConfig = config
        self.denoiser = _Denoiser(config)
        # The CS:GO configuration sets both of these to null: that
        # experiment trains a world model on static data, with no
        # reinforcement learning to give an agent anything to learn from.
        self.reward_end = _RewardEndModel(config) if config.with_agent else None
        self.actor_critic = _ActorCritic(config) if config.with_agent else None
        self.upsampler = (
            _Upsampler(config) if config.upsampler_channels is not None else None
        )

    def step_reward_end(
        self,
        frame: Tensor,
        next_frame: Tensor,
        action: Tensor,
        state: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        """One step of the reward/termination model, typed.

        Parameters
        ----------
        frame, next_frame : Tensor
            ``(B, C, H, W)`` each — the transition.  Both are needed
            because a reward is a property of the transition, and the
            released encoder reads them stacked.
        action : Tensor
            ``(B,)`` action indices taken from ``frame``.
        state : tuple of Tensor or None, optional
            Carried LSTM state.

        Returns
        -------
        (Tensor, Tensor, tuple of Tensor)
            Reward class logits ``(B, 3)``, termination class logits
            ``(B, 2)``, and the new state.
        """
        pair = lucid.cat([frame, next_frame], dim=1)
        return cast(_RewardEndCall, self.reward_end)(pair, action, state)

    def step_actor_critic(
        self, frame: Tensor, state: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        """One step of the actor-critic, typed.

        Parameters
        ----------
        frame : Tensor
            ``(B, C, H, W)``.
        state : tuple of Tensor or None, optional
            Carried LSTM state.

        Returns
        -------
        (Tensor, Tensor, tuple of Tensor)
            Action logits, state value, and the new state.
        """
        return cast(_ActorCriticCall, self.actor_critic)(frame, state)

    def preconditioners(self, sigma: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""EDM's four scalings at a noise level.

        Parameters
        ----------
        sigma : Tensor
            ``(B,)`` noise levels.

        Returns
        -------
        (Tensor, Tensor, Tensor, Tensor)
            ``c_in``, ``c_out``, ``c_skip`` — each ``(B, 1, 1, 1)`` so
            they broadcast over a frame — and ``c_noise`` as ``(B,)``.

        Notes
        -----
        Reference: Alonso et al., arXiv:2405.12399, Appendix C,
        equations 9-12, with :math:`\sigma_{\text{data}} = 0.5`.
        """
        data = self.config.sigma_data
        total = (sigma**2 + data**2).sqrt()
        c_in = (1.0 / total).reshape(-1, 1, 1, 1)
        c_out = (sigma * data / total).reshape(-1, 1, 1, 1)
        c_skip = (data**2 / (data**2 + sigma**2)).reshape(-1, 1, 1, 1)
        c_noise = lucid.log(sigma) * 0.25
        return c_in, c_out, c_skip, c_noise

    def denoise(
        self, noised: Tensor, sigma: Tensor, frames: Tensor, actions: Tensor
    ) -> Tensor:
        r"""Apply :math:`D_\theta` — the preconditioned denoiser.

        Parameters
        ----------
        noised : Tensor
            ``(B, C, H, W)``, the next frame with noise added.
        sigma : Tensor
            ``(B,)`` noise levels.
        frames : Tensor
            ``(B, L, C, H, W)`` clean history.
        actions : Tensor
            ``(B, L)`` action indices.

        Returns
        -------
        Tensor
            ``(B, C, H, W)`` estimate of the clean next frame.
        """
        c_in, c_out, c_skip, c_noise = self.preconditioners(sigma)
        history = frames.reshape(int(frames.shape[0]), -1, *frames.shape[3:])
        stacked = lucid.cat([noised * c_in, history], dim=1)
        cond = self.denoiser.conditioning(c_noise, actions)
        return c_skip * noised + c_out * cast(Tensor, self.denoiser(stacked, cond))

    def sigma_schedule(self, steps: int, device: str) -> Tensor:
        r"""The noise levels an Euler sampler walks down.

        Parameters
        ----------
        steps : int
            Number of denoising steps.
        device : str
            Where to build the schedule.

        Returns
        -------
        Tensor
            ``(steps + 1,)`` descending to exactly zero.

        Notes
        -----
        Karras et al.'s :math:`\rho`-schedule, which Algorithm 1 refers
        to as "the default identity schedule from EDM".  ⚠️ DIAMOND's
        tables give the sampler and the step count but **not**
        :math:`\sigma_{\min}`, :math:`\sigma_{\max}` or :math:`\rho`, so
        those come from EDM itself and are the one place here where a
        number is inherited rather than cited.
        """
        rho = 7.0
        sigma_min, sigma_max = 2e-3, 5.0
        ramp = lucid.arange(steps, dtype=lucid.float32, device=device) / max(
            steps - 1, 1
        )
        inv = sigma_max ** (1.0 / rho) + ramp * (
            sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho)
        )
        return lucid.cat([inv**rho, lucid.zeros((1,), device=device)], dim=0)

    def imagine_frame(
        self,
        frames: Tensor,
        actions: Tensor,
        *,
        steps: int | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        r"""Sample the next frame with Euler's method.

        Parameters
        ----------
        frames : Tensor
            ``(B, L, C, H, W)`` clean history.
        actions : Tensor
            ``(B, L)`` action indices.
        steps : int or None, optional
            Denoising steps; defaults to the configured 3.
        noise : Tensor or None, optional
            Starting sample, drawn when absent.

        Returns
        -------
        Tensor
            ``(B, C, H, W)``.

        Notes
        -----
        Each step moves along :math:`\mathrm{d}x/\mathrm{d}\sigma =
        (x - D_\theta(x, \sigma)) / \sigma`, which is the probability-flow
        ODE written in EDM's variables.
        """
        steps = self.config.denoise_steps if steps is None else steps
        if steps < 1:
            raise ValueError(f"steps must be positive, got {steps}")
        batch = int(frames.shape[0])
        device = frames.device.type
        schedule = self.sigma_schedule(steps, device)

        first = float(schedule[0].item())
        x = (
            lucid.randn(
                (batch, self.config.in_channels, *frames.shape[3:]), device=device
            )
            * first
            if noise is None
            else noise * first
        )
        for index in range(steps):
            sigma = schedule[index]
            level = sigma + lucid.zeros((batch,), device=device)
            denoised = self.denoise(x, level, frames, actions)
            derivative = (x - denoised) / sigma
            x = x + derivative * (schedule[index + 1] - sigma)
        return x

    @override
    def forward(  # type: ignore[override]
        self,
        frames: Tensor,
        actions: Tensor,
        next_frame: Tensor,
        *,
        sigma: Tensor | None = None,
    ) -> DIAMONDOutput:
        r"""Train the denoiser on one transition.

        Parameters
        ----------
        frames : Tensor
            ``(B, L, C, H, W)`` clean history.
        actions : Tensor
            ``(B, L)`` action indices.
        next_frame : Tensor
            ``(B, C, H, W)`` the frame to predict.
        sigma : Tensor or None, optional, keyword-only
            ``(B,)`` noise levels.  Drawn from the paper's log-normal
            when absent; supplied by tests that need a fixed level.

        Returns
        -------
        DIAMONDOutput
            Loss, denoised frame, and the noise levels used.

        Notes
        -----
        Reference: Alonso et al., arXiv:2405.12399, Algorithm 1 — sample
        :math:`\log \sigma \sim \mathcal{N}(P_{\text{mean}},
        P_{\text{std}}^2)`, noise the target, and take the squared error
        in *pixel* space.  The EDM weighting is already inside
        :math:`D_\theta`, so no extra loss weight appears here.
        """
        batch = int(next_frame.shape[0])
        device = next_frame.device.type
        if sigma is None:
            log_sigma = (
                lucid.randn((batch,), device=device) * self.config.p_std
                + self.config.p_mean
            )
            sigma = lucid.exp(log_sigma)

        # Offset noise: one scalar per sample and channel, on top of the
        # usual isotropic draw.  It lets the target's overall level shift,
        # which isotropic noise at a single sigma cannot express — and it
        # exists only in the released configuration, not the paper.
        offset = (
            lucid.randn((batch, int(next_frame.shape[1]), 1, 1), device=device)
            * self.config.sigma_offset_noise
        )
        noised = (
            next_frame
            + offset
            + lucid.randn_like(next_frame) * sigma.reshape(-1, 1, 1, 1)
        )
        prediction = self.denoise(noised, sigma, frames, actions)
        loss = ((prediction - next_frame) ** 2).mean()
        return DIAMONDOutput(loss=loss, prediction=prediction, sigma=sigma)


class DIAMONDForWorldModeling(WorldModelingModel):
    r"""DIAMOND posed as a world model: imagination and its objectives.

    Parameters
    ----------
    config : DIAMONDConfig
        Frozen configuration.

    Attributes
    ----------
    diamond : DIAMONDModel
        The three networks.

    Notes
    -----
    Reference: Alonso et al., arXiv:2405.12399, Appendix F and
    Algorithm 1.

    The imagined trajectory is autoregressive in the strong sense: the
    frame the denoiser produced and the action the policy chose from it
    both become conditioning for the next step, so an error in either
    compounds.  That is the whole reason the paper cares which diffusion
    parameterisation it uses.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.diamond import (
    ...     DIAMONDConfig, DIAMONDForWorldModeling)
    >>> config = DIAMONDConfig(
    ...     sample_size=16, unet_channels=(8, 8), unet_layers=(1, 1),
    ...     reward_channels=(8, 8), reward_layers=(1, 1),
    ...     actor_channels=(8, 8), actor_layers=(1, 1),
    ...     cond_dim=16, reward_cond_dim=8, reward_lstm_dim=16,
    ...     actor_lstm_dim=16, num_actions=4, horizon=3)
    >>> model = DIAMONDForWorldModeling(config).eval()
    >>> frames = lucid.randn((2, 4, 3, 16, 16))
    >>> actions = lucid.tensor([[0, 1, 2, 3], [1, 1, 0, 2]], dtype=lucid.int64)
    >>> with lucid.no_grad():
    ...     out = model(frames, actions)
    >>> out.frames.shape
    (2, 3, 3, 16, 16)
    """

    config_class: ClassVar[type[DIAMONDConfig]] = DIAMONDConfig

    def __init__(self, config: DIAMONDConfig) -> None:
        """Initialise the wrapper. See the class docstring for parameters."""
        super().__init__(config)
        self.config: DIAMONDConfig = config
        self.diamond = DIAMONDModel(config)

    def act(self, frame: Tensor, state: tuple[Tensor, Tensor] | None = None) -> Tensor:
        """Sample an action from the policy.

        Parameters
        ----------
        frame : Tensor
            ``(B, C, H, W)``.
        state : tuple of Tensor or None, optional
            Carried actor-critic LSTM state.

        Returns
        -------
        Tensor
            ``(B,)`` action indices.
        """
        logits, _value, _state = self.diamond.step_actor_critic(frame, state)
        return lucid.multinomial(lucid.softmax(logits, dim=-1), num_samples=1).reshape(
            -1
        )

    @override
    def forward(  # type: ignore[override]
        self, frames: Tensor, actions: Tensor, *, horizon: int | None = None
    ) -> DIAMONDBehaviorOutput:
        r"""Imagine a trajectory and score the policy on it.

        Parameters
        ----------
        frames : Tensor
            ``(B, L, C, H, W)`` real frames the rollout starts from.
        actions : Tensor
            ``(B, L)`` the actions that produced them.
        horizon : int or None, optional, keyword-only
            Steps to imagine; defaults to the configured 15.

        Returns
        -------
        DIAMONDBehaviorOutput
            The two losses, the policy's entropy, the
            :math:`\lambda`-returns and every imagined frame.

        Notes
        -----
        Reference: Alonso et al., arXiv:2405.12399, equations 14-16.
        The value target stops gradients, and the policy is REINFORCE
        against :math:`\Lambda_t - V(x_t)` — an advantage estimate, but
        one built from a *state-value* baseline rather than a critic.
        """
        horizon = self.config.horizon if horizon is None else horizon
        if horizon < 2:
            raise ValueError(
                f"lambda-returns need at least two imagined states, got {horizon}"
            )
        config = self.config
        history = frames
        history_actions = actions

        reward_state: tuple[Tensor, Tensor] | None = None
        actor_state: tuple[Tensor, Tensor] | None = None
        # The reward model reads a transition, so the burn-in can only
        # feed it pairs — one fewer step than the actor-critic gets.
        for step in range(int(frames.shape[1])):
            if step + 1 < int(frames.shape[1]):
                _r, _d, reward_state = self.diamond.step_reward_end(
                    frames[:, step],
                    frames[:, step + 1],
                    actions[:, step],
                    reward_state,
                )
            _l, _v, actor_state = self.diamond.step_actor_critic(
                frames[:, step], actor_state
            )

        imagined: list[Tensor] = []
        log_probs: list[Tensor] = []
        entropies: list[Tensor] = []
        values: list[Tensor] = []
        rewards: list[Tensor] = []
        continues: list[Tensor] = []

        for _ in range(horizon):
            logits, value, actor_state = self.diamond.step_actor_critic(
                history[:, -1], actor_state
            )
            probs = lucid.softmax(logits, dim=-1)
            action = lucid.multinomial(probs, num_samples=1).reshape(-1)
            log_prob = lucid.log(probs + 1e-8)
            # Picked out with a one-hot rather than advanced indexing: the
            # gradient path is the same and it does not depend on how the
            # backend handles a pair of index tensors.
            onehot = F.one_hot(action, num_classes=config.num_actions).to(
                log_prob.dtype
            )
            chosen = (log_prob * onehot).sum(dim=-1)
            entropy = -(probs * log_prob).sum(dim=-1)

            frame = self.diamond.imagine_frame(history, history_actions)
            reward_logits, end_logits, reward_state = self.diamond.step_reward_end(
                history[:, -1], frame, action, reward_state
            )
            # Three classes, in the order the clipping produces them.
            reward_probs = lucid.softmax(reward_logits, dim=-1)
            reward = reward_probs[:, 2] - reward_probs[:, 0]
            keep = lucid.softmax(end_logits, dim=-1)[:, 0]

            imagined.append(frame)
            log_probs.append(chosen)
            entropies.append(entropy)
            values.append(value)
            rewards.append(reward)
            continues.append(keep)

            history = lucid.cat(
                [
                    history[:, 1:],
                    frame.reshape(int(frame.shape[0]), 1, *frame.shape[1:]),
                ],
                dim=1,
            )
            history_actions = lucid.cat(
                [history_actions[:, 1:], action.reshape(-1, 1)], dim=1
            )

        _logits, bootstrap, _state = self.diamond.step_actor_critic(
            history[:, -1], actor_state
        )
        value_seq = lucid.stack([*values, bootstrap], dim=1)
        reward_seq = lucid.stack([*rewards, lucid.zeros_like(rewards[0])], dim=1)
        discount = lucid.stack(continues, dim=1) * config.gamma
        discount = lucid.cat([discount, lucid.zeros_like(discount[:, :1])], dim=1)

        returns = lambda_return(reward_seq, value_seq, discount, config.lambda_)
        advantage = (returns - value_seq[:, :horizon]).detach()

        log_prob_seq = lucid.stack(log_probs, dim=1)
        entropy_seq = lucid.stack(entropies, dim=1)
        policy_loss = -(
            log_prob_seq * advantage + config.entropy_weight * entropy_seq
        ).mean()
        value_loss = ((value_seq[:, :horizon] - returns.detach()) ** 2).mean()

        return DIAMONDBehaviorOutput(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy_seq.mean(),
            returns=returns,
            frames=lucid.stack(imagined, dim=1),
            history=history,
            history_actions=history_actions,
        )

    def world_model_loss(
        self, frames: Tensor, actions: Tensor, next_frame: Tensor
    ) -> DIAMONDOutput:
        """Train the denoiser — :meth:`DIAMONDModel.forward` by another name.

        Parameters
        ----------
        frames : Tensor
            ``(B, L, C, H, W)`` clean history.
        actions : Tensor
            ``(B, L)`` action indices.
        next_frame : Tensor
            ``(B, C, H, W)`` target.

        Returns
        -------
        DIAMONDOutput
            Loss, denoised frame, and the noise levels used.
        """
        return cast(DIAMONDOutput, self.diamond(frames, actions, next_frame))

    def reward_end_loss(
        self, frames: Tensor, actions: Tensor, rewards: Tensor, ends: Tensor
    ) -> Tensor:
        r"""Cross-entropy on the reward's *sign* and on termination.

        Parameters
        ----------
        frames : Tensor
            ``(B, T, C, H, W)``.
        actions : Tensor
            ``(B, T)``.
        rewards : Tensor
            ``(B, T)`` real rewards.  Only their sign is predicted —
            Algorithm 1 writes ``CE(r_hat, sign(r))``, which is all the
            environment's clipping to :math:`\{-1, 0, 1\}` leaves.
        ends : Tensor
            ``(B, T)`` termination flags in ``{0, 1}``.

        Returns
        -------
        Tensor
            Scalar, the two cross-entropies summed.
        """
        state: tuple[Tensor, Tensor] | None = None
        reward_loss = lucid.zeros(())
        end_loss = lucid.zeros(())
        steps = int(frames.shape[1]) - 1
        if steps < 1:
            raise ValueError(
                f"the reward model reads transitions, so it needs at least "
                f"two frames, got {int(frames.shape[1])}"
            )
        for step in range(steps):
            logits, end_logits, state = self.diamond.step_reward_end(
                frames[:, step], frames[:, step + 1], actions[:, step], state
            )
            target = (rewards[:, step].sign() + 1.0).to(lucid.int64)
            reward_loss = reward_loss + F.cross_entropy(logits, target)
            end_loss = end_loss + F.cross_entropy(
                end_logits, ends[:, step].to(lucid.int64)
            )
        return (reward_loss + end_loss) / float(steps)
