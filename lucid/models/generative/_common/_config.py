"""Shared base configs for ``lucid.models.generative``.

Mirrors the role of :class:`LanguageModelConfig` for text models: the common
fields every generative family needs are captured here once, so family
configs only add their unique knobs.

Two tiers:

    * :class:`GenerativeModelConfig` — every generative family (VAE / DDPM /
      NCSN / future flow models).  Holds the image shape and the trunk
      activation.
    * :class:`DiffusionModelConfig` — adds noise-schedule knobs used by
      every diffusion family.  VAE skips this tier.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal, override

from lucid.models._base import ModelConfig

# Activation alias accepted by every generative family.  Diffusion U-Nets
# typically use SiLU (Swish); VAEs use either SiLU or ReLU; the world
# models split — PlaNet specifies ReLU, Dreamer specifies ELU.
GenerativeActivation = Literal["silu", "swish", "relu", "gelu", "elu"]

# Noise schedule shape — see ``make_beta_schedule`` in ``_utils/_generative``.
BetaSchedule = Literal["linear", "cosine", "scaled_linear"]

# Factorised base distribution of a normalizing flow's latent space — see
# ``flow_prior_log_prob`` in ``_utils/_generative``.
FlowPrior = Literal["logistic", "gaussian"]


@dataclass(frozen=True)
class GenerativeModelConfig(ModelConfig):
    """Shared base for image-generative families.

    Args:
        sample_size: Output spatial resolution in pixels.  Square images use
            an int (``32`` → ``32 × 32``); rectangular targets pass a tuple.
        in_channels: Input image channels (3 for RGB, 1 for greyscale, 4
            for VAE latent samples, …).
        out_channels: Output channels — usually equal to ``in_channels``;
            diffusion models that predict variance use ``2 * in_channels``.
        act_fn: Activation used inside the trunk.  Most modern image
            generators default to ``"silu"``.
    """

    model_type: ClassVar[str] = "generative"

    sample_size: int | tuple[int, int] = 32
    in_channels: int = 3
    out_channels: int = 3
    act_fn: GenerativeActivation = "silu"

    def __post_init__(self) -> None:
        if isinstance(self.sample_size, tuple):
            if len(self.sample_size) != 2 or any(s <= 0 for s in self.sample_size):
                raise ValueError(
                    f"sample_size tuple must be (H, W) with both positive, got {self.sample_size}"
                )
        elif self.sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {self.sample_size}")
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {self.in_channels}")
        if self.out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {self.out_channels}")


@dataclass(frozen=True)
class DiffusionModelConfig(GenerativeModelConfig):
    """Shared base for *diffusion*-family generative models.

    Args:
        num_train_timesteps: Number of forward-process steps ``T``.  DDPM
            uses 1000, NCSN traditionally a smaller grid.
        beta_start: First step's noise rate ``β_1``, the low end of the
            linear schedule.  Ignored when ``beta_schedule == "cosine"``.
        beta_end: Last step's noise rate ``β_T``, the high end of the
            same schedule.  Ignored when ``beta_schedule == "cosine"``.
        beta_schedule: ``"linear"`` (Ho et al., 2020), ``"scaled_linear"``
            (Rombach et al., 2022 — linear in ``sqrt(beta)``, then squared;
            *not* the same curve as ``"linear"`` between the same
            endpoints) or ``"cosine"`` (Nichol
            & Dhariwal, 2021 — improves low-resolution sample quality).
        prediction_type: What the network predicts at each step.  ``"epsilon"``
            (the noise, default) is the canonical Ho parameterisation;
            ``"sample"`` predicts ``x_0`` directly; ``"v_prediction"`` is the
            Imagen / Progressive Distillation reparameterisation.
    """

    model_type: ClassVar[str] = "diffusion"

    num_train_timesteps: int = 1_000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: BetaSchedule = "linear"
    prediction_type: Literal["epsilon", "sample", "v_prediction"] = "epsilon"

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_train_timesteps <= 0:
            raise ValueError(
                f"num_train_timesteps must be positive, got {self.num_train_timesteps}"
            )
        if not 0.0 < self.beta_start < self.beta_end < 1.0:
            raise ValueError(
                f"Require 0 < beta_start < beta_end < 1, got "
                f"start={self.beta_start}, end={self.beta_end}"
            )


@dataclass(frozen=True)
class NormalizingFlowConfig(GenerativeModelConfig):
    """Shared base for *normalizing-flow* families (exact likelihood).

    A flow is a bijection ``f`` between data and a factorised latent space,
    trained by maximising ``log p(x) = log p_H(f(x)) + log|det ∂f/∂x|``.
    Every family therefore needs to declare which base distribution the
    latent is measured against; the bijection's own knobs (coupling depth,
    scale parameterisation, …) stay in the family config.

    Args:
        prior: Factorised base distribution over ``h = f(x)``.
            ``"logistic"`` — heavier-tailed, the default in Dinh et al.,
            2014 for dequantised pixel data; ``"gaussian"`` — standard
            normal.
    """

    model_type: ClassVar[str] = "normalizing_flow"

    prior: FlowPrior = "logistic"


__all__ = [
    "GenerativeModelConfig",
    "DiffusionModelConfig",
    "NormalizingFlowConfig",
    "WorldModelConfig",
    "GenerativeActivation",
    "BetaSchedule",
    "FlowPrior",
]


#: The only frame size the shared world-model convolutional schedule produces —
#: see :mod:`lucid.models.generative._common._pixel_nets`.
WORLD_MODEL_IMAGE_SIZE = 64


@dataclass(frozen=True)
class WorldModelConfig(GenerativeModelConfig):
    r"""Shared base for *world-model* families (latent dynamics from pixels).

    A world model learns an environment's transition function in a compact
    latent space and generates futures there, never rendering a frame in
    order to decide anything.  Every family in this group carries PlaNet's
    recurrent state-space model — a deterministic path ``h`` beside a
    stochastic latent ``s`` — and reads pixels through the same
    convolutional stack, so the state geometry, the encoder width and the
    variational bound's two knobs are declared once here.

    What each family does *with* the learned dynamics is what separates
    them, and that stays in the family config: PlaNet plans over them,
    Dreamer trains an actor and a critic on them.

    Args:
        sample_size: Frame resolution.  Must be 64 — the shared
            convolutional schedule lands on that and nothing else.
        action_dim: Width of the action vector conditioning the
            transition.  Genuinely task-dependent — the Control Suite
            tasks range from 1 to 12 — so there is no paper value to
            inherit and it must be set per use.
        stoch_size: Width of the stochastic latent :math:`s`.
        deter_size: Width of the deterministic latent :math:`h`.
        hidden_size: Width of the hidden layer inside the RSSM heads.
        cnn_depth: Channel width of the first encoder convolution; the
            stack widens ``depth, 2*depth, 4*depth, 8*depth``.  A scale
            knob, not a shape knob — the spatial schedule is fixed.
        min_std: Floor on the latent standard deviation.
        mean_only: Take the latent's mean instead of sampling it, making
            the whole recurrence deterministic.
        free_nats: The KL is clamped below at this value before it enters
            the loss, so no gradient is spent driving an already-small
            divergence lower.
        kl_weight: Multiplier :math:`\beta` on the clamped KL.

    Notes:
        Every default here is a value **both** PlaNet (Hafner et al., 2019)
        and Dreamer (Hafner et al., 2020) state.  Fields the two papers
        specify differently — ``act_fn``, the reward head's width and depth
        — are deliberately *not* hoisted, because this class would then
        have to pick a number neither paper backs.  Promoting a field here
        means the families were observed to agree, not that they could.
    """

    model_type: ClassVar[str] = "world_model"

    sample_size: int | tuple[int, int] = WORLD_MODEL_IMAGE_SIZE

    action_dim: int = 1

    stoch_size: int = 30
    deter_size: int = 200
    hidden_size: int = 200
    cnn_depth: int = 32
    min_std: float = 0.1
    mean_only: bool = False

    free_nats: float = 3.0
    kl_weight: float = 1.0

    @override
    def __post_init__(self) -> None:
        super().__post_init__()

        size = self.sample_size
        square = size if isinstance(size, int) else None
        if isinstance(size, tuple):
            square = size[0] if size[0] == size[1] else None
        if square != WORLD_MODEL_IMAGE_SIZE:
            raise ValueError(
                f"the world models' convolutional schedule only produces "
                f"{WORLD_MODEL_IMAGE_SIZE}x{WORLD_MODEL_IMAGE_SIZE} frames; got "
                f"sample_size={self.sample_size}. Scale the model with "
                f"cnn_depth / deter_size / stoch_size instead."
            )

        for name, value in (
            ("action_dim", self.action_dim),
            ("stoch_size", self.stoch_size),
            ("deter_size", self.deter_size),
            ("hidden_size", self.hidden_size),
            ("cnn_depth", self.cnn_depth),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.min_std <= 0.0:
            raise ValueError(f"min_std must be positive, got {self.min_std}")
        if self.free_nats < 0.0:
            raise ValueError(f"free_nats must be non-negative, got {self.free_nats}")
        if self.kl_weight < 0.0:
            raise ValueError(f"kl_weight must be non-negative, got {self.kl_weight}")

    @property
    def embed_size(self) -> int:
        """Width of the encoder output — ``8 * cnn_depth`` over a 2x2 grid."""
        return 8 * self.cnn_depth * 2 * 2

    @property
    def latent_size(self) -> int:
        """Width of the full latent ``[h; s]`` the decoder and heads read."""
        return self.deter_size + self.stoch_size
