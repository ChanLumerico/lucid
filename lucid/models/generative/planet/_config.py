"""PlaNet configuration — Hafner et al., 2019.

The Deep Planning Network: a latent dynamics model trained from pixels, whose
contribution is the recurrent state-space model (RSSM) that keeps a
deterministic path and a stochastic path side by side.

Defaults reproduce the paper's DeepMind Control Suite setup — 64x64 frames, a
200-unit deterministic state, a 30-unit stochastic state, a 1024-wide
embedding, and 3 free nats on the KL.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal, override

from lucid.models._meta import model_family_meta
from lucid.models.generative._config import GenerativeModelConfig

#: The paper's convolutional schedule only lands on this resolution — see
#: :class:`PlaNetConfig.__post_init__`.
PLANET_IMAGE_SIZE = 64


@model_family_meta(
    canonical_name="PlaNet",
    citation=(
        "Hafner, Danijar, et al. "
        '"Learning Latent Dynamics for Planning from Pixels." '
        "Proceedings of the 36th International Conference on Machine "
        "Learning, 2019, pp. 2555-2565."
    ),
    theory=r"""
    PlaNet learns the dynamics of an environment *in latent space* and plans
    there, never reconstructing a frame in order to choose an action.  The
    difficulty it addresses is that pixels are a partial observation: the
    agent must both remember what it has seen and stay uncertain about what
    it has not.

    Its answer is the **recurrent state-space model**, which refuses to pick
    between a deterministic and a stochastic latent and carries both.  A
    deterministic path :math:`h_t` is updated by a gated recurrence, and a
    stochastic latent :math:`s_t` is drawn conditioned on it:

    .. math::

        h_t = f\big(h_{t-1},\, s_{t-1},\, a_{t-1}\big),
        \qquad
        s_t \sim p\big(s_t \mid h_t\big).

    The motivation is stated as an ablation in the paper: a purely
    stochastic model cannot retain information over many steps, because each
    step's sampling noise degrades it; a purely deterministic model cannot
    represent the several futures a partially-observed environment permits,
    so it averages them and blurs.  Only the model carrying both matches the
    reported performance.

    Training maximises a variational bound.  An encoder supplies an
    approximate posterior :math:`q(s_t \mid h_t, o_t)` that has seen the
    frame; the objective rewards reconstructing the observation and the
    reward from that posterior while pulling the dynamics' own prior toward
    it:

    .. math::

        \mathcal{L} =
            \underbrace{\mathbb{E}_q\big[\log p(o_t \mid h_t, s_t)
                                    + \log p(r_t \mid h_t, s_t)\big]}
                       _{\text{reconstruction}}
            \;-\;
            \beta \underbrace{\mathrm{KL}\big(q(s_t \mid h_t, o_t)
                              \,\big\|\, p(s_t \mid h_t)\big)}
                       _{\text{consistency}} .

    The KL is clamped below at a **free-nats** threshold, so no gradient is
    spent driving an already-small divergence lower — without it the
    posterior collapses onto the prior and the latent stops carrying the
    observation.  The asymmetry is worth noting: the reconstruction term
    never reaches the prior head at all, since nothing reconstructed is
    computed from the prior.  The KL is the prior's only teacher.

    Once trained, planning runs the prior forward alone — the model
    imagines action sequences and scores them without touching the
    environment.  That the rollout happens entirely in a compact latent
    makes searching over thousands of candidate trajectories affordable,
    which is what lets PlaNet match model-free agents at a fraction of the
    episodes.
    """,
)
@dataclass(frozen=True)
class PlaNetConfig(GenerativeModelConfig):
    r"""Frozen configuration for the PlaNet family.

    Defaults reproduce Hafner et al., 2019 on the DeepMind Control Suite.

    Parameters
    ----------
    sample_size : int or tuple of int, default=64
        Frame resolution.  Must be 64 — see Notes.
    in_channels : int, default=3
        Observation channels.
    out_channels : int, default=3
        Reconstruction channels.
    act_fn : {"silu", "swish", "relu", "gelu"}, default="relu"
        Activation throughout.  The paper uses ReLU, which is why this
        family overrides the generative-domain default of ``"silu"``.
    action_dim : int, default=1
        Width of the action vector conditioning the transition.  Genuinely
        task-dependent — the Control Suite tasks range from 1 to 12 — so
        there is no paper value to inherit and it must be set per use.
    stoch_size : int, default=30
        Width of the stochastic latent :math:`s`.
    deter_size : int, default=200
        Width of the deterministic latent :math:`h`.
    hidden_size : int, default=200
        Width of the hidden layer inside the RSSM heads.
    cnn_depth : int, default=32
        Channel width of the first encoder convolution; the stack widens
        ``depth, 2*depth, 4*depth, 8*depth``.  A scale knob, not a shape
        knob — the spatial schedule is fixed.
    min_std : float, default=0.1
        Floor on the latent standard deviation.
    free_nats : float, default=3.0
        The KL is clamped below at this value before it enters the loss.
    kl_weight : float, default=1.0
        Multiplier :math:`\beta` on the clamped KL.
    reward_hidden : int, default=200
        Width of each reward-head hidden layer.
    reward_layers : int, default=2
        Number of reward-head hidden layers.

    Notes
    -----
    Reference: Hafner, Lillicrap, Fischer, Villegas, Ha, Lee, and Davidson,
    *"Learning Latent Dynamics for Planning from Pixels"*, ICML, 2019
    (arXiv:1811.04551).

    ``sample_size`` is pinned to 64 rather than made general.  The encoder
    is four stride-2 convolutions with 4x4 kernels and the decoder four
    transposed convolutions with kernels ``5, 5, 6, 6`` — an irregular
    schedule that lands on ``64`` exactly (``1 -> 5 -> 13 -> 30 -> 64``) and
    on nothing else.  Generalising it would mean inventing a schedule the
    paper does not give, so an unsupported size raises here instead of
    silently reconstructing to the wrong shape.  Scale the model with
    ``cnn_depth`` / ``deter_size`` / ``stoch_size``.

    Examples
    --------
    >>> from lucid.models.generative.planet import PlaNetConfig
    >>> cfg = PlaNetConfig(action_dim=6)
    >>> cfg.stoch_size, cfg.deter_size, cfg.free_nats
    (30, 200, 3.0)
    >>> cfg.latent_size
    230
    """

    model_type: ClassVar[str] = "planet"

    sample_size: int | tuple[int, int] = PLANET_IMAGE_SIZE
    act_fn: Literal["silu", "swish", "relu", "gelu"] = "relu"

    action_dim: int = 1

    stoch_size: int = 30
    deter_size: int = 200
    hidden_size: int = 200
    cnn_depth: int = 32
    min_std: float = 0.1

    free_nats: float = 3.0
    kl_weight: float = 1.0

    reward_hidden: int = 200
    reward_layers: int = 2

    @override
    def __post_init__(self) -> None:
        super().__post_init__()

        size = self.sample_size
        square = size if isinstance(size, int) else None
        if isinstance(size, tuple):
            square = size[0] if size[0] == size[1] else None
        if square != PLANET_IMAGE_SIZE:
            raise ValueError(
                f"PlaNet's convolutional schedule only produces "
                f"{PLANET_IMAGE_SIZE}x{PLANET_IMAGE_SIZE} frames; got "
                f"sample_size={self.sample_size}. Scale the model with "
                f"cnn_depth / deter_size / stoch_size instead."
            )

        for name, value in (
            ("action_dim", self.action_dim),
            ("stoch_size", self.stoch_size),
            ("deter_size", self.deter_size),
            ("hidden_size", self.hidden_size),
            ("cnn_depth", self.cnn_depth),
            ("reward_hidden", self.reward_hidden),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.reward_layers <= 0:
            raise ValueError(
                f"reward_layers must be positive, got {self.reward_layers}"
            )
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
        """Width of the full latent ``[h; s]`` the decoder and reward head read."""
        return self.deter_size + self.stoch_size
