"""Dreamer configuration — Hafner et al., 2020.

Dreamer keeps PlaNet's world model and replaces its planner with two learned
networks: an actor that proposes actions and a critic that scores them.
Both are trained on trajectories the model imagines, so the gradient of the
return flows *through* the learned dynamics rather than through sampled
environment interaction.

Defaults reproduce the paper's DeepMind Control Suite setup — 64x64 frames,
a 200-unit deterministic state, a 30-unit stochastic state, 3-layer
300-unit ELU heads, a 15-step imagination horizon, and TD(0.95) returns.
"""

from dataclasses import dataclass
from typing import ClassVar, override

from lucid.models._meta import model_family_meta
from lucid.models.generative._config import GenerativeActivation, WorldModelConfig


@model_family_meta(
    canonical_name="Dreamer",
    citation=(
        "Hafner, Danijar, et al. "
        '"Dream to Control: Learning Behaviors by Latent Imagination." '
        "International Conference on Learning Representations, 2020."
    ),
    theory=r"""
    Dreamer inherits PlaNet's recurrent state-space model and throws away
    its planner.  Where PlaNet searches over action sequences at every step
    — thousands of imagined trajectories to choose one action — Dreamer
    *learns* a policy, so acting costs a single forward pass.

    The trade is that a learned policy needs a learning signal, and the
    paper's contribution is where that signal comes from.  An **actor**
    :math:`a_\tau \sim q_\phi(a_\tau \mid s_\tau)` and a **critic**
    :math:`v_\psi(s_\tau)` are trained entirely on trajectories the world
    model imagines, never on environment interaction.  Because the imagined
    states are produced by a differentiable transition and the actions are
    reparameterised samples, the gradient of a predicted return flows
    *backwards through the dynamics* into the actor:

    .. math::

        \max_\phi \; \mathbb{E}_{q_\theta, q_\phi}
            \Big[\textstyle\sum_{\tau} V_\lambda(s_\tau)\Big].

    This is what separates it from a model-free actor-critic, which can
    only estimate that gradient from scalar rewards it has actually
    observed.  Here the model supplies an analytic path.

    The target :math:`V_\lambda` is an exponentially-weighted average of
    :math:`k`-step returns — TD(:math:`\lambda`) computed inside the
    imagination:

    .. math::

        V_\lambda(s_\tau) = (1 - \lambda)
            \sum_{n=1}^{H-1} \lambda^{n-1} V_N^n(s_\tau)
            + \lambda^{H-1} V_N^H(s_\tau),

    with each :math:`V_N^k` summing discounted imagined rewards and
    bootstrapping from the critic.  The weighting is the usual bias-variance
    dial: short horizons trust the critic, long ones trust the model, and
    :math:`\lambda` interpolates.  The critic then regresses onto that same
    quantity, which is why the two networks are trained together but with
    the target held fixed.

    Imagination is short — 15 steps in the paper — because the model's
    error compounds, and the critic's bootstrap is what lets a short
    horizon still represent long-term value.
    """,
)
@dataclass(frozen=True)
class DreamerConfig(WorldModelConfig):
    r"""Frozen configuration for the Dreamer family.

    Defaults reproduce Hafner et al., 2020 on the DeepMind Control Suite.
    Fields shared with PlaNet are inherited from :class:`WorldModelConfig`.

    Parameters
    ----------
    act_fn : {"silu", "swish", "relu", "gelu", "elu"}, default="elu"
        Activation throughout.  The paper specifies ELU, where PlaNet
        specifies ReLU — the two families genuinely differ here.
    horizon : int, default=15
        How many steps each imagined trajectory runs for.  Short on
        purpose: model error compounds, and the critic's bootstrap is what
        carries value beyond the horizon.
    discount : float, default=0.99
        Reward discount :math:`\gamma`.
    lambda_ : float, default=0.95
        TD(:math:`\lambda`) weighting between short- and long-horizon
        returns.  Trailing underscore because ``lambda`` is a keyword.
    actor_hidden, actor_layers : int, default=300, 3
        Width and depth of the actor.
    value_hidden, value_layers : int, default=300, 3
        Width and depth of the critic.
    actor_min_std : float, default=1e-4
        Floor on the action distribution's scale.
    actor_init_std : float, default=5.0
        Offset added before ``softplus`` so the policy starts wide enough
        to explore.
    actor_mean_scale : float, default=5.0
        The mean is passed through ``s * tanh(x / s)`` before squashing,
        which keeps it from saturating the outer ``tanh``.
    reward_hidden, reward_layers : int, default=300, 3
        Width and depth of the reward head.

    Notes
    -----
    Reference: Hafner, Lillicrap, Ba, and Norouzi, *"Dream to Control:
    Learning Behaviors by Latent Imagination"*, ICLR, 2020
    (arXiv:1912.01603).

    The actor emits a ``tanh``-squashed diagonal Gaussian, so actions are
    bounded to ``(-1, 1)`` — the Control Suite's range.  ``actor_min_std``,
    ``actor_init_std`` and ``actor_mean_scale`` are **not stated in the
    paper**; they are the released implementation's, and are exposed here
    rather than buried so a reader can see which numbers have a citation
    and which have a source.

    The three losses need **three optimisers over three parameter groups**
    — see :class:`DreamerForWorldModeling`.  Summing them and taking one
    step is not this algorithm: the actor's gradient would then also
    descend the world model, and the critic would chase a target it is
    simultaneously moving.

    Examples
    --------
    >>> from lucid.models.generative.dreamer import DreamerConfig
    >>> cfg = DreamerConfig(action_dim=6)
    >>> cfg.horizon, cfg.lambda_, cfg.discount
    (15, 0.95, 0.99)
    >>> cfg.act_fn
    'elu'
    """

    model_type: ClassVar[str] = "dreamer"

    act_fn: GenerativeActivation = "elu"

    horizon: int = 15
    discount: float = 0.99
    lambda_: float = 0.95

    actor_hidden: int = 300
    actor_layers: int = 3
    value_hidden: int = 300
    value_layers: int = 3

    actor_min_std: float = 1e-4
    actor_init_std: float = 5.0
    actor_mean_scale: float = 5.0

    reward_hidden: int = 300
    reward_layers: int = 3

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        for name, value in (
            ("horizon", self.horizon),
            ("actor_hidden", self.actor_hidden),
            ("actor_layers", self.actor_layers),
            ("value_hidden", self.value_hidden),
            ("value_layers", self.value_layers),
            ("reward_hidden", self.reward_hidden),
            ("reward_layers", self.reward_layers),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError(f"discount must be in (0, 1], got {self.discount}")
        if not 0.0 <= self.lambda_ <= 1.0:
            raise ValueError(f"lambda_ must be in [0, 1], got {self.lambda_}")
        if self.actor_min_std <= 0.0:
            raise ValueError(
                f"actor_min_std must be positive, got {self.actor_min_std}"
            )
        if self.actor_mean_scale <= 0.0:
            raise ValueError(
                f"actor_mean_scale must be positive, got {self.actor_mean_scale}"
            )
