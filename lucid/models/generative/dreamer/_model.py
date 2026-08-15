r"""Dreamer — Hafner et al., 2020.

PlaNet's world model with its planner replaced by two learned networks.
The world model is trained exactly as before; what is new is that an actor
and a critic are fitted on trajectories the model *imagines*, so the
gradient of a predicted return reaches the policy through the learned
dynamics rather than through sampled experience.

Layout
------
``DreamerModel`` is the world model — encoder, RSSM, decoder, reward head —
plus the actor and critic.  ``DreamerForWorldModeling`` adds the three
objectives.

Sequences are ``(B, T, ...)`` throughout, matching the rest of the zoo.
``actions[:, t]`` is the action taken *into* step ``t``.

Three losses, three optimisers
------------------------------
The world model, the actor and the critic are trained by three separate
optimisers over three disjoint parameter groups.  This is not a stylistic
choice: the actor *maximises* what the critic predicts, so descending their
sum would have the actor's gradient also drag the world model, and the
critic would chase a target it is simultaneously moving.  Use
:meth:`DreamerForWorldModeling.world_parameters`,
:meth:`~DreamerForWorldModeling.actor_parameters` and
:meth:`~DreamerForWorldModeling.value_parameters`.
"""

import math
from dataclasses import dataclass
from typing import ClassVar, cast, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._output import ModelOutput
from lucid.models.generative._pixel_nets import DenseHead, PixelDecoder, PixelEncoder
from lucid.models.generative._rssm import RSSM, RSSMState, rssm_kl
from lucid.models.generative.dreamer._config import DreamerConfig

__all__ = [
    "DreamerModel",
    "DreamerForWorldModeling",
    "DreamerOutput",
    "DreamerBehaviorOutput",
]


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class DreamerBehaviorOutput(ModelOutput):
    r"""What the imagination pass produces — the actor's and critic's terms.

    Attributes
    ----------
    actor_loss : Tensor
        ``-mean(w * V_lambda)``, minimised by the actor's optimiser only.
    value_loss : Tensor
        Half the discounted squared error between the critic and the
        (detached) :math:`V_\lambda` target.
    lambda_return : Tensor
        The targets themselves, ``(N, H)`` for ``N`` imagined trajectories
        over a horizon of ``H``.
    imagined_reward : Tensor
        Reward the model predicted along the imagined trajectories,
        ``(N, H + 1)``.
    imagined_value : Tensor
        The critic's estimate along the same states, ``(N, H + 1)``.
    imagined_action : Tensor
        Actions the actor proposed, ``(N, H, action_dim)``.

    Notes
    -----
    ``actor_loss`` and ``value_loss`` must be given **separate optimisers
    over disjoint parameter groups** — see the module docstring.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.dreamer import (
    ...     DreamerConfig, DreamerForWorldModeling)
    >>> cfg = DreamerConfig(action_dim=2, cnn_depth=2, stoch_size=4,
    ...                     deter_size=8, hidden_size=8, actor_hidden=8,
    ...                     value_hidden=8, reward_hidden=8, horizon=3)
    >>> model = DreamerForWorldModeling(cfg)
    >>> out = model(lucid.randn((1, 3, 3, 64, 64)), lucid.randn((1, 3, 2)),
    ...             lucid.randn((1, 3)))
    >>> out.behavior.lambda_return.shape
    (3, 3)
    """

    actor_loss: Tensor
    value_loss: Tensor
    lambda_return: Tensor
    imagined_reward: Tensor
    imagined_value: Tensor
    imagined_action: Tensor


@dataclass(slots=True)
class DreamerOutput(ModelOutput):
    r"""What :class:`DreamerModel` returns after filtering a trajectory.

    Attributes
    ----------
    observation : Tensor
        Reconstruction from the posterior states, ``(B, T, C, 64, 64)``.
    reward : Tensor
        Predicted reward at each posterior state, ``(B, T)``.
    value : Tensor
        The critic's estimate at each posterior state, ``(B, T)``.
    posterior_stoch, posterior_mean, posterior_std : Tensor
        The filtered latent and the Gaussian it came from, ``(B, T, S)``.
    prior_mean, prior_std : Tensor
        The Gaussian the dynamics predicted before seeing the frame,
        ``(B, T, S)``.
    deter : Tensor
        The deterministic path, ``(B, T, D)``.
    loss, recon_loss, reward_loss, kl_loss : Tensor or None
        World-model terms, set only by
        :class:`DreamerForWorldModeling`.  ``loss`` is the **world-model**
        loss alone — the behaviour losses live on
        :class:`DreamerBehaviorOutput`, because they belong to different
        optimisers.
    behavior : DreamerBehaviorOutput or None
        Actor and critic terms, set only by
        :class:`DreamerForWorldModeling`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.dreamer import DreamerConfig, DreamerModel
    >>> cfg = DreamerConfig(action_dim=2, cnn_depth=2, stoch_size=4,
    ...                     deter_size=8, hidden_size=8, actor_hidden=8,
    ...                     value_hidden=8, reward_hidden=8)
    >>> model = DreamerModel(cfg)
    >>> out = model(lucid.randn((1, 3, 3, 64, 64)), lucid.randn((1, 3, 2)))
    >>> out.observation.shape, out.value.shape
    ((1, 3, 3, 64, 64), (1, 3))
    """

    observation: Tensor
    reward: Tensor
    value: Tensor
    posterior_stoch: Tensor
    posterior_mean: Tensor
    posterior_std: Tensor
    prior_mean: Tensor
    prior_std: Tensor
    deter: Tensor

    loss: Tensor | None = None
    recon_loss: Tensor | None = None
    reward_loss: Tensor | None = None
    kl_loss: Tensor | None = None
    behavior: DreamerBehaviorOutput | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Internal building blocks
# ─────────────────────────────────────────────────────────────────────────────


class _Actor(nn.Module):
    r"""A ``tanh``-squashed diagonal Gaussian over actions.

    Reads a latent state and emits an action in ``(-1, 1)``.  The squash is
    what bounds the action to the Control Suite's range, and doing it with
    ``tanh`` — rather than clipping — keeps the sample differentiable, which
    is the whole point: Dreamer's actor gradient arrives *through* the
    action.

    Parameters
    ----------
    latent_size : int
        Width of the state read.
    hidden, layers : int
        Shape of the dense trunk.
    action_dim : int
        Width of the action emitted.
    act_fn : str
        Activation in the trunk.
    min_std, init_std, mean_scale : float
        See :class:`DreamerConfig`.

    Notes
    -----
    ``init_std`` is applied as an offset *inside* ``softplus``, chosen so
    that an untrained head — whose raw output is near zero — produces
    exactly ``init_std``.  Solving ``softplus(c) = init_std`` gives
    ``c = log(exp(init_std) - 1)``, which is what is added.
    """

    def __init__(
        self,
        latent_size: int,
        hidden: int,
        layers: int,
        action_dim: int,
        act_fn: str,
        min_std: float,
        init_std: float,
        mean_scale: float,
    ) -> None:
        """Initialise the actor. See the class docstring for parameters."""
        super().__init__()
        self.action_dim = action_dim
        self.min_std = min_std
        self.mean_scale = mean_scale
        # softplus(c) = init_std  =>  c = log(exp(init_std) - 1)
        self._raw_init_std = math.log(math.expm1(init_std))
        self.head = DenseHead(
            latent_size, hidden, layers, out_features=2 * action_dim, act_fn=act_fn
        )

    def distribution(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        """Return the pre-squash ``(mean, std)`` for a state.

        Parameters
        ----------
        feature : Tensor
            Latent state, ``(B, T, latent_size)``.

        Returns
        -------
        mean : Tensor
            Pre-squash mean, ``(B, T, action_dim)``, softly bounded to
            ``(-mean_scale, mean_scale)``.
        std : Tensor
            Scale, ``(B, T, action_dim)``, floored at ``min_std``.
        """
        out = cast(Tensor, self.head(feature))
        raw_mean = out[..., : self.action_dim]
        raw_std = out[..., self.action_dim :]
        mean = self.mean_scale * lucid.tanh(raw_mean / self.mean_scale)
        std = F.softplus(raw_std + self._raw_init_std) + self.min_std
        return mean, std

    @override
    def forward(  # type: ignore[override]
        self, feature: Tensor, *, sample: bool = True
    ) -> Tensor:
        """Propose an action for a state — ``(B, T, action_dim)`` in ``(-1, 1)``.

        Parameters
        ----------
        feature : Tensor
            Latent state, ``(B, T, latent_size)``.
        sample : bool, default=True, keyword-only
            Draw reparameterised (``True``) or take the squashed mean
            (``False``, which is how a trained policy should act).

        Returns
        -------
        Tensor
            Actions bounded to ``(-1, 1)``.
        """
        mean, std = self.distribution(feature)
        if not sample:
            return lucid.tanh(mean)
        noise = lucid.randn(
            tuple(int(s) for s in mean.shape), device=mean.device, dtype=mean.dtype
        )
        return lucid.tanh(mean + std * noise)


def _lambda_return(
    reward: Tensor, value: Tensor, discount: float, lambda_: float
) -> Tensor:
    r"""TD(:math:`\lambda`) returns over an imagined trajectory.

    Parameters
    ----------
    reward : Tensor
        Predicted reward at each imagined state, ``(N, H + 1)``.
    value : Tensor
        The critic's estimate at the same states, ``(N, H + 1)``.
    discount : float
        :math:`\gamma`.
    lambda_ : float
        :math:`\lambda`; ``0`` gives the one-step TD target, ``1`` the full
        Monte-Carlo return bootstrapped at the horizon.

    Returns
    -------
    Tensor
        Targets ``(N, H)``, one per state except the last, which is only
        ever used as the bootstrap.

    Notes
    -----
    Computed by the backward recursion the released implementation uses,

    .. math::

        V_\lambda(s_t) = r_t + \gamma\big[(1 - \lambda)\, v(s_{t+1})
                          + \lambda\, V_\lambda(s_{t+1})\big],

    terminated at :math:`V_\lambda(s_H) = v(s_H)`.  Written this way it
    costs one pass rather than the :math:`O(H^2)` the closed form suggests,
    and each :math:`\lambda` power appears exactly once.
    """
    horizon = int(reward.shape[1]) - 1
    if horizon < 1:
        raise ValueError(
            f"lambda-returns need at least two states, got {int(reward.shape[1])}"
        )

    agg = value[:, horizon]
    out: list[Tensor] = []
    for t in range(horizon - 1, -1, -1):
        inputs = reward[:, t] + discount * (1.0 - lambda_) * value[:, t + 1]
        agg = inputs + discount * lambda_ * agg
        out.append(agg)
    out.reverse()
    return lucid.stack(out, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Direct model
# ─────────────────────────────────────────────────────────────────────────────


class DreamerModel(PretrainedModel):
    r"""PlaNet's world model plus a learned actor and critic.

    The world model is unchanged from PlaNet — the same encoder, the same
    RSSM, the same decoder and reward head.  Two heads are added: an actor
    that proposes actions from a latent state and a critic that scores one.

    Parameters
    ----------
    config : DreamerConfig
        Frozen configuration.

    Notes
    -----
    Reference: Hafner, Lillicrap, Ba, and Norouzi, *"Dream to Control:
    Learning Behaviors by Latent Imagination"*, ICLR, 2020.

    ``actions[:, t]`` is the action taken *into* step ``t``, matching
    :class:`~lucid.models.generative.planet.PlaNetModel`.

    This class holds no environment and never steps one.  Everything it
    does — filtering, imagining, acting — happens in latent space, which is
    what makes the whole family trainable from a replay buffer alone.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.dreamer import DreamerConfig, DreamerModel
    >>> cfg = DreamerConfig(action_dim=2, cnn_depth=2, stoch_size=4,
    ...                     deter_size=8, hidden_size=8, actor_hidden=8,
    ...                     value_hidden=8, reward_hidden=8)
    >>> model = DreamerModel(cfg)
    >>> _, posteriors = model.observe(lucid.randn((1, 4, 3, 64, 64)),
    ...                               lucid.randn((1, 4, 2)))
    >>> model.act(posteriors, sample=False).shape
    (1, 4, 2)
    """

    config_class: ClassVar[type[DreamerConfig]] = DreamerConfig
    base_model_prefix: ClassVar[str] = "dreamer"

    def __init__(self, config: DreamerConfig) -> None:
        super().__init__(config)
        self.encoder = PixelEncoder(config.in_channels, config.cnn_depth, config.act_fn)
        self.rssm = RSSM(
            stoch_size=config.stoch_size,
            deter_size=config.deter_size,
            hidden_size=config.hidden_size,
            action_dim=config.action_dim,
            embed_size=config.embed_size,
            act_fn=config.act_fn,
            min_std=config.min_std,
        )
        self.decoder = PixelDecoder(
            config.latent_size, config.out_channels, config.cnn_depth, config.act_fn
        )
        self.reward_head = DenseHead(
            config.latent_size,
            config.reward_hidden,
            config.reward_layers,
            act_fn=config.act_fn,
            squeeze=True,
        )
        self.value_head = DenseHead(
            config.latent_size,
            config.value_hidden,
            config.value_layers,
            act_fn=config.act_fn,
            squeeze=True,
        )
        self.actor = _Actor(
            config.latent_size,
            config.actor_hidden,
            config.actor_layers,
            config.action_dim,
            config.act_fn,
            config.actor_min_std,
            config.actor_init_std,
            config.actor_mean_scale,
        )
        self._sample = not config.mean_only

    def encode(self, observations: Tensor) -> Tensor:
        """Embed a frame sequence — ``(B, T, C, 64, 64) -> (B, T, embed_size)``."""
        return cast(Tensor, self.encoder(observations))

    def observe(
        self,
        observations: Tensor,
        actions: Tensor,
        state: RSSMState | None = None,
        *,
        sample: bool | None = None,
    ) -> tuple[RSSMState, RSSMState]:
        """Filter a trajectory into posterior states.

        Parameters
        ----------
        observations : Tensor
            Frames, ``(B, T, C, 64, 64)``.
        actions : Tensor
            Actions taken *into* each step, ``(B, T, action_dim)``.
        state : RSSMState or None, optional
            Starting belief; ``None`` starts from zeros.
        sample : bool or None, optional, keyword-only
            Draw the latent (``True``) or take its mean (``False``).
            ``None`` follows the config's ``mean_only`` setting.

        Returns
        -------
        priors : RSSMState
            What the dynamics predicted, ``(B, T, ·)``.
        posteriors : RSSMState
            What they believed after seeing each frame, ``(B, T, ·)``.
        """
        draw = self._sample if sample is None else sample
        return self.rssm.observe(self.encode(observations), actions, state, sample=draw)

    def decode(self, state: RSSMState) -> Tensor:
        """Reconstruct frames from a state — ``(B, T, C, 64, 64)``."""
        return cast(Tensor, self.decoder(state.feature))

    def predict_reward(self, state: RSSMState) -> Tensor:
        """Predict reward from a state — ``(B, T)``."""
        return cast(Tensor, self.reward_head(state.feature))

    def predict_value(self, state: RSSMState) -> Tensor:
        """Estimate the value of a state — ``(B, T)``."""
        return cast(Tensor, self.value_head(state.feature))

    def act(self, state: RSSMState, *, sample: bool = True) -> Tensor:
        """Propose actions for a state — ``(B, T, action_dim)`` in ``(-1, 1)``.

        Parameters
        ----------
        state : RSSMState
            The belief to act from.
        sample : bool, default=True, keyword-only
            Draw from the policy (``True``) or take its squashed mean
            (``False``).

        Returns
        -------
        Tensor
            Actions bounded to ``(-1, 1)``.
        """
        return cast(Tensor, self.actor(state.feature, sample=sample))

    def imagine(
        self, state: RSSMState, horizon: int, *, sample: bool = True
    ) -> tuple[RSSMState, Tensor]:
        r"""Roll the dynamics forward under the actor's own policy.

        This is the loop the whole method rests on.  Unlike PlaNet's
        :meth:`~lucid.models.generative.planet.PlaNetModel.imagine`, which
        is handed a fixed action sequence to evaluate, here the action at
        each step is *produced* by the actor from the state the model just
        imagined — so the trajectory and the policy are coupled, and a
        gradient taken at the end reaches the policy at every step along
        the way.

        Parameters
        ----------
        state : RSSMState
            Flat starting beliefs, ``(N, ·)``.
        horizon : int
            Number of steps to imagine.
        sample : bool, default=True, keyword-only
            Draw both the latent and the action (``True``) or take their
            means (``False``).

        Returns
        -------
        states : RSSMState
            The imagined states including the start, ``(N, horizon + 1, ·)``.
        actions : Tensor
            What the actor proposed, ``(N, horizon, action_dim)``.

        Notes
        -----
        The start state is *not* detached here — that is the caller's
        decision, and :class:`DreamerForWorldModeling` does detach it so
        the actor's gradient cannot reach the world model.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")

        current = state
        deters, stochs, means, stds = [state.deter], [state.stoch], [], []
        actions: list[Tensor] = []
        for _ in range(horizon):
            feature = current.feature.reshape(int(current.deter.shape[0]), 1, -1)
            action = cast(Tensor, self.actor(feature, sample=sample))[:, 0]
            current = self.rssm.prior_step(current, action, sample=sample)
            actions.append(action)
            deters.append(current.deter)
            stochs.append(current.stoch)
            means.append(current.mean)
            stds.append(current.std)

        # The start state has no prior Gaussian of its own — it came from a
        # posterior — so its mean/std slots are filled with the first
        # imagined step's, keeping every field the same length.  Nothing
        # reads index 0 of mean/std; the KL is a world-model term and does
        # not run inside imagination.
        rolled = RSSMState(
            deter=lucid.stack(deters, dim=1),
            stoch=lucid.stack(stochs, dim=1),
            mean=lucid.stack([means[0]] + means, dim=1),
            std=lucid.stack([stds[0]] + stds, dim=1),
        )
        return rolled, lucid.stack(actions, dim=1)

    @override
    def forward(  # type: ignore[override]
        self, observations: Tensor, actions: Tensor
    ) -> DreamerOutput:
        priors, posteriors = self.observe(observations, actions)
        return DreamerOutput(
            observation=self.decode(posteriors),
            reward=self.predict_reward(posteriors),
            value=self.predict_value(posteriors),
            posterior_stoch=posteriors.stoch,
            posterior_mean=posteriors.mean,
            posterior_std=posteriors.std,
            prior_mean=priors.mean,
            prior_std=priors.std,
            deter=posteriors.deter,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task wrapper — the three objectives
# ─────────────────────────────────────────────────────────────────────────────


class DreamerForWorldModeling(PretrainedModel):
    r"""Dreamer with its world-model, actor and critic objectives.

    Three losses come out of one forward pass:

    - the **world-model** loss, identical in form to PlaNet's — pixel and
      reward reconstruction against a free-nats-clamped KL;
    - the **actor** loss, :math:`-\mathbb{E}[V_\lambda]` over trajectories
      imagined under the policy itself;
    - the **critic** loss, regressing onto those same targets held fixed.

    Parameters
    ----------
    config : DreamerConfig
        Frozen configuration.

    Notes
    -----
    Reference: Hafner, Lillicrap, Ba, and Norouzi, *"Dream to Control:
    Learning Behaviors by Latent Imagination"*, ICLR, 2020.

    **These three must be optimised separately.**  Summing them and taking
    one step is a different algorithm: the actor's gradient would descend
    the world model too, and the critic would chase a moving target.  The
    parameter groups are exposed as :meth:`world_parameters`,
    :meth:`actor_parameters` and :meth:`value_parameters`, which partition
    the model exactly.

    Imagination starts from **detached** posterior states, so no behaviour
    gradient reaches the encoder or the filtering that produced them.
    Within the horizon, though, the actor's gradient deliberately flows
    *through the learned dynamics* — through the RSSM transition, the
    reward head and the critic — because that analytic path is the paper's
    entire contribution.  Both the latent draw and the action draw are
    reparameterised, which is what lets a return computed 15 steps out
    reach the policy that chose step 1.  Only the actor's parameters are
    updated by it; that is what the parameter grouping is for.

    The critic is the exception: it reads detached states, so ``value_loss``
    reaches nothing but the critic itself.

    Reconstruction is a Gaussian log-likelihood with unit variance, so it
    reduces to a squared error summed over pixels and averaged over the
    batch — the same convention as PlaNet.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.dreamer import (
    ...     DreamerConfig, DreamerForWorldModeling)
    >>> cfg = DreamerConfig(action_dim=2, cnn_depth=2, stoch_size=4,
    ...                     deter_size=8, hidden_size=8, actor_hidden=8,
    ...                     value_hidden=8, reward_hidden=8, horizon=3)
    >>> model = DreamerForWorldModeling(cfg)
    >>> out = model(lucid.randn((1, 3, 3, 64, 64)), lucid.randn((1, 3, 2)),
    ...             lucid.randn((1, 3)))
    >>> bool(out.loss.ndim == 0), bool(out.behavior.actor_loss.ndim == 0)
    (True, True)
    """

    config_class: ClassVar[type[DreamerConfig]] = DreamerConfig
    base_model_prefix: ClassVar[str] = "dreamer"

    def __init__(self, config: DreamerConfig) -> None:
        super().__init__(config)
        self.dreamer = DreamerModel(config)
        self._free_nats = config.free_nats
        self._kl_weight = config.kl_weight
        self._horizon = config.horizon
        self._discount = config.discount
        self._lambda = config.lambda_

    def world_parameters(self) -> list[nn.Parameter]:
        """Everything the world-model loss trains — encoder, RSSM, decoder, reward.

        Returns
        -------
        list of Parameter
            The world model's parameters, disjoint from
            :meth:`actor_parameters` and :meth:`value_parameters`.
        """
        model = self.dreamer
        return [
            p
            for module in (model.encoder, model.rssm, model.decoder, model.reward_head)
            for p in module.parameters()
        ]

    def actor_parameters(self) -> list[nn.Parameter]:
        """The actor's parameters.

        Returns
        -------
        list of Parameter
            Trained by ``actor_loss`` alone.
        """
        return list(self.dreamer.actor.parameters())

    def value_parameters(self) -> list[nn.Parameter]:
        """The critic's parameters.

        Returns
        -------
        list of Parameter
            Trained by ``value_loss`` alone.
        """
        return list(self.dreamer.value_head.parameters())

    def _behavior(self, posteriors: RSSMState) -> DreamerBehaviorOutput:
        """Imagine under the policy and score it.

        Parameters
        ----------
        posteriors : RSSMState
            Filtered beliefs, ``(B, T, ·)``.  Flattened to ``(B * T, ·)`` and
            detached, so every filtered step becomes an independent
            imagination start.

        Returns
        -------
        DreamerBehaviorOutput
            The actor and critic terms.
        """
        b, t = int(posteriors.deter.shape[0]), int(posteriors.deter.shape[1])

        def flat(x: Tensor) -> Tensor:
            return x.reshape(b * t, int(x.shape[2])).detach()

        start = RSSMState(
            deter=flat(posteriors.deter),
            stoch=flat(posteriors.stoch),
            mean=flat(posteriors.mean),
            std=flat(posteriors.std),
        )

        states, actions = self.dreamer.imagine(start, self._horizon)
        reward = self.dreamer.predict_reward(states)
        value = self.dreamer.predict_value(states)

        returns = _lambda_return(reward, value, self._discount, self._lambda)

        # The released implementation weights both behaviour losses by the
        # cumulative discount, so a step the agent is unlikely to still be
        # around for counts for less.  Built as a constant: it is a weight,
        # never a thing to differentiate.
        weight = lucid.tensor(
            [[self._discount**i for i in range(self._horizon)]],
            device=returns.device,
            dtype=returns.dtype,
        )

        actor_loss = -(weight * returns).mean()

        # The critic reads *detached* states.  Its job is to regress onto
        # the target, not to shape the dynamics that produced it — and the
        # released implementation gets the same effect by giving the critic
        # its own tape over its own variables.  Doing it with a detach
        # instead means the separation survives a caller who reaches for
        # one optimiser anyway.
        detached = RSSMState(
            deter=states.deter.detach(),
            stoch=states.stoch.detach(),
            mean=states.mean.detach(),
            std=states.std.detach(),
        )
        target = returns.detach()
        predicted = self.dreamer.predict_value(detached)[:, : self._horizon]
        value_loss = (0.5 * weight * (predicted - target) ** 2).mean()

        return DreamerBehaviorOutput(
            actor_loss=actor_loss,
            value_loss=value_loss,
            lambda_return=returns,
            imagined_reward=reward,
            imagined_value=value,
            imagined_action=actions,
        )

    @override
    def forward(  # type: ignore[override]
        self, observations: Tensor, actions: Tensor, rewards: Tensor
    ) -> DreamerOutput:
        """Train the world model and the behaviour on one batch of trajectories.

        Parameters
        ----------
        observations : Tensor
            Frames, ``(B, T, C, 64, 64)``.
        actions : Tensor
            Actions taken *into* each step, ``(B, T, action_dim)``.
        rewards : Tensor
            Observed reward at each step, ``(B, T)``.

        Returns
        -------
        DreamerOutput
            ``loss`` is the world-model loss; the behaviour losses are on
            ``.behavior`` and take their own optimisers.
        """
        model = self.dreamer
        priors, posteriors = model.observe(observations, actions)
        reconstruction = model.decode(posteriors)
        predicted_reward = model.predict_reward(posteriors)

        b = int(observations.shape[0])
        t = int(observations.shape[1])
        # Unit-variance Gaussian log-densities, exactly as PlaNet scores
        # them: one-half squared error, summed over pixels and averaged
        # over (B, T).
        diff = (reconstruction - observations) ** 2
        recon_loss = 0.5 * diff.reshape(b, t, -1).sum(dim=-1).mean()
        reward_loss = 0.5 * ((predicted_reward - rewards) ** 2).mean()
        kl_loss = rssm_kl(posteriors, priors, free_nats=self._free_nats)
        loss = recon_loss + reward_loss + self._kl_weight * kl_loss

        return DreamerOutput(
            observation=reconstruction,
            reward=predicted_reward,
            value=model.predict_value(posteriors),
            posterior_stoch=posteriors.stoch,
            posterior_mean=posteriors.mean,
            posterior_std=posteriors.std,
            prior_mean=priors.mean,
            prior_std=priors.std,
            deter=posteriors.deter,
            loss=loss,
            recon_loss=recon_loss,
            reward_loss=reward_loss,
            kl_loss=kl_loss,
            behavior=self._behavior(posteriors),
        )
