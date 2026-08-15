"""Registry factories for Dreamer.

Hafner et al., 2020 report **one** architecture, evaluated across 20
DeepMind Control Suite tasks.  As with PlaNet, the tasks differ only in
action dimensionality and the action-repeat constant — neither is a variant
of the network — so under the project's paper-cited-variants-only rule this
family gets nominal names and no size suffix:

    * ``dreamer``             — world model plus actor and critic.
    * ``dreamer_world_model`` — the same, with all three objectives.

Set ``action_dim`` for your environment at ``create_model`` time.

No parameter count is registered — the paper states no trainable-parameter
total, and the docs site introspects the real figure anyway.
"""

from dataclasses import replace
from typing import Any, cast

from lucid.models._registry import register_model
from lucid.models._utils._common import reject_unavailable_pretrained
from lucid.models.generative.dreamer._config import DreamerConfig
from lucid.models.generative.dreamer._model import (
    DreamerForWorldModeling,
    DreamerModel,
)

# The paper's DeepMind Control Suite setup: PlaNet's world model unchanged,
# with 300-unit ELU heads, a 15-step imagination horizon and TD(0.95)
# returns.  Depths differ per head and are cited individually — the paper
# states three layers for the action and value models and says nothing
# about the reward model, whose two layers come from the released
# implementation (and match PlaNet's).
_CFG_DREAMER = DreamerConfig(
    sample_size=64,
    in_channels=3,
    out_channels=3,
    action_dim=1,
    stoch_size=30,
    deter_size=200,
    hidden_size=200,
    cnn_depth=32,
    min_std=0.1,
    free_nats=3.0,
    horizon=15,
    discount=0.99,
    lambda_=0.95,
    actor_hidden=300,
    actor_layers=3,
    value_hidden=300,
    value_layers=3,
    reward_hidden=300,
    reward_layers=2,
)


def _apply(cfg: DreamerConfig, overrides: dict[str, object]) -> DreamerConfig:
    return replace(cfg, **cast(dict[str, Any], overrides)) if overrides else cfg


# ── Trunk ────────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="dreamer",
    model_type="dreamer",
    model_class=DreamerModel,
    default_config=_CFG_DREAMER,
)
def dreamer(pretrained: bool = False, **overrides: object) -> DreamerModel:
    r"""Construct Dreamer's world model, actor and critic — no objectives.

    PlaNet's architecture unchanged — a four-layer stride-2 convolutional
    encoder, a recurrent state-space model with a 200-unit deterministic
    path and a 30-unit stochastic state, a mirrored decoder — plus the two
    heads that replace PlaNet's planner: a ``tanh``-squashed Gaussian actor
    and a scalar critic, each three 300-unit ELU layers.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises
        rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DreamerConfig` field overrides.  ``action_dim`` is
        the one you almost always need — it is set by the environment, not
        by the paper.

    Returns
    -------
    DreamerModel
        The trunk, configured with the paper defaults and any overrides.

    Notes
    -----
    Reference: Hafner, Lillicrap, Ba, and Norouzi, *"Dream to Control:
    Learning Behaviors by Latent Imagination"*, ICLR, 2020
    (arXiv:1912.01603).

    Examples
    --------
    >>> from lucid.models.generative.dreamer import dreamer
    >>> model = dreamer(action_dim=6).eval()
    >>> model.config.horizon, model.config.lambda_
    (15, 0.95)
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer")
    return DreamerModel(_apply(_CFG_DREAMER, overrides))


# ── World-modeling head ──────────────────────────────────────────────────────


@register_model(
    task="world-modeling",
    family="dreamer",
    model_type="dreamer",
    model_class=DreamerForWorldModeling,
    default_config=_CFG_DREAMER,
)
def dreamer_world_model(
    pretrained: bool = False, **overrides: object
) -> DreamerForWorldModeling:
    r"""Construct Dreamer with all three of its objectives.

    Same trunk as :func:`dreamer`, wrapped with the world-model bound of
    Hafner et al., 2019 and the two behaviour losses of Hafner et al.,
    2020 — an actor maximising :math:`V_\lambda` over imagined
    trajectories and a critic regressing onto it.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerConfig` field overrides.  ``horizon``,
        ``lambda_`` and ``discount`` shape the imagined return;
        ``action_dim`` is set by the environment.

    Returns
    -------
    DreamerForWorldModeling
        The trunk plus the objectives.

    Notes
    -----
    Reference: Hafner, Lillicrap, Ba, and Norouzi, *"Dream to Control:
    Learning Behaviors by Latent Imagination"*, ICLR, 2020
    (arXiv:1912.01603).

    The three losses take **three separate optimisers** over the parameter
    groups the model exposes — see :class:`DreamerForWorldModeling`.

    Examples
    --------
    >>> from lucid.models.generative.dreamer import dreamer_world_model
    >>> model = dreamer_world_model(action_dim=6).eval()
    >>> len(model.actor_parameters()) > 0
    True
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_world_model")
    return DreamerForWorldModeling(_apply(_CFG_DREAMER, overrides))
