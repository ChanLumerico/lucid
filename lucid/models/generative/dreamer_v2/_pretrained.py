"""Registry factories for DreamerV2.

The paper tunes two settings — Atari and DeepMind Control — but they are
task configurations rather than architecture variants, so under the
project's paper-cited-variants-only rule this family gets nominal names
and the two settings are tabulated in :class:`DreamerV2Config`'s Notes:

    * ``dreamer_v2``             — world model, actor and critic.
    * ``dreamer_v2_world_model`` — the same, with all three objectives.

Defaults are the released implementation's ``defaults`` block.  Set
``action_dim`` for your environment at ``create_model`` time.

No parameter count is registered — the paper reports 20M for the world
model without breaking it down, and the docs site introspects the real
figure anyway.
"""

from dataclasses import replace
from typing import Any, cast

from lucid.models._registry import register_model
from lucid.models._utils._common import reject_unavailable_pretrained
from lucid.models.generative.dreamer_v2._config import DreamerV2Config
from lucid.models.generative.dreamer_v2._model import (
    DreamerV2ForWorldModeling,
    DreamerV2Model,
)

_CFG_DREAMER_V2 = DreamerV2Config()


def _apply(cfg: DreamerV2Config, overrides: dict[str, object]) -> DreamerV2Config:
    return replace(cfg, **cast(dict[str, Any], overrides)) if overrides else cfg


@register_model(
    task="base",
    family="dreamer_v2",
    model_type="dreamer_v2",
    model_class=DreamerV2Model,
    default_config=_CFG_DREAMER_V2,
)
def dreamer_v2(pretrained: bool = False, **overrides: object) -> DreamerV2Model:
    r"""Construct DreamerV2's world model, actor and critic — no objectives.

    Dreamer's architecture with a categorical latent: 32 variables of 32
    classes, sampled as one-hots and carried as a 1024-wide sparse binary
    vector.  Four-layer 400-unit ELU heads for reward, value and policy,
    plus a frozen copy of the critic to compute returns against.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerV2Config` field overrides.  ``action_dim``
        is set by the environment, not by the paper.

    Returns
    -------
    DreamerV2Model
        The trunk, configured with the released defaults.

    Notes
    -----
    Reference: Hafner, Lillicrap, Norouzi, and Ba, *"Mastering Atari with
    Discrete World Models"*, ICLR, 2021 (arXiv:2010.02193).

    Examples
    --------
    >>> from lucid.models import dreamer_v2
    >>> model = dreamer_v2(action_dim=6).eval()
    >>> model.config.stoch_size, model.config.discrete
    (32, 32)
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_v2")
    return DreamerV2Model(_apply(_CFG_DREAMER_V2, overrides))


@register_model(
    task="world-modeling",
    family="dreamer_v2",
    model_type="dreamer_v2",
    model_class=DreamerV2ForWorldModeling,
    default_config=_CFG_DREAMER_V2,
)
def dreamer_v2_world_model(
    pretrained: bool = False, **overrides: object
) -> DreamerV2ForWorldModeling:
    r"""Construct DreamerV2 with all three of its objectives.

    Same trunk as :func:`dreamer_v2`, wrapped with the world-model bound
    under a **balanced** divergence, an actor maximising imagined
    :math:`V_\lambda` with an entropy bonus, and a critic regressing onto
    a target copy of itself.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerV2Config` field overrides.  ``kl_balance``
        and ``actor_grad`` are the two that change the algorithm rather
        than its size.

    Returns
    -------
    DreamerV2ForWorldModeling
        The trunk plus the objectives.

    Notes
    -----
    Reference: Hafner, Lillicrap, Norouzi, and Ba, *"Mastering Atari with
    Discrete World Models"*, ICLR, 2021 (arXiv:2010.02193).

    The three losses take **three separate optimisers**; use
    :meth:`~lucid.models.DreamerV2ForWorldModeling.backward` to fill their
    gradients, and call ``update_slow_target`` once per step.

    Examples
    --------
    >>> from lucid.models import dreamer_v2_world_model
    >>> model = dreamer_v2_world_model(action_dim=6).eval()
    >>> model.config.kl_balance, model.config.actor_grad
    (0.8, 'dynamics')
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_v2_world_model")
    return DreamerV2ForWorldModeling(_apply(_CFG_DREAMER_V2, overrides))
