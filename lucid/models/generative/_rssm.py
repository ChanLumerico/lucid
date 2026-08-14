"""Recurrent State-Space Model — the shared trunk of the world-model families.

Introduced by Hafner, Lillicrap, Fischer, Villegas, Ha, Lee & Davidson,
*"Learning Latent Dynamics for Planning from Pixels"* (ICML, 2019) as the
dynamics model inside PlaNet, and reused with modifications by the Dreamer
line.  It lives here rather than inside a family for the same reason
``_schedulers.py`` does: it is the substrate several families share, not
something any one of them owns.

The state is deliberately **split in two**.  A purely stochastic latent
cannot reliably remember information across many steps — each step's noise
erodes it — while a purely deterministic one cannot represent the several
futures a partially-observed environment allows.  So each step carries

    deter  — a GRU hidden state, the path information survives along
    stoch  — a Gaussian sample, the path uncertainty enters through

and the transition conditions on both.  That split is the paper's central
architectural claim, and it is why the ablations against a pure RNN and a
pure state-space model both underperform it.

Two distributions are produced at every step:

    prior      p(s_t | h_t)          — what the dynamics predict
    posterior  q(s_t | h_t, e_t)     — what the dynamics predict *after*
                                       seeing the encoded observation

Training pulls the prior toward the posterior; planning then runs the prior
alone, which is what :meth:`RSSM.imagine` does.  Reconstruction reads off
the posterior, which is what :meth:`RSSM.observe` returns.

Layout convention: every public method takes and returns ``(B, T, ...)``,
matching the rest of the zoo.  The unroll itself is time-major internally
because a recurrence has to be — the same shape ``nn.GRU.forward`` takes.
"""

from typing import NamedTuple, cast, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.models._utils._generative import generative_activation, normal_kl

__all__ = ["RSSMState", "RSSM", "rssm_kl"]


class RSSMState(NamedTuple):
    r"""One step (or one unrolled sequence) of recurrent state-space state.

    Attributes
    ----------
    deter : Tensor
        Deterministic path :math:`h`, shaped ``(B, deter_size)`` for a
        single step or ``(B, T, deter_size)`` after an unroll.
    stoch : Tensor
        Sampled stochastic latent :math:`s`, drawn with the
        reparameterisation trick so gradients flow through it.
    mean : Tensor
        Mean of the Gaussian ``stoch`` was drawn from.
    std : Tensor
        Standard deviation of that Gaussian.  Always positive — the
        network emits an unconstrained real which is passed through
        ``softplus`` and floored at ``min_std``.

    Notes
    -----
    ``mean`` and ``std`` are carried alongside the sample because the
    training objective needs the *distributions*, not just draws: the KL
    between prior and posterior is computed in closed form from these.
    """

    deter: Tensor
    stoch: Tensor
    mean: Tensor
    std: Tensor

    @property
    def feature(self) -> Tensor:
        """``[h; s]`` — what a decoder or a reward head reads off the state."""
        return lucid.cat([self.deter, self.stoch], dim=-1)

    @property
    def logvar(self) -> Tensor:
        r""":math:`\log \sigma^2`, for the helpers that parameterise that way.

        Safe without an epsilon: ``std`` is floored at ``min_std`` by
        construction, so the logarithm never approaches its pole.
        """
        return 2.0 * self.std.log()


class RSSM(nn.Module):
    r"""Recurrent state-space model (Hafner et al., 2019).

    One step of the recurrence is

    .. math::

        h_t = f\big(h_{t-1},\, [s_{t-1};\, a_{t-1}]\big),

    a GRU update whose input is the previous stochastic latent concatenated
    with the action, followed by two heads reading off that deterministic
    path:

    .. math::

        p(s_t \mid h_t) = \mathcal{N}\big(\mu_p(h_t),\, \sigma_p(h_t)\big),
        \qquad
        q(s_t \mid h_t, e_t)
            = \mathcal{N}\big(\mu_q(h_t, e_t),\, \sigma_q(h_t, e_t)\big),

    where :math:`e_t` is the encoded observation.  The prior is the model's
    guess before it looks; the posterior is its belief after.

    Parameters
    ----------
    stoch_size : int
        Width of the stochastic latent :math:`s`.
    deter_size : int
        Width of the GRU hidden state :math:`h`.
    hidden_size : int
        Width of the hidden layer inside each head and inside the
        pre-recurrence projection.
    action_dim : int
        Width of the action vector conditioning the transition.
    embed_size : int
        Width of the encoded observation the posterior head consumes.
    act_fn : {"silu", "swish", "relu", "gelu"}, default="relu"
        Activation used by the projections.  The paper uses ReLU.
    min_std : float, default=0.1
        Floor added after ``softplus``.  Without it a head can drive the
        standard deviation to zero, at which point the KL diverges and the
        sample stops carrying gradient.

    Attributes
    ----------
    cell : nn.GRUCell
        The deterministic recurrence.
    prior_head : nn.Module
        Maps :math:`h_t` to :math:`(\mu_p, \sigma_p)`.
    posterior_head : nn.Module
        Maps :math:`[h_t; e_t]` to :math:`(\mu_q, \sigma_q)`.

    Notes
    -----
    Reference: Hafner, Lillicrap, Fischer, Villegas, Ha, Lee, and Davidson,
    *"Learning Latent Dynamics for Planning from Pixels"*, ICML, 2019
    (arXiv:1811.04551).

    The Gaussian is parameterised by ``(mean, std)`` because that is what
    ``softplus(raw) + min_std`` produces; :attr:`RSSMState.logvar` converts
    for the helpers that want the other form.

    The divergence goes through :func:`normal_kl`, which is elementwise
    between two *free* Gaussians and applies no reduction — leaving free
    nats to :func:`rssm_kl`.  :func:`gaussian_kl_divergence` in the same
    module is the wrong tool here: it measures against a fixed
    :math:`\mathcal{N}(0, I)`, and the entire point of the prior head is
    that the reference is learned.  ``lucid.distributions.kl_divergence``
    would serve equally (the two agree to ~4e-06), but no other module
    under ``lucid/models/`` imports ``lucid.distributions``, and one
    family is not a reason to open that direction.

    **Action alignment.**  ``actions[:, t]`` is the action taken *into*
    step ``t`` — :math:`a_{t-1}` in the paper's indexing, the one that
    produced the observation at ``t``.  A caller holding a trajectory
    :math:`(o_0, a_0, o_1, a_1, \dots)` must therefore shift by one and
    pass a zero action at index 0.  Nothing can enforce this: a
    mis-aligned caller gets a model that trains to a worse optimum, never
    an error.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative._rssm import RSSM
    >>> rssm = RSSM(stoch_size=4, deter_size=8, hidden_size=8,
    ...             action_dim=2, embed_size=6).eval()
    >>> embed = lucid.randn((2, 5, 6))
    >>> actions = lucid.randn((2, 5, 2))
    >>> priors, posteriors = rssm.observe(embed, actions)
    >>> posteriors.stoch.shape, posteriors.deter.shape
    ((2, 5, 4), (2, 5, 8))
    >>> rssm.imagine(rssm.initial(2), actions).stoch.shape
    (2, 5, 4)
    """

    def __init__(
        self,
        stoch_size: int,
        deter_size: int,
        hidden_size: int,
        action_dim: int,
        embed_size: int,
        act_fn: str = "relu",
        min_std: float = 0.1,
    ) -> None:
        """Initialise the RSSM. See the class docstring for parameter semantics."""
        super().__init__()
        for name, value in (
            ("stoch_size", stoch_size),
            ("deter_size", deter_size),
            ("hidden_size", hidden_size),
            ("embed_size", embed_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if min_std <= 0.0:
            raise ValueError(f"min_std must be positive, got {min_std}")

        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.action_dim = action_dim
        self.embed_size = embed_size
        self.min_std = min_std
        self._act_name = act_fn

        self.pre_cell = nn.Linear(stoch_size + action_dim, hidden_size)
        self.cell = nn.GRUCell(hidden_size, deter_size)

        self.prior_head = nn.Sequential(
            nn.Linear(deter_size, hidden_size),
            nn.Linear(hidden_size, 2 * stoch_size),
        )
        self.posterior_head = nn.Sequential(
            nn.Linear(deter_size + embed_size, hidden_size),
            nn.Linear(hidden_size, 2 * stoch_size),
        )

    # ── internals ────────────────────────────────────────────────────────

    def _head(self, head: nn.Module, x: Tensor) -> tuple[Tensor, Tensor]:
        """Run a two-layer head and split its output into ``(mean, std)``."""
        first, second = cast(nn.Sequential, head)[0], cast(nn.Sequential, head)[1]
        h = generative_activation(self._act_name, cast(Tensor, first(x)))
        out = cast(Tensor, second(h))
        mean = out[..., : self.stoch_size]
        std = nn.functional.softplus(out[..., self.stoch_size :]) + self.min_std
        return mean, std

    @staticmethod
    def _draw(mean: Tensor, std: Tensor, sample: bool) -> Tensor:
        """Reparameterised draw — gradient flows through ``mean`` and ``std``.

        With ``sample=False`` the mean is returned instead, which makes the
        whole recurrence deterministic.  Training wants the draw; planning
        wants the mean, because a search that re-samples every candidate is
        ranking its own noise rather than the actions.
        """
        if not sample:
            return mean
        noise = lucid.randn(
            tuple(int(s) for s in mean.shape), device=mean.device, dtype=mean.dtype
        )
        return mean + std * noise

    # ── single steps ─────────────────────────────────────────────────────

    def initial(self, batch_size: int, device: str = "cpu") -> RSSMState:
        """Return the all-zero state the unroll starts from."""
        deter = lucid.zeros(batch_size, self.deter_size, device=device)
        stoch = lucid.zeros(batch_size, self.stoch_size, device=device)
        return RSSMState(deter=deter, stoch=stoch, mean=stoch, std=stoch)

    def prior_step(
        self, state: RSSMState, action: Tensor, *, sample: bool = True
    ) -> RSSMState:
        r"""Advance one step without an observation — :math:`p(s_t \mid h_t)`.

        Parameters
        ----------
        state : RSSMState
            The previous step's state, ``(B, ·)``.
        action : Tensor
            The action taken into this step, ``(B, action_dim)``.
        sample : bool, default=True
            Draw the latent, or take its mean when ``False``, which makes
            the step deterministic.

        Returns
        -------
        RSSMState
            The predicted state, ``(B, ·)``.
        """
        x = lucid.cat([state.stoch, action], dim=-1)
        x = generative_activation(self._act_name, cast(Tensor, self.pre_cell(x)))
        deter = cast(Tensor, self.cell(x, state.deter))
        mean, std = self._head(self.prior_head, deter)
        return RSSMState(
            deter=deter, stoch=self._draw(mean, std, sample), mean=mean, std=std
        )

    def posterior_step(
        self,
        state: RSSMState,
        action: Tensor,
        embed: Tensor,
        *,
        sample: bool = True,
    ) -> tuple[RSSMState, RSSMState]:
        r"""Advance one step *with* an observation.

        Parameters
        ----------
        state : RSSMState
            The previous step's state, ``(B, ·)``.
        action : Tensor
            The action taken into this step, ``(B, action_dim)``.
        embed : Tensor
            The encoded observation at this step, ``(B, embed_size)``.
        sample : bool, default=True
            Draw both latents, or take their means when ``False``.

        Returns
        -------
        prior : RSSMState
            :math:`p(s_t \mid h_t)` — the prediction made before looking.
        posterior : RSSMState
            :math:`q(s_t \mid h_t, e_t)` — the belief after looking.

        Notes
        -----
        Both share the same ``deter``: the observation refines the belief
        about :math:`s_t`, it does not change the path that led there.
        The pair is returned together because the KL term needs both, and
        computing them apart would mean running the recurrence twice.
        """
        prior = self.prior_step(state, action, sample=sample)
        x = lucid.cat([prior.deter, embed], dim=-1)
        mean, std = self._head(self.posterior_head, x)
        posterior = RSSMState(
            deter=prior.deter, stoch=self._draw(mean, std, sample), mean=mean, std=std
        )
        return prior, posterior

    # ── unrolls ──────────────────────────────────────────────────────────

    @staticmethod
    def _stack(states: list[RSSMState]) -> RSSMState:
        """Stack per-step states into ``(B, T, ...)``."""
        return RSSMState(
            deter=lucid.stack([s.deter for s in states], dim=1),
            stoch=lucid.stack([s.stoch for s in states], dim=1),
            mean=lucid.stack([s.mean for s in states], dim=1),
            std=lucid.stack([s.std for s in states], dim=1),
        )

    def observe(
        self,
        embed: Tensor,
        actions: Tensor,
        state: RSSMState | None = None,
        *,
        sample: bool = True,
    ) -> tuple[RSSMState, RSSMState]:
        r"""Filter a sequence of observations into posterior states.

        Parameters
        ----------
        embed : Tensor
            Encoded observations, ``(B, T, embed_size)``.
        actions : Tensor
            Actions taken *into* each step, ``(B, T, action_dim)`` — see
            the class docstring on action alignment.  ``actions[:, t]`` is
            the action that produced ``embed[:, t]``, so index 0 is
            normally a zero action.
        state : RSSMState or None, optional
            Starting state; ``None`` starts from :meth:`initial`.
        sample : bool, default=True
            Draw each latent, or take its mean when ``False``, which makes
            the whole filter deterministic.

        Returns
        -------
        priors : RSSMState
            The dynamics' predictions, ``(B, T, ...)``.
        posteriors : RSSMState
            The beliefs after seeing each observation, ``(B, T, ...)``.
        """
        if embed.ndim != 3 or actions.ndim != 3:
            raise ValueError(
                f"observe expects (B, T, ·) inputs, got embed {embed.shape} "
                f"and actions {actions.shape}"
            )
        if int(embed.shape[1]) != int(actions.shape[1]):
            raise ValueError(
                f"embed and actions must agree on T, got {int(embed.shape[1])} "
                f"and {int(actions.shape[1])}"
            )

        if state is None:
            state = self.initial(int(embed.shape[0]), device=embed.device.type)
        priors: list[RSSMState] = []
        posteriors: list[RSSMState] = []
        for t in range(int(embed.shape[1])):
            prior, state = self.posterior_step(
                state, actions[:, t], embed[:, t], sample=sample
            )
            priors.append(prior)
            posteriors.append(state)
        return self._stack(priors), self._stack(posteriors)

    def imagine(
        self, state: RSSMState, actions: Tensor, *, sample: bool = True
    ) -> RSSMState:
        r"""Roll the prior forward with no observations at all.

        This is the model dreaming: every step's latent comes from the
        dynamics' own prediction rather than from an encoded frame, so the
        trajectory is a function of the parameters and the actions only.

        Parameters
        ----------
        state : RSSMState
            Starting state — typically one step of a posterior produced by
            :meth:`observe`.
        actions : Tensor
            Actions to imagine taking, ``(B, T, action_dim)``.
        sample : bool, default=True
            Draw each latent, or take its mean when ``False``.  Planning
            wants the mean: a search that re-samples every candidate ranks
            its own noise instead of the actions.

        Returns
        -------
        RSSMState
            The imagined prior states, ``(B, T, ...)``.
        """
        if actions.ndim != 3:
            raise ValueError(f"imagine expects (B, T, ·) actions, got {actions.shape}")
        states: list[RSSMState] = []
        for t in range(int(actions.shape[1])):
            state = self.prior_step(state, actions[:, t], sample=sample)
            states.append(state)
        return self._stack(states)

    @override
    def forward(  # type: ignore[override]
        self,
        embed: Tensor,
        actions: Tensor,
        state: RSSMState | None = None,
    ) -> tuple[RSSMState, RSSMState]:
        """Alias for :meth:`observe`, so the module is callable."""
        return self.observe(embed, actions, state)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"stoch_size={self.stoch_size}, deter_size={self.deter_size}, "
            f"action_dim={self.action_dim}, embed_size={self.embed_size}"
        )


def rssm_kl(
    posterior: RSSMState, prior: RSSMState, *, free_nats: float = 3.0
) -> Tensor:
    r"""Scalar :math:`\mathrm{KL}(q \,\|\, p)` with a free-nats floor.

    The divergence is summed over the latent dimension, averaged over batch
    and time, and then clipped from below:

    .. math::

        \mathcal{L}_{\mathrm{KL}} =
            \max\big(0,\;
                \mathbb{E}\,\mathrm{KL}\big(q(s \mid h, o)
                    \,\|\, p(s \mid h)\big) - \texttt{free\_nats}\big).

    Parameters
    ----------
    posterior : RSSMState
        The belief that has seen the observation — the first argument of
        the divergence.
    prior : RSSMState
        The dynamics' own prediction.
    free_nats : float, default=3.0
        Divergence below this many nats costs nothing.

    Returns
    -------
    Tensor
        A 0-d tensor.

    Notes
    -----
    Without the floor the posterior collapses onto the prior: the cheapest
    way to drive the divergence to zero is to stop encoding the
    observation, and then the latent carries nothing. The threshold buys
    the model that much divergence for free, so gradient is only spent
    once the two distributions genuinely disagree.

    Two forms of this appear in the literature and they are **not**
    interchangeable to read, only to differentiate: PlaNet clips the
    excess, :math:`\max(0, \mathrm{KL} - \texttt{free\_nats})`, while the
    Dreamer line clips the value, :math:`\max(\mathrm{KL},
    \texttt{free\_nats})`. Their gradients are identical — zero below the
    threshold, one above — but the reported numbers differ by a constant.
    This is the PlaNet form, so a value printed here is comparable with the
    PlaNet paper and sits ``free_nats`` below a Dreamer-style report.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative._rssm import RSSM, rssm_kl
    >>> rssm = RSSM(stoch_size=4, deter_size=8, hidden_size=8,
    ...             action_dim=2, embed_size=6).eval()
    >>> priors, posteriors = rssm.observe(lucid.randn((2, 3, 6)),
    ...                                   lucid.randn((2, 3, 2)))
    >>> float(rssm_kl(posteriors, priors, free_nats=1e9).item())
    0.0
    """
    if free_nats < 0.0:
        raise ValueError(f"free_nats must be non-negative, got {free_nats}")
    per_step = normal_kl(
        posterior.mean, posterior.logvar, prior.mean, prior.logvar
    ).sum(dim=-1)
    kl = per_step.mean()
    return (kl - free_nats).clip(0.0, None) if free_nats > 0.0 else kl
