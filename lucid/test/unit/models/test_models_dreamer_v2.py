"""Unit tests for DreamerV2 (Hafner et al., 2021).

Three of this family's four changes are invisible to a test that checks
values, which is what makes them worth testing carefully.

**KL balancing** produces the *same number* as an ordinary KL on the same
forward pass — the split is entirely about where the gradient goes. So
the tests measure which head receives it, not what the loss reads.

**The target critic** is a copy that must not move when the critic does.
A test that only asserts "value_loss is finite" passes on a model with no
target at all.

**The gradient modes** all produce a scalar and all train the actor;
telling them apart means checking that they produce *different* gradients
from the same weights.

The truncated normal underneath is checked against its own closed forms —
density integrating to one, entropy against the numerical integral —
because an entropy bonus computed from a wrong entropy still trains, just
toward the wrong thing.
"""

import math

import pytest

import lucid
import lucid.optim as optim
from lucid.models import (
    DreamerV2BehaviorOutput,
    DreamerV2Config,
    DreamerV2ForWorldModeling,
    DreamerV2Model,
    DreamerV2Output,
    create_model,
    is_model,
    list_models,
)
from lucid.models.generative.dreamer_v2._dists import TruncatedNormal
from lucid.models.generative.dreamer_v2._model import balanced_kl


def _tiny_cfg(**overrides: object) -> DreamerV2Config:
    base: dict[str, object] = {
        "action_dim": 2,
        "cnn_depth": 4,
        "stoch_size": 4,
        "discrete": 5,
        "deter_size": 16,
        "hidden_size": 16,
        "actor_hidden": 16,
        "value_hidden": 16,
        "reward_hidden": 16,
        "horizon": 4,
        "pcont": False,
    }
    base.update(overrides)
    return DreamerV2Config(**base)  # type: ignore[arg-type]


def _batch(b: int = 2, t: int = 4) -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor]:
    return lucid.randn((b, t, 3, 64, 64)), lucid.randn((b, t, 2)), lucid.randn((b, t))


class TestDreamerV2Config:
    def test_defaults_are_the_released_ones(self) -> None:
        cfg = DreamerV2Config(action_dim=6)
        assert (cfg.stoch_size, cfg.discrete) == (32, 32)
        assert (cfg.deter_size, cfg.hidden_size, cfg.cnn_depth) == (1024, 1024, 48)
        assert cfg.kl_balance == 0.8 and cfg.free_nats == 0.0
        assert (cfg.actor_layers, cfg.actor_hidden) == (4, 400)
        assert cfg.pcont is True and cfg.actor_grad == "dynamics"

    def test_latent_is_the_flattened_grid(self) -> None:
        cfg = DreamerV2Config(action_dim=6)
        assert cfg.stoch_width == 32 * 32
        assert cfg.latent_size == 1024 + 1024

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"discrete": 1},
            {"discrete": 0},
            {"kl_balance": 1.5},
            {"kl_balance": -0.1},
            {"actor_grad_mix": 1.1},
            {"slow_target_fraction": 0.0},
            {"slow_target_update": 0},
            {"actor_entropy": -1.0},
            {"pcont_scale": -1.0},
        ],
    )
    def test_rejects_bad_config(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            DreamerV2Config(**kwargs)  # type: ignore[arg-type]


class TestTruncatedNormal:
    """Checked against its own closed forms, not against a shape."""

    LOC, SCALE = 0.3, 0.8
    STEPS = 40001

    def _grid(self) -> tuple[lucid.Tensor, lucid.Tensor, TruncatedNormal]:
        dist = TruncatedNormal(lucid.tensor([self.LOC]), lucid.tensor([self.SCALE]))
        xs = lucid.linspace(-1.0, 1.0, self.STEPS)
        return xs, lucid.exp(dist.log_prob(xs)), dist

    @staticmethod
    def _integrate(xs: lucid.Tensor, ys: lucid.Tensor) -> float:
        total = float(ys.sum().item()) - 0.5 * (float(ys[0]) + float(ys[-1]))
        return total * (2.0 / (int(xs.shape[0]) - 1))

    def test_density_integrates_to_one(self) -> None:
        xs, ys, _ = self._grid()
        assert abs(self._integrate(xs, ys) - 1.0) < 1e-3

    def test_mean_matches_the_integral(self) -> None:
        xs, ys, dist = self._grid()
        assert abs(float(dist.mean.item()) - self._integrate(xs, xs * ys)) < 1e-3

    def test_entropy_matches_the_integral(self) -> None:
        """The bonus is paid on this number; a wrong one still trains."""
        xs, ys, dist = self._grid()
        numeric = -self._integrate(xs, ys * dist.log_prob(xs))
        assert abs(float(dist.entropy().item()) - numeric) < 1e-3

    def test_samples_stay_inside(self) -> None:
        dist = TruncatedNormal(lucid.zeros((5000,)), lucid.ones((5000,)) * 3.0)
        drawn = dist.rsample()
        assert bool((drawn.abs() <= 1.0).all().item())

    def test_sampling_is_reparameterised(self) -> None:
        loc = lucid.tensor([0.0], requires_grad=True)
        scale = lucid.tensor([1.0], requires_grad=True)
        TruncatedNormal(loc, scale).rsample().sum().backward()
        assert float(loc.grad.abs().sum().item()) > 0
        assert float(scale.grad.abs().sum().item()) > 0

    def test_entropy_approaches_the_uniform_limit(self) -> None:
        """A very wide truncated normal is uniform on the interval."""
        wide = TruncatedNormal(lucid.tensor([0.0]), lucid.tensor([100.0]))
        assert abs(float(wide.entropy().item()) - math.log(2.0)) < 1e-3

    def test_rejects_an_empty_interval(self) -> None:
        with pytest.raises(ValueError):
            TruncatedNormal(lucid.zeros((1,)), lucid.ones((1,)), low=1.0, high=-1.0)


class TestKLBalancing:
    """The number is the same either way; only the gradient differs."""

    def _heads(self, balance: float, weights: object) -> tuple[float, float, float]:
        model = DreamerV2ForWorldModeling(_tiny_cfg(kl_balance=balance))
        model.load_state_dict(weights)
        out = model(*_batch())
        model.zero_grad()
        out.kl_loss.backward()

        def total(module: object) -> float:
            return sum(
                float(p.grad.abs().sum().item())
                for p in module.parameters()
                if p.grad is not None
            )

        rssm = model.dreamer_v2.rssm
        return (
            float(out.kl_loss.item()),
            total(rssm.prior_head),
            total(rssm.posterior_head),
        )

    def test_all_pressure_on_the_prior(self) -> None:
        weights = DreamerV2ForWorldModeling(_tiny_cfg()).state_dict()
        _, prior, posterior = self._heads(1.0, weights)
        assert prior > 0
        # Not exactly zero: the posterior head still feeds the recurrence
        # that produced the prior's own input. What matters is the ratio.
        assert posterior < prior / 10.0

    def test_all_pressure_on_the_posterior(self) -> None:
        weights = DreamerV2ForWorldModeling(_tiny_cfg()).state_dict()
        _, prior, posterior = self._heads(0.0, weights)
        assert prior == 0.0, "stop-grad on the prior should cut it entirely"
        assert posterior > 0

    def test_the_paper_value_mixes_them(self) -> None:
        weights = DreamerV2ForWorldModeling(_tiny_cfg()).state_dict()
        _, prior, posterior = self._heads(0.8, weights)
        assert prior > 0 and posterior > 0

    def test_value_alone_cannot_tell_them_apart(self) -> None:
        """Why the tests above measure gradients instead.

        Both halves are the same divergence on one forward pass; the
        weights sum to 1, so the reported loss is identical whatever the
        balance is.
        """
        model = DreamerV2Model(_tiny_cfg(mean_only=True))
        priors, posteriors = model.observe(*_batch()[:2])
        values = [
            float(balanced_kl(posteriors, priors, balance=b).item())
            for b in (0.0, 0.5, 1.0)
        ]
        assert max(values) - min(values) < 1e-5

    def test_refuses_a_gaussian_state(self) -> None:
        from lucid.models.generative._rssm import RSSM

        gaussian = RSSM(
            stoch_size=4, deter_size=8, hidden_size=8, action_dim=2, embed_size=6
        ).eval()
        priors, posteriors = gaussian.observe(
            lucid.randn((2, 2, 6)), lucid.randn((2, 2, 2))
        )
        with pytest.raises(ValueError):
            balanced_kl(posteriors, priors, balance=0.8)


class TestTargetCritic:
    def test_target_does_not_move_when_the_critic_does(self) -> None:
        model = DreamerV2ForWorldModeling(_tiny_cfg())
        batch = _batch()
        _, posteriors = model.dreamer_v2.observe(batch[0], batch[1], sample=False)
        before = float(
            model.dreamer_v2.predict_value(posteriors, target=True).mean().item()
        )

        optimiser = optim.Adam(model.value_parameters(), lr=0.5)
        out = model(*batch)
        model.backward(out)
        optimiser.step()

        after = float(
            model.dreamer_v2.predict_value(posteriors, target=True).mean().item()
        )
        live = float(model.dreamer_v2.predict_value(posteriors).mean().item())
        assert abs(after - before) < 1e-9
        assert abs(live - before) > 1e-6, "the critic itself should have moved"

    def test_refresh_copies_the_critic(self) -> None:
        model = DreamerV2ForWorldModeling(_tiny_cfg())
        batch = _batch()
        optimiser = optim.Adam(model.value_parameters(), lr=0.5)
        out = model(*batch)
        model.backward(out)
        optimiser.step()

        model._updates = 0
        model.update_slow_target()
        _, posteriors = model.dreamer_v2.observe(batch[0], batch[1], sample=False)
        live = float(model.dreamer_v2.predict_value(posteriors).mean().item())
        target = float(
            model.dreamer_v2.predict_value(posteriors, target=True).mean().item()
        )
        assert abs(live - target) < 1e-5

    def test_it_moves_only_on_the_schedule(self) -> None:
        model = DreamerV2ForWorldModeling(_tiny_cfg(slow_target_update=3))
        batch = _batch()
        _, posteriors = model.dreamer_v2.observe(batch[0], batch[1], sample=False)
        optimiser = optim.Adam(model.value_parameters(), lr=0.5)

        moved = []
        for _ in range(6):
            out = model(*batch)
            model.backward(out)
            optimiser.step()
            before = float(
                model.dreamer_v2.predict_value(posteriors, target=True).mean().item()
            )
            model.update_slow_target()
            after = float(
                model.dreamer_v2.predict_value(posteriors, target=True).mean().item()
            )
            moved.append(abs(after - before) > 1e-9)

        assert [i for i, m in enumerate(moved) if m] == [0, 3]

    def test_the_target_is_not_in_any_optimiser_group(self) -> None:
        """It is written by the schedule, never descended."""
        model = DreamerV2ForWorldModeling(_tiny_cfg())
        owned = {
            id(p)
            for group in (
                model.world_parameters(),
                model.actor_parameters(),
                model.value_parameters(),
            )
            for p in group
        }
        target = {id(p) for p in model.dreamer_v2.target_value_head.parameters()}
        assert not (owned & target)


class TestActorGradientModes:
    @pytest.mark.parametrize("mode", ["dynamics", "reinforce", "both"])
    def test_each_mode_trains_the_actor(self, mode: str) -> None:
        model = DreamerV2ForWorldModeling(_tiny_cfg(actor_grad=mode))
        out = model(*_batch())
        assert out.behavior is not None
        model.zero_grad()
        out.behavior.actor_loss.backward()
        reached = sum(
            float(p.grad.abs().sum().item())
            for p in model.actor_parameters()
            if p.grad is not None
        )
        assert reached > 0

    def test_the_modes_give_different_gradients(self) -> None:
        """Guards the test above — otherwise the switch could be dead."""
        weights = DreamerV2ForWorldModeling(_tiny_cfg(mean_only=True)).state_dict()
        batch = _batch()
        grads = {}
        for mode in ("dynamics", "reinforce"):
            model = DreamerV2ForWorldModeling(
                _tiny_cfg(actor_grad=mode, mean_only=True)
            )
            model.load_state_dict(weights)
            out = model(*batch)
            model.zero_grad()
            out.behavior.actor_loss.backward()
            grads[mode] = [
                p.grad.clone() for p in model.actor_parameters() if p.grad is not None
            ]
        spread = max(
            float((a - b).abs().max().item())
            for a, b in zip(grads["dynamics"], grads["reinforce"])
        )
        assert spread > 1e-6

    def test_entropy_is_reported(self) -> None:
        model = DreamerV2ForWorldModeling(_tiny_cfg())
        out = model(*_batch())
        assert out.behavior is not None
        assert out.behavior.entropy.ndim == 0

    def test_entropy_bonus_changes_the_loss(self) -> None:
        weights = DreamerV2ForWorldModeling(_tiny_cfg(mean_only=True)).state_dict()
        batch = _batch()
        losses = []
        for coefficient in (0.0, 1.0):
            model = DreamerV2ForWorldModeling(
                _tiny_cfg(actor_entropy=coefficient, mean_only=True)
            )
            model.load_state_dict(weights)
            out = model(*batch)
            assert out.behavior is not None
            losses.append(float(out.behavior.actor_loss.item()))
        assert abs(losses[0] - losses[1]) > 1e-4


class TestForwardAndShapes:
    def test_direct_model_output(self) -> None:
        model = DreamerV2Model(_tiny_cfg())
        out = model(*_batch()[:2])
        assert isinstance(out, DreamerV2Output)
        assert out.posterior_logits.shape == (2, 4, 4, 5)
        assert out.posterior_stoch.shape == (2, 4, 20)
        assert out.observation.shape == (2, 4, 3, 64, 64)
        assert out.behavior is None

    def test_latent_is_one_hot(self) -> None:
        model = DreamerV2Model(_tiny_cfg())
        out = model(*_batch()[:2])
        grid = out.posterior_stoch.reshape(2, 4, 4, 5)
        assert float((grid.sum(dim=-1) - 1.0).abs().max().item()) < 1e-5

    def test_wrapper_populates_every_loss(self) -> None:
        model = DreamerV2ForWorldModeling(_tiny_cfg())
        out = model(*_batch())
        assert isinstance(out.behavior, DreamerV2BehaviorOutput)
        for value in (out.loss, out.recon_loss, out.reward_loss, out.kl_loss):
            assert value is not None and value.ndim == 0
        assert out.pcont_loss is None

    def test_actions_are_bounded(self) -> None:
        model = DreamerV2ForWorldModeling(_tiny_cfg())
        out = model(*_batch())
        assert out.behavior is not None
        assert bool((out.behavior.imagined_action.abs() <= 1.0).all().item())

    def test_pcont_path(self) -> None:
        model = DreamerV2ForWorldModeling(_tiny_cfg(pcont=True))
        obs, actions, rewards = _batch()
        with pytest.raises(ValueError):
            model(obs, actions, rewards)
        out = model(obs, actions, rewards, lucid.ones((2, 4)))
        assert out.pcont_loss is not None
        assert out.behavior is not None
        assert out.behavior.imagined_discount is not None


class TestGradientRouting:
    GROUPS = ("encoder", "rssm", "decoder", "reward_head", "value_head", "actor")

    def _reached(self, loss_name: str) -> set[str]:
        model = DreamerV2ForWorldModeling(_tiny_cfg())
        out = model(*_batch())
        assert out.behavior is not None and out.loss is not None
        loss = out.loss if loss_name == "world" else getattr(out.behavior, loss_name)
        model.zero_grad()
        loss.backward()
        return {
            name
            for name in self.GROUPS
            if any(
                p.grad is not None and float(abs(p.grad).sum().item()) > 0
                for p in getattr(model.dreamer_v2, name).parameters()
            )
        }

    def test_world_loss_reaches_only_the_world_model(self) -> None:
        assert self._reached("world") == {"encoder", "rssm", "decoder", "reward_head"}

    def test_value_loss_reaches_only_the_critic(self) -> None:
        assert self._reached("value_loss") == {"value_head"}

    def test_actor_loss_reaches_the_actor_but_not_the_encoder(self) -> None:
        reached = self._reached("actor_loss")
        assert "actor" in reached
        assert "encoder" not in reached


class TestTrainingStep:
    def test_groups_partition_the_learnable_model(self) -> None:
        model = DreamerV2ForWorldModeling(_tiny_cfg(pcont=True))
        world = {id(p) for p in model.world_parameters()}
        actor = {id(p) for p in model.actor_parameters()}
        value = {id(p) for p in model.value_parameters()}
        target = {id(p) for p in model.dreamer_v2.target_value_head.parameters()}
        everything = {id(p) for p in model.parameters()}
        assert not world & actor and not world & value and not actor & value
        assert world | actor | value | target == everything

    def test_backward_matches_independent_passes(self) -> None:
        model = DreamerV2ForWorldModeling(_tiny_cfg(mean_only=True))
        weights = model.state_dict()
        batch = _batch()
        picks = {
            "world": lambda o: o.loss,
            "actor": lambda o: o.behavior.actor_loss,
            "value": lambda o: o.behavior.value_loss,
        }
        reference = {}
        for name, pick in picks.items():
            other = DreamerV2ForWorldModeling(_tiny_cfg(mean_only=True))
            other.load_state_dict(weights)
            result = other(*batch)
            other.zero_grad()
            pick(result).backward()
            group = getattr(other, f"{name}_parameters")()
            reference[name] = [
                None if p.grad is None else p.grad.clone() for p in group
            ]

        out = model(*batch)
        model.backward(out)
        for name, expected in reference.items():
            group = getattr(model, f"{name}_parameters")()
            for want, param in zip(expected, group):
                if want is None:
                    continue
                # Compared relatively, with a tolerance set from measurement
                # rather than taste. Over eight trials the two paths — one
                # backward alone, one with the graph retained across three —
                # differ by at most 0.9% on these gradients, which are small
                # enough (1e-4) that float32 noise on O(1) intermediates
                # lands there. A mis-routed gradient differs by its whole
                # magnitude, so 5% sits 5x above the noise and 20x below
                # anything real. `test_the_comparison_catches_contamination`
                # holds that second claim up.
                scale = float(want.abs().max().item())
                if scale < 1e-8:
                    continue
                error = float((want - param.grad).abs().max().item())
                assert error / scale < 5e-2, f"{name}: {error / scale:.3e}"

    def test_the_comparison_catches_contamination(self) -> None:
        """The 5% tolerance above is only meaningful if this fails.

        Accumulate the actor's gradient on top of the world model's — the
        exact mistake ``backward`` exists to prevent — and check the same
        relative comparison rejects it by a wide margin.
        """
        model = DreamerV2ForWorldModeling(_tiny_cfg(mean_only=True))
        batch = _batch()

        out = model(*batch)
        model.zero_grad()
        out.loss.backward(retain_graph=True)
        clean = [p.grad.clone() for p in model.world_parameters() if p.grad is not None]
        out.behavior.actor_loss.backward(retain_graph=True)
        mixed = [p.grad for p in model.world_parameters() if p.grad is not None]

        worst = max(
            float((a - b).abs().max().item()) / max(float(a.abs().max().item()), 1e-8)
            for a, b in zip(clean, mixed)
        )
        assert worst > 5e-2, f"contamination only shifted gradients by {worst:.2e}"

    def test_losses_fall_over_a_short_run(self) -> None:
        lucid.manual_seed(0)
        model = DreamerV2ForWorldModeling(_tiny_cfg(horizon=3))
        optimisers = [
            optim.Adam(model.world_parameters(), lr=1e-4),
            optim.Adam(model.value_parameters(), lr=2e-4),
            optim.Adam(model.actor_parameters(), lr=8e-5),
        ]
        obs = lucid.rand((2, 4, 3, 64, 64))
        actions = lucid.rand((2, 4, 2)) * 2 - 1
        rewards = obs.reshape(2, 4, -1).mean(dim=-1) * 10.0

        first = last = None
        for step in range(12):
            out = model(obs, actions, rewards)
            pair = (float(out.recon_loss.item()), float(out.reward_loss.item()))
            first = pair if step == 0 else first
            last = pair
            model.backward(out)
            for opt in optimisers:
                opt.step()
            model.update_slow_target()

        assert last[0] < first[0], "reconstruction did not improve"
        assert last[1] < first[1], "reward prediction did not improve"


class TestRegistry:
    def test_factories_are_registered(self) -> None:
        assert is_model("dreamer_v2") and is_model("dreamer_v2_world_model")
        listed = list_models()
        assert "dreamer_v2" in listed and "dreamer_v2_world_model" in listed

    def test_create_model_applies_overrides(self) -> None:
        model = create_model("dreamer_v2", action_dim=4, discrete=8)
        assert isinstance(model, DreamerV2Model)
        assert model.config.action_dim == 4 and model.config.discrete == 8

    def test_pretrained_is_refused(self) -> None:
        with pytest.raises(Exception):
            create_model("dreamer_v2", pretrained=True)
