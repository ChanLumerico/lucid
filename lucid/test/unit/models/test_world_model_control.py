"""Does a world model learn to act at all?

Everything else in the suite checks the parts. Shapes are right, gradients
reach the modules they should, losses fall, lambda-returns match the
paper's closed form. None of it would catch a recurrence that carries the
wrong state forward, because a model like that still reconstructs frames
and still reports a shrinking loss.

So this closes the loop: seed a replay buffer, train the three objectives,
collect with the policy being learned, and check the return beats random.

**What this does not establish.** The target here is fixed, so a policy
that ignores its observation entirely and pushes in one direction already
scores 15.78 against random's 5.77 — measured, and pinned below. Passing
this therefore shows the agent found a good *constant* action, not that it
learned to read a state and respond to it. On a variant with the target
moved each episode, where a constant action cannot work, this same setup
peaks at 0.87x the best constant and then degrades: state-dependent
control is not demonstrated here, and whether that is scale or a defect
is open.

The tests are kept because "finds the best constant action" is still far
more than the rest of the suite establishes — a mis-wired recurrence
fails it — but the name of the file should not be read as more than that.

Cost, measured on an M1 Pro: about 45 seconds for both families.
"""

import math

import pytest

import lucid
import lucid.models as M
import lucid.optim as optim
from lucid.test._fixtures.point_mass import PointMass
from lucid.utils.rollout import (
    LatentPolicy,
    RandomPolicy,
    SequenceReplay,
    StepResult,
    rollout,
)

pytestmark = pytest.mark.slow

_TINY = dict(
    action_dim=2,
    cnn_depth=8,
    stoch_size=8,
    deter_size=32,
    hidden_size=32,
    actor_hidden=32,
    value_hidden=32,
    reward_hidden=32,
    horizon=8,
)

TRAIN_STEPS = 120
SEED_EPISODES = 5
BATCH, LENGTH = 8, 15


def _policy(model: object, noise: float = 0.0) -> LatentPolicy:
    return LatentPolicy(
        model.encode,
        model.rssm,
        lambda state: model.act(state, sample=False),
        2,
        noise=noise,
    )


def _average_return(env: PointMass, policy: object, episodes: int = 2) -> float:
    return sum(rollout(env, policy)[1] for _ in range(episodes)) / episodes


class _Greedy:
    """Straight at the target — what a solved task looks like."""

    def __init__(self, env: PointMass) -> None:
        self.env = env

    def reset(self) -> None:
        return None

    def __call__(self, observation: lucid.Tensor) -> lucid.Tensor:
        return self.env.optimal_action()


class TestPointMassEnvironment:
    """Guard the instrument before trusting what it measures."""

    def test_greedy_beats_random_by_a_wide_margin(self) -> None:
        """If it did not, the learning test could not distinguish anything."""
        env = PointMass()
        random_return = _average_return(env, RandomPolicy(2), episodes=5)
        greedy_return = _average_return(env, _Greedy(env), episodes=2)
        assert greedy_return > 2.5 * random_return

    def test_reward_is_dense(self) -> None:
        """A sparse reward would measure luck over a run this short."""
        env = PointMass()
        env.reset()
        far = env.step(lucid.tensor([-1.0, 1.0])).reward
        assert far > 0.05

    def test_observation_shows_agent_and_target_apart(self) -> None:
        env = PointMass()
        observation = env.reset()
        assert observation.shape == (3, 64, 64)
        agent, target = observation[0], observation[1]
        assert float(agent.max().item()) > 0.9
        assert float(target.max().item()) > 0.9
        # Different places, or the task would be solved at t=0.
        assert float((agent * target).max().item()) < 0.5


class TestWhatTheThresholdActuallyRequires:
    """Pins the limitation of the tests below, so it cannot be forgotten.

    A threshold of 1.3x random reads like evidence of control. On a fixed
    target it is not: the numbers here show a constant action clearing it
    outright. Keeping the measurement in the tree is the difference
    between a known limitation and a wrong claim.
    """

    class _Constant:
        def __init__(self, value: list[float]) -> None:
            self.value = lucid.tensor(value)

        def reset(self) -> None:
            return None

        def __call__(self, observation: lucid.Tensor) -> lucid.Tensor:
            return self.value

    def test_a_constant_action_already_clears_the_bar(self) -> None:
        env = PointMass(horizon=20)
        baseline = _average_return(env, RandomPolicy(2), episodes=5)
        best = max(
            _average_return(env, self._Constant(v), episodes=1)
            for v in ([1.0, -1.0], [1.0, 0.0], [0.0, -1.0], [0.7, -0.7])
        )
        assert best > 1.3 * baseline, (
            "if this ever fails the tests below became stronger than "
            "documented, and the module docstring needs revisiting"
        )

    def test_the_optimal_policy_is_still_better(self) -> None:
        """There is headroom above the constant — the task is not degenerate."""
        env = PointMass(horizon=20)
        greedy = _average_return(env, _Greedy(env), episodes=2)
        best_constant = _average_return(env, self._Constant([1.0, -1.0]), episodes=1)
        assert greedy > best_constant


class TestDreamerLearnsControl:
    def test_return_improves_over_random(self) -> None:
        lucid.manual_seed(0)
        env = PointMass(horizon=20)
        model = M.create_model("dreamer_world_model", **_TINY)
        optimisers = [
            optim.Adam(model.world_parameters(), lr=6e-4),
            optim.Adam(model.value_parameters(), lr=8e-5),
            optim.Adam(model.actor_parameters(), lr=8e-5),
        ]

        replay = SequenceReplay(capacity=20_000)
        for _ in range(SEED_EPISODES):
            replay.add(rollout(env, RandomPolicy(2))[0])
        baseline = _average_return(env, RandomPolicy(2), episodes=5)

        collector = _policy(model.dreamer, noise=0.3)
        evaluator = _policy(model.dreamer)

        curve = []
        for step in range(TRAIN_STEPS):
            batch = replay.sample(BATCH, LENGTH)
            out = model(batch.observations, batch.actions, batch.rewards)
            model.backward(out)
            for opt in optimisers:
                opt.step()

            if (step + 1) % 40 == 0:
                replay.add(rollout(env, collector)[0])
                curve.append(_average_return(env, evaluator))

        best = max(curve)
        assert best > 1.3 * baseline, (
            f"policy did not learn: best {best:.2f} vs random {baseline:.2f}, "
            f"curve {[round(c, 2) for c in curve]}"
        )
        assert curve[-1] > curve[0], f"return did not trend upward: {curve}"

    def test_collected_episodes_feed_back_into_the_buffer(self) -> None:
        """The loop is closed — otherwise it is just offline training."""
        lucid.manual_seed(0)
        env = PointMass(horizon=10)
        model = M.create_model("dreamer_world_model", **_TINY)
        replay = SequenceReplay()
        for _ in range(2):
            replay.add(rollout(env, RandomPolicy(2))[0])
        before = replay.steps
        replay.add(rollout(env, _policy(model.dreamer, noise=0.3))[0])
        assert replay.steps > before

    def test_exploration_noise_changes_what_is_collected(self) -> None:
        lucid.manual_seed(0)
        env = PointMass(horizon=10)
        model = M.create_model("dreamer", **_TINY).eval()
        quiet = rollout(env, _policy(model))[0]
        loud = rollout(env, _policy(model, noise=0.5))[0]
        assert not bool((quiet.actions == loud.actions).all().item())


class _Cliff:
    """Terminates rather than truncating, so ``discounts`` reaches zero."""

    def __init__(self, at: int = 5) -> None:
        self.at = at

    def reset(self) -> lucid.Tensor:
        self.t = 0
        return lucid.rand((3, 64, 64))

    def step(self, action: lucid.Tensor) -> StepResult:
        self.t += 1
        return StepResult(lucid.rand((3, 64, 64)), 1.0, self.t >= self.at, False)


class TestDiscountHeadEndToEnd:
    """Termination has to survive the whole path, not just each end of it.

    ``pcont`` is tested against hand-written discounts elsewhere, and the
    driver's termination flag is tested without a model. This is the seam
    between them: an environment that really ends, through a real rollout,
    into a real buffer, into the loss.
    """

    def test_termination_reaches_the_loss(self) -> None:
        lucid.manual_seed(0)
        model = M.create_model("dreamer_world_model", pcont=True, **_TINY)
        replay = SequenceReplay()
        for _ in range(4):
            replay.add(rollout(_Cliff(5), RandomPolicy(2))[0])

        batch = replay.sample(4, 5)
        assert float(batch.discounts[0][-1]) == 0.0, "termination lost in the buffer"

        out = model(batch.observations, batch.actions, batch.rewards, batch.discounts)
        assert out.pcont_loss is not None
        assert out.behavior is not None and out.behavior.imagined_pcont is not None

        model.backward(out)
        grads = [
            p.grad for p in model.dreamer.pcont_head.parameters() if p.grad is not None
        ]
        assert grads and sum(float(g.abs().sum().item()) for g in grads) > 0

    def test_a_truncating_environment_leaves_the_discount_alone(self) -> None:
        """PointMass only ever truncates; nothing should read as terminal."""
        replay = SequenceReplay()
        replay.add(rollout(PointMass(horizon=8), RandomPolicy(2))[0])
        batch = replay.sample(2, 8)
        assert bool((batch.discounts == 1.0).all().item())


_TINY_V2 = dict(
    action_dim=2,
    cnn_depth=4,
    stoch_size=6,
    discrete=6,
    deter_size=24,
    hidden_size=24,
    actor_hidden=32,
    value_hidden=32,
    reward_hidden=32,
    horizon=6,
    pcont=False,
)


def _v2_policy(model: object, noise: float = 0.0) -> LatentPolicy:
    return LatentPolicy(
        model.encode,
        model.rssm,
        lambda state: model.act(state, sample=False),
        2,
        noise=noise,
    )


class TestDreamerV2LearnsControl:
    """The same question asked of the discrete latent.

    Worth asking separately rather than trusting the shared parts: the
    latent, the divergence, the critic's target and the policy's
    distribution are all different, and every one of them can be wired in
    a way that trains happily and controls nothing.
    """

    def test_return_improves_over_random(self) -> None:
        lucid.manual_seed(0)
        env = PointMass(horizon=20)
        model = M.create_model("dreamer_v2_world_model", **_TINY_V2)
        optimisers = [
            optim.Adam(model.world_parameters(), lr=6e-4),
            optim.Adam(model.value_parameters(), lr=2e-4),
            # 8e-4, not the released 8e-5. DreamerV2's policy starts at
            # 99% of the maximum entropy the interval allows — std is
            # 2*sigmoid(0) + min_std ~= 1.1, which on [-1, 1] is nearly
            # uniform, so the mean barely moves the sample. Sharpening it
            # is most of the learning, and the published rate is set for
            # millions of steps rather than the sixty here. Measured: at
            # 8e-5 the policy is still uniform after 120 steps and scores
            # 0.70x random; at 8e-4 it reaches 2.9x by step 50.
            optim.Adam(model.actor_parameters(), lr=8e-4),
        ]

        replay = SequenceReplay(capacity=20_000)
        for _ in range(SEED_EPISODES):
            replay.add(rollout(env, RandomPolicy(2))[0])
        baseline = _average_return(env, RandomPolicy(2), episodes=5)

        inner = model.dreamer_v2
        collector = _v2_policy(inner, noise=0.3)
        evaluator = _v2_policy(inner)

        curve = []
        entropy_first = entropy_last = None
        for step in range(60):
            batch = replay.sample(BATCH, LENGTH)
            out = model(batch.observations, batch.actions, batch.rewards)
            assert out.behavior is not None
            entropy = float(out.behavior.entropy.item())
            entropy_first = entropy if step == 0 else entropy_first
            entropy_last = entropy
            model.backward(out)
            for opt in optimisers:
                opt.step()
            model.update_slow_target()
            if (step + 1) % 30 == 0:
                replay.add(rollout(env, collector)[0])
                curve.append(_average_return(env, evaluator))

        best = max(curve)
        assert best > 1.3 * baseline, (
            f"policy did not learn: best {best:.2f} vs random {baseline:.2f}, "
            f"curve {[round(c, 2) for c in curve]}"
        )
        # The mechanism, on the same run rather than a second one. A
        # policy that never leaves the uniform limit scores exactly
        # random however good the world model gets — which is what the
        # released actor rate produces over sixty steps.
        assert entropy_last < entropy_first - 0.04, (
            f"policy never sharpened: {entropy_first:.3f} -> {entropy_last:.3f}"
        )

    def test_it_starts_near_the_uniform_limit(self) -> None:
        """Pins the starting point the test above measures movement from."""
        model = M.create_model("dreamer_v2", **_TINY_V2).eval()
        feature = lucid.randn((4, 3, model.config.latent_size))
        entropy = float(model.actor.distribution(feature).entropy().mean().item())
        assert entropy > 0.9 * math.log(2.0)
