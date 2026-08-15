"""Does a world model actually learn to control anything?

Everything else in the suite checks the parts. Shapes are right, gradients
reach the modules they should, losses fall, lambda-returns match the
paper's closed form. None of it would catch a recurrence that carries the
wrong state forward, because a model like that still reconstructs frames
and still reports a shrinking loss — it simply never learns to act.

So this closes the loop: seed a replay buffer with random episodes, train
the three objectives, collect with the policy it is learning, and check
that the return beats what random gets. It is the only test here that can
fail for the right reason.

Cost, measured on an M1 Pro: about 20 seconds. That is why the run is
short and the model is small — the point is the direction of the curve,
not a competitive score.
"""

import pytest

import lucid
import lucid.models as M
import lucid.optim as optim
from lucid.test._fixtures.point_mass import PointMass
from lucid.utils.rollout import LatentPolicy, RandomPolicy, SequenceReplay, rollout

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
