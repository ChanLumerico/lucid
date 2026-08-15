"""Episode storage with contiguous-chunk sampling.

The distinguishing requirement is in the sampling, not the storage.  A
recurrent state-space model learns from *sequences*: it is trained to
predict what follows from what came before, so a batch of loose
transitions gives it nothing to be recurrent about.  What it needs is
contiguous windows that do not straddle an episode boundary, which is
what this draws.
"""

import lucid
from lucid.utils.rollout._env import Episode

__all__ = ["SequenceReplay"]


class SequenceReplay:
    r"""Whole episodes in, contiguous chunks out.

    Parameters
    ----------
    capacity : int, default=1_000_000
        Budget in *transitions*, not episodes.  Frames dominate the
        memory of a pixel-based agent — at 64x64x3 in float32 a single
        observation is 48 KiB, so a million of them is not a number to
        set carelessly — and counting episodes would let the true cost
        drift with their length.  Oldest episodes are dropped first.

    Attributes
    ----------
    episodes : list of Episode
        Stored trajectories, oldest first.

    Notes
    -----
    Episodes shorter than the requested chunk length are skipped rather
    than padded.  Padding would put frames into the sequence that the
    dynamics never produced, and the model has no way to tell them from
    real ones.

    Sampling picks an episode uniformly and then a start position within
    it uniformly, which is what the world-model papers' released code
    does.  Note the consequence: a step in a short episode is more likely
    to be drawn than a step in a long one.  Sampling uniformly over all
    valid start positions instead would remove that bias, and is a
    deliberate non-choice here — matching the reference matters more than
    correcting it.

    Examples
    --------
    >>> import lucid
    >>> from lucid.utils.rollout import Episode, SequenceReplay
    >>> replay = SequenceReplay()
    >>> replay.add(Episode(lucid.zeros((20, 3, 8, 8)), lucid.zeros((20, 2)),
    ...                    lucid.zeros((20,)), lucid.ones((20,))))
    >>> batch = replay.sample(4, 5)
    >>> batch.observations.shape, batch.actions.shape
    ((4, 5, 3, 8, 8), (4, 5, 2))
    """

    def __init__(self, capacity: int = 1_000_000) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        self.capacity = capacity
        self.episodes: list[Episode] = []
        self._steps = 0

    def __len__(self) -> int:
        """Number of stored episodes."""
        return len(self.episodes)

    @property
    def steps(self) -> int:
        """Total transitions held."""
        return self._steps

    def add(self, episode: Episode) -> None:
        """Store an episode, evicting the oldest until the budget is met.

        Parameters
        ----------
        episode : Episode
            The trajectory to store.

        Raises
        ------
        ValueError
            If the episode's fields disagree on length, or it is empty.
        """
        length = len(episode)
        if length == 0:
            raise ValueError("cannot store an empty episode")
        for name, field in (
            ("actions", episode.actions),
            ("rewards", episode.rewards),
            ("discounts", episode.discounts),
        ):
            if int(field.shape[0]) != length:
                raise ValueError(
                    f"{name} has {int(field.shape[0])} steps but observations "
                    f"has {length}"
                )

        self.episodes.append(episode)
        self._steps += length
        while self._steps > self.capacity and len(self.episodes) > 1:
            self._steps -= len(self.episodes.pop(0))

    def sample(self, batch_size: int, length: int) -> Episode:
        r"""Draw contiguous chunks.

        Parameters
        ----------
        batch_size : int
            Number of chunks.
        length : int
            Steps per chunk.  Each one is a window into a single episode.

        Returns
        -------
        Episode
            Batched, with a leading batch axis — ``observations`` is
            ``(batch_size, length, C, H, W)`` and the rest follow.  It is
            the same NamedTuple because the fields mean the same thing;
            only the rank differs.

        Raises
        ------
        ValueError
            If no stored episode is at least ``length`` steps long.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if length < 1:
            raise ValueError(f"length must be at least 1, got {length}")

        usable = [e for e in self.episodes if len(e) >= length]
        if not usable:
            longest = max((len(e) for e in self.episodes), default=0)
            raise ValueError(
                f"no stored episode is {length} steps long "
                f"(longest is {longest}, {len(self.episodes)} stored)"
            )

        observations, actions, rewards, discounts = [], [], [], []
        for _ in range(batch_size):
            episode = usable[self._randint(len(usable))]
            start = self._randint(len(episode) - length + 1)
            stop = start + length
            observations.append(episode.observations[start:stop])
            actions.append(episode.actions[start:stop])
            rewards.append(episode.rewards[start:stop])
            discounts.append(episode.discounts[start:stop])

        return Episode(
            observations=lucid.stack(observations, dim=0),
            actions=lucid.stack(actions, dim=0),
            rewards=lucid.stack(rewards, dim=0),
            discounts=lucid.stack(discounts, dim=0),
        )

    @staticmethod
    def _randint(bound: int) -> int:
        """Uniform integer in ``[0, bound)``, drawn through Lucid's own RNG.

        Parameters
        ----------
        bound : int
            Exclusive upper bound, at least 1.

        Returns
        -------
        int
            The drawn index.

        Notes
        -----
        Uses ``lucid.rand`` rather than the standard library's ``random``
        so that :func:`lucid.manual_seed` makes a whole training run
        reproducible — the buffer's draws included.
        """
        drawn = int(float(lucid.rand(()).item()) * bound)
        return min(drawn, bound - 1)
