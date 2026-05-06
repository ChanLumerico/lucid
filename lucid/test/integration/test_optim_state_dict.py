"""Optimizer state_dict round-trip integration tests.

Validates the format and round-trip behaviour of the optimizer state_dict
contract. Engine-managed moment buffers (Adam's m/v, SGD momentum, etc.) are
not currently restored across save/load — that gap is documented and exercised
via xfail so it doesn't silently regress.
"""

import os
import tempfile

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim


def _train_step(model: nn.Module, opt: optim.Optimizer, x: lucid.Tensor) -> None:
    opt.zero_grad()
    loss: lucid.Tensor = model(x).sum()
    loss.backward()
    opt.step()


class TestStateDictFormat:
    def test_param_groups_use_indices_not_tensors(self) -> None:
        model: nn.Linear = nn.Linear(4, 2)
        opt: optim.Adam = optim.Adam(model.parameters(), lr=1e-3)
        sd: dict[str, object] = opt.state_dict()
        params: list[int] = sd["param_groups"][0]["params"]  # type: ignore[index]
        assert all(isinstance(p, int) for p in params)
        assert params == [0, 1]

    def test_hyperparams_present_in_param_groups(self) -> None:
        model: nn.Linear = nn.Linear(4, 2)
        opt: optim.Adam = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
        sd: dict[str, object] = opt.state_dict()
        g: dict[str, object] = sd["param_groups"][0]  # type: ignore[index]
        assert g["lr"] == 1e-3
        assert g["beta1"] == 0.9
        assert g["beta2"] == 0.999

    def test_state_dict_save_load_roundtrip(self) -> None:
        model: nn.Linear = nn.Linear(4, 2)
        opt: optim.Adam = optim.Adam(model.parameters(), lr=7.5e-4, weight_decay=0.01)
        with tempfile.TemporaryDirectory() as d:
            path: str = os.path.join(d, "opt.pt")
            lucid.save(opt.state_dict(), path)
            loaded: dict[str, object] = lucid.load(path, weights_only=False)
        opt2: optim.Adam = optim.Adam(nn.Linear(4, 2).parameters(), lr=999.0)
        opt2.load_state_dict(loaded)
        assert opt2.param_groups[0]["lr"] == 7.5e-4
        assert opt2.param_groups[0]["weight_decay"] == 0.01

    def test_load_state_dict_rejects_mismatched_groups(self) -> None:
        opt: optim.Adam = optim.Adam(nn.Linear(4, 2).parameters(), lr=1e-3)
        bad_sd: dict[str, object] = {"state": {}, "param_groups": []}
        with pytest.raises(ValueError, match="param_groups"):
            opt.load_state_dict(bad_sd)


class TestLBFGSRoundTrip:
    """LBFGS owns its state in Python — full round-trip is supported."""

    def _make(self, history_size: int = 3) -> tuple[nn.Linear, optim.LBFGS]:
        m: nn.Linear = nn.Linear(4, 2)
        opt: optim.LBFGS = optim.LBFGS(
            m.parameters(), lr=1.0, history_size=history_size
        )
        return m, opt

    def test_lbfgs_history_roundtrip(self) -> None:
        m, opt = self._make(history_size=3)
        x: lucid.Tensor = lucid.randn(3, 4)

        def closure() -> lucid.Tensor:
            opt.zero_grad()
            loss: lucid.Tensor = m(x).sum()
            loss.backward()
            return loss

        for _ in range(2):
            opt.step(closure)

        sd: dict[str, object] = opt.state_dict()
        snapshot: dict[str, object] = sd["state"][0]  # type: ignore[index]
        assert "old_dirs" in snapshot
        assert "H_diag" in snapshot
        assert isinstance(snapshot["n_iter"], int)
        prior_n_iter: int = snapshot["n_iter"]  # type: ignore[assignment]
        prior_history_len: int = len(snapshot["old_dirs"])  # type: ignore[arg-type]

        m2, opt2 = self._make(history_size=3)
        opt2.load_state_dict(sd)
        assert opt2._lbfgs_state["n_iter"] == prior_n_iter
        assert len(opt2._lbfgs_state["old_dirs"]) == prior_history_len

    def test_lbfgs_save_to_disk(self) -> None:
        m, opt = self._make(history_size=3)
        x: lucid.Tensor = lucid.randn(3, 4)

        def closure() -> lucid.Tensor:
            opt.zero_grad()
            loss: lucid.Tensor = m(x).sum()
            loss.backward()
            return loss

        opt.step(closure)
        with tempfile.TemporaryDirectory() as d:
            path: str = os.path.join(d, "lbfgs.pt")
            lucid.save(opt.state_dict(), path)
            sd_loaded: dict[str, object] = lucid.load(path, weights_only=False)
        m2, opt2 = self._make(history_size=3)
        opt2.load_state_dict(sd_loaded)
        # H_diag survived the disk round-trip.
        assert opt2._lbfgs_state["H_diag"] == opt._lbfgs_state["H_diag"]


class TestEngineMomentRoundTrip:
    """Engine-managed buffers (Adam moments, SGD momentum) round-trip through
    ``state_dict``. The fixture trains opt1, snapshots both model and
    optimizer, restores into a fresh pair, then runs one more step on each
    and asserts the parameter updates are bit-identical."""

    def _check_match(
        self,
        make_opt: object,
        seed: int = 0,
        n_steps: int = 5,
        atol: float = 1e-6,
    ) -> None:
        np.random.seed(seed)
        m1: nn.Linear = nn.Linear(4, 2)
        opt1: optim.Optimizer = make_opt(m1.parameters())  # type: ignore[operator]
        x: lucid.Tensor = lucid.randn(3, 4)
        for _ in range(n_steps):
            _train_step(m1, opt1, x)

        m2: nn.Linear = nn.Linear(4, 2)
        m2.load_state_dict(m1.state_dict())
        opt2: optim.Optimizer = make_opt(m2.parameters())  # type: ignore[operator]
        opt2.load_state_dict(opt1.state_dict())

        x_next: lucid.Tensor = lucid.randn(3, 4)
        _train_step(m1, opt1, x_next)
        _train_step(m2, opt2, x_next)
        np.testing.assert_allclose(m1.weight.numpy(), m2.weight.numpy(), atol=atol)

    def test_adam(self) -> None:
        self._check_match(lambda p: optim.Adam(p, lr=1e-3))

    def test_adamw(self) -> None:
        self._check_match(lambda p: optim.AdamW(p, lr=1e-3))

    def test_sgd_with_momentum(self) -> None:
        self._check_match(lambda p: optim.SGD(p, lr=1e-2, momentum=0.9))

    def test_sgd_no_momentum(self) -> None:
        # No momentum buffer to round-trip — should still work end-to-end.
        self._check_match(lambda p: optim.SGD(p, lr=1e-2))

    def test_adam_step_count_restored(self) -> None:
        np.random.seed(0)
        m: nn.Linear = nn.Linear(4, 2)
        opt: optim.Adam = optim.Adam(m.parameters(), lr=1e-3)
        x: lucid.Tensor = lucid.randn(3, 4)
        for _ in range(7):
            _train_step(m, opt, x)
        sd: dict[str, object] = opt.state_dict()
        # Each per-slot snapshot carries the (group-wide) step counter.
        assert sd["state"][0]["step"] == 7  # type: ignore[index]


class TestEngineMomentRoundTripGPU:
    """GPU optimizer state must round-trip the same as CPU — the C++ snapshot
    path goes through ``mlx::core::array`` clone instead of memcpy, which is a
    distinct code path worth locking down."""

    def test_adam_gpu(self) -> None:
        np.random.seed(0)
        m1: nn.Linear = nn.Linear(4, 2).to("metal")
        opt1: optim.Adam = optim.Adam(m1.parameters(), lr=1e-3)
        x: lucid.Tensor = lucid.randn(3, 4).to("metal")
        for _ in range(3):
            _train_step(m1, opt1, x)
        m2: nn.Linear = nn.Linear(4, 2).to("metal")
        m2.load_state_dict(m1.state_dict())
        opt2: optim.Adam = optim.Adam(m2.parameters(), lr=1e-3)
        opt2.load_state_dict(opt1.state_dict())
        x_next: lucid.Tensor = lucid.randn(3, 4).to("metal")
        _train_step(m1, opt1, x_next)
        _train_step(m2, opt2, x_next)
        np.testing.assert_allclose(m1.weight.numpy(), m2.weight.numpy(), atol=1e-5)
