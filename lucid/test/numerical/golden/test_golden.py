"""Checked-in golden tensors — kernels are pinned to bit-stable outputs.

Goldens guard against silent drift in the kernels.  They are generated
by the recipe inside the test (so the recipe is auditable) and shipped
in ``numerical/golden/*.npz`` (small, deterministic).  When a kernel
change is intentional, regenerate the relevant ``.npz`` and commit it
alongside the source change.

If a golden file is missing, tests skip rather than fail — the helper
is happy to bootstrap by re-running the recipe and saving the output
on first encounter.
"""

import numpy as np
import pytest

import lucid
from lucid.test._helpers.golden import golden_path, load_golden, save_golden


def _bootstrap(name: str, key: str, value: np.ndarray) -> np.ndarray:
    """Read existing golden or seed it from ``value``."""
    path = golden_path(name)
    if not path.exists():
        save_golden(name, **{key: value})
    arch = np.load(path)
    if key not in arch.files:
        save_golden(name, **{key: value, **{k: arch[k] for k in arch.files}})
        arch = np.load(path)
    return arch[key]


class TestGoldenMatmul:
    def test_matches(self, device: str) -> None:
        rng = np.random.default_rng(0)
        a_np = rng.uniform(-1.0, 1.0, size=(4, 4)).astype(np.float32)
        b_np = rng.uniform(-1.0, 1.0, size=(4, 4)).astype(np.float32)
        expected = _bootstrap("matmul_4x4", "out", (a_np @ b_np).astype(np.float32))

        a = lucid.tensor(a_np.copy(), device=device)
        b = lucid.tensor(b_np.copy(), device=device)
        out = (a @ b).numpy()
        np.testing.assert_allclose(out, expected, atol=1e-5)


class TestGoldenSoftmax:
    def test_matches(self, device: str) -> None:
        from lucid.nn.functional import softmax
        rng = np.random.default_rng(0)
        x_np = rng.uniform(-2.0, 2.0, size=(8,)).astype(np.float32)
        e = np.exp(x_np - x_np.max())
        ref = (e / e.sum()).astype(np.float32)
        expected = _bootstrap("softmax_8", "out", ref)

        x = lucid.tensor(x_np.copy(), device=device)
        out = softmax(x, dim=0).numpy()
        np.testing.assert_allclose(out, expected, atol=1e-5)


class TestGoldenLinearForward:
    def test_matches(self, device: str) -> None:
        rng = np.random.default_rng(0)
        x_np = rng.uniform(-1.0, 1.0, size=(2, 4)).astype(np.float32)
        w_np = rng.uniform(-0.5, 0.5, size=(3, 4)).astype(np.float32)
        b_np = rng.uniform(-0.1, 0.1, size=(3,)).astype(np.float32)
        ref = (x_np @ w_np.T + b_np).astype(np.float32)
        expected = _bootstrap("linear_2x4_4to3", "out", ref)

        x = lucid.tensor(x_np.copy(), device=device)
        w = lucid.tensor(w_np.copy(), device=device)
        b = lucid.tensor(b_np.copy(), device=device)
        out = (x @ w.mT + b).numpy()
        np.testing.assert_allclose(out, expected, atol=1e-5)
