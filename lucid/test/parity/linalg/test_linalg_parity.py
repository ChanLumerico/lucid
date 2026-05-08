"""Reference parity for linalg ops."""

from typing import Any

import numpy as np
import pytest

import lucid
from lucid.test._helpers.compare import assert_close


def _spd(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((4, 4)).astype(np.float32)
    return (M @ M.T + 4 * np.eye(4)).astype(np.float32)


@pytest.mark.parity
class TestLinalgParity:
    def test_norm_l2(self, ref: Any) -> None:
        v = np.array([3.0, 4.0], dtype=np.float32)
        assert_close(
            lucid.linalg.norm(lucid.tensor(v.copy())),
            ref.linalg.norm(ref.tensor(v.copy())),
            atol=1e-5,
        )

    def test_solve(self, ref: Any) -> None:
        A_np = _spd()
        b_np = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        l = lucid.linalg.solve(lucid.tensor(A_np.copy()), lucid.tensor(b_np.copy()))
        r = ref.linalg.solve(ref.tensor(A_np.copy()), ref.tensor(b_np.copy()))
        assert_close(l, r, atol=1e-3)

    def test_inv(self, ref: Any) -> None:
        A_np = _spd()
        l = lucid.linalg.inv(lucid.tensor(A_np.copy()))
        r = ref.linalg.inv(ref.tensor(A_np.copy()))
        assert_close(l, r, atol=1e-3)

    def test_cholesky_reconstruction(self, ref: Any) -> None:
        # The factor itself isn't unique up to sign, but Lᴿ Lᴿᵀ = A holds for both.
        A_np = _spd()
        A_l = lucid.tensor(A_np.copy())
        A_r = ref.tensor(A_np.copy())
        Lᴸ = lucid.linalg.cholesky(A_l)
        Lᴿ = ref.linalg.cholesky(A_r)
        assert_close(Lᴸ @ Lᴸ.mT, Lᴿ @ Lᴿ.mT, atol=1e-3)
