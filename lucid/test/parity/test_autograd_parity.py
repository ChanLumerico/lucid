"""Reference parity for backward / gradient values."""

from typing import Any

import numpy as np
import pytest

import lucid


@pytest.mark.parity
class TestAutogradParity:
    def test_square_sum_backward(self, ref: Any) -> None:
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Lucid pass.
        x_l = lucid.tensor(x_np.copy(), requires_grad=True)
        (x_l * x_l).sum().backward()

        # Reference pass.
        x_r = ref.tensor(x_np.copy(), requires_grad=True)
        (x_r * x_r).sum().backward()

        np.testing.assert_allclose(
            x_l.grad.numpy(),
            x_r.grad.detach().cpu().numpy(),
            atol=1e-5,
        )

    def test_chain_backward(self, ref: Any) -> None:
        x_np = np.array([1.5, 2.5], dtype=np.float32)

        x_l = lucid.tensor(x_np.copy(), requires_grad=True)
        (x_l.exp().sum() + x_l.sin().sum()).backward()

        x_r = ref.tensor(x_np.copy(), requires_grad=True)
        (x_r.exp().sum() + x_r.sin().sum()).backward()

        np.testing.assert_allclose(
            x_l.grad.numpy(),
            x_r.grad.detach().cpu().numpy(),
            atol=1e-5,
        )

    def test_matmul_backward(self, ref: Any) -> None:
        np.random.seed(0)
        a_np = np.random.standard_normal(size=(3, 4)).astype(np.float32)
        b_np = np.random.standard_normal(size=(4, 2)).astype(np.float32)

        a_l = lucid.tensor(a_np.copy(), requires_grad=True)
        b_l = lucid.tensor(b_np.copy(), requires_grad=True)
        (a_l @ b_l).sum().backward()

        a_r = ref.tensor(a_np.copy(), requires_grad=True)
        b_r = ref.tensor(b_np.copy(), requires_grad=True)
        (a_r @ b_r).sum().backward()

        np.testing.assert_allclose(
            a_l.grad.numpy(),
            a_r.grad.detach().cpu().numpy(),
            atol=1e-4,
        )
        np.testing.assert_allclose(
            b_l.grad.numpy(),
            b_r.grad.detach().cpu().numpy(),
            atol=1e-4,
        )
