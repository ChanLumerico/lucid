"""nn.utils — clip_grad, parametrize, weight_norm, etc."""

import numpy as np
import pytest

import lucid
import lucid.nn as nn


class TestClipGradNorm:
    def test_clamps_to_max(self) -> None:
        x = lucid.tensor([3.0, 4.0], requires_grad=True)
        (x * x).sum().backward()
        # ||grad||_2 = ||[6, 8]||_2 = 10.
        total_norm = nn.utils.clip_grad_norm_([x], max_norm=5.0)
        # ``total_norm`` may be a Tensor or a Python float depending on
        # the implementation — accept either.
        tn = total_norm.item() if hasattr(total_norm, "item") else float(total_norm)
        assert abs(tn - 10.0) < 1e-4
        # After clipping, ||grad||_2 should be ~5.
        new_norm = float(np.sqrt((x.grad.numpy() ** 2).sum()))
        assert abs(new_norm - 5.0) < 1e-4


class TestClipGradValue:
    def test_clamps_each_element(self) -> None:
        x = lucid.tensor([1.0, 2.0], requires_grad=True)
        (x * x * x).sum().backward()
        # grad = 3x² = [3, 12].
        nn.utils.clip_grad_value_([x], clip_value=5.0)
        np.testing.assert_array_equal(x.grad.numpy(), [3.0, 5.0])
