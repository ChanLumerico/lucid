"""Reference parity for distributions — log_prob + closed-form KL."""

import math
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.distributions as D
from lucid.test._helpers.compare import assert_close


@pytest.mark.parity
class TestNormalParity:
    def test_log_prob(self, ref: Any) -> None:
        loc, scale = 0.5, 1.5
        x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        l = D.Normal(loc, scale).log_prob(lucid.tensor(x.copy())).numpy()
        r = ref.distributions.Normal(loc, scale).log_prob(ref.tensor(x.copy())).detach().cpu().numpy()
        np.testing.assert_allclose(l, r, atol=1e-5)

    def test_entropy(self, ref: Any) -> None:
        loc, scale = 0.5, 1.5
        l = D.Normal(loc, scale).entropy().item()
        r = ref.distributions.Normal(loc, scale).entropy().item()
        assert abs(l - r) < 1e-5


@pytest.mark.parity
class TestKLParity:
    def test_kl_normal(self, ref: Any) -> None:
        l = D.kl_divergence(D.Normal(0.0, 1.0), D.Normal(1.0, 2.0)).item()
        r = ref.distributions.kl_divergence(
            ref.distributions.Normal(0.0, 1.0),
            ref.distributions.Normal(1.0, 2.0),
        ).item()
        assert abs(l - r) < 1e-5

    def test_kl_categorical(self, ref: Any) -> None:
        p = [0.2, 0.5, 0.3]
        q = [0.3, 0.3, 0.4]
        l = D.kl_divergence(
            D.Categorical(probs=lucid.tensor(p)),
            D.Categorical(probs=lucid.tensor(q)),
        ).item()
        r = ref.distributions.kl_divergence(
            ref.distributions.Categorical(probs=ref.tensor(p)),
            ref.distributions.Categorical(probs=ref.tensor(q)),
        ).item()
        assert abs(l - r) < 1e-5
