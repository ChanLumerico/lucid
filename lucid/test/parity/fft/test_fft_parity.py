"""Reference parity for FFT ops."""

from typing import Any

import numpy as np
import pytest

import lucid
from lucid.test._helpers.compare import assert_close


@pytest.mark.parity
class TestFFTParity:
    @pytest.fixture
    def x_pair(self, ref: Any) -> tuple[lucid.Tensor, Any]:
        np.random.seed(0)
        x = np.random.standard_normal(size=(8,)).astype(np.float32)
        return lucid.tensor(x.copy()), ref.tensor(x.copy())

    def test_fft(self, x_pair, ref) -> None:  # type: ignore[no-untyped-def]
        l, r = x_pair
        l_out = lucid.fft.fft(l).numpy()
        r_out = ref.fft.fft(r).detach().cpu().numpy()
        np.testing.assert_allclose(l_out.real, r_out.real, atol=1e-4)
        np.testing.assert_allclose(l_out.imag, r_out.imag, atol=1e-4)

    def test_rfft(self, x_pair, ref) -> None:  # type: ignore[no-untyped-def]
        l, r = x_pair
        l_out = lucid.fft.rfft(l).numpy()
        r_out = ref.fft.rfft(r).detach().cpu().numpy()
        np.testing.assert_allclose(l_out.real, r_out.real, atol=1e-4)
        np.testing.assert_allclose(l_out.imag, r_out.imag, atol=1e-4)

    def test_fftshift(self, x_pair, ref) -> None:  # type: ignore[no-untyped-def]
        l, r = x_pair
        assert_close(lucid.fft.fftshift(l), ref.fft.fftshift(r), atol=1e-6)
