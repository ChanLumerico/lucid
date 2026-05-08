"""``lucid.signal.windows`` — 12 spectral window functions."""

import numpy as np
import pytest

import lucid


_WINDOW_NAMES = [
    "bartlett", "blackman", "cosine", "exponential", "gaussian",
    "general_cosine", "general_hamming", "general_gaussian",
    "hamming", "hann", "kaiser", "nuttall",
]


class TestWindowSurface:
    def test_all_present(self) -> None:
        for name in _WINDOW_NAMES:
            assert hasattr(lucid.signal.windows, name), f"missing {name}"


class TestWindowShapes:
    @pytest.mark.parametrize("M", [1, 2, 8, 31, 32, 64])
    def test_hann_length(self, M: int) -> None:
        out = lucid.signal.windows.hann(M)
        assert out.shape == (M,)

    @pytest.mark.parametrize("M", [1, 8, 32])
    def test_hamming_length(self, M: int) -> None:
        out = lucid.signal.windows.hamming(M)
        assert out.shape == (M,)


class TestKnownValues:
    def test_hann_endpoints(self) -> None:
        # Symmetric Hann at length M=8: w[0] == w[-1] == 0.
        out = lucid.signal.windows.hann(8, sym=True).numpy()
        assert abs(out[0]) < 1e-6
        assert abs(out[-1]) < 1e-6

    def test_hamming_value_at_center(self) -> None:
        # Symmetric Hamming peaks at the center index.
        out = lucid.signal.windows.hamming(11, sym=True).numpy()
        assert out[5] >= max(out[0], out[-1])

    def test_blackman_min_zero(self) -> None:
        out = lucid.signal.windows.blackman(8, sym=True).numpy()
        # Blackman touches near-zero at the boundaries.
        assert out[0] < 0.01
        assert out[-1] < 0.01

    def test_cosine_endpoints_nonzero(self) -> None:
        out = lucid.signal.windows.cosine(8).numpy()
        # Cosine is half-cycle — non-zero at both ends.
        assert out[0] > 0.0
        assert out[-1] > 0.0


class TestSym:
    def test_sym_vs_not(self) -> None:
        sym = lucid.signal.windows.hann(8, sym=True).numpy()
        peri = lucid.signal.windows.hann(8, sym=False).numpy()
        # The two outputs should differ for even-length windows.
        assert not np.allclose(sym, peri)
