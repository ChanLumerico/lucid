"""``lucid.fft`` — DFT / Hermitian / shift / freq."""

import numpy as np

import lucid


class TestFFT:
    def test_constant_input(self) -> None:
        # FFT of a real constant signal — only DC bin is non-zero.
        t = lucid.tensor([1.0, 1.0, 1.0, 1.0])
        out = lucid.fft.fft(t).numpy()
        assert abs(out[0]) > 0
        for i in range(1, 4):
            assert abs(out[i]) < 1e-5

    def test_inverse_round_trip(self) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        out = lucid.fft.ifft(lucid.fft.fft(t)).numpy()
        # Imaginary part should vanish; real part must match.
        np.testing.assert_allclose(out.real, t.numpy(), atol=1e-4)


class TestRFFT:
    def test_round_trip(self) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        out = lucid.fft.irfft(lucid.fft.rfft(t), n=8).numpy()
        np.testing.assert_allclose(out, t.numpy(), atol=1e-4)


class TestFFT2:
    def test_basic_shape(self) -> None:
        t = lucid.zeros(4, 8)
        out = lucid.fft.fft2(t)
        assert out.shape == t.shape


class TestFFTShift:
    def test_basic(self) -> None:
        t = lucid.tensor([0.0, 1.0, 2.0, 3.0])
        out = lucid.fft.fftshift(t).numpy()
        np.testing.assert_array_equal(out, [2.0, 3.0, 0.0, 1.0])


class TestFreqs:
    def test_fftfreq(self) -> None:
        out = lucid.fft.fftfreq(4).numpy()
        # Standard fftfreq for n=4 is [0, 0.25, -0.5, -0.25].
        np.testing.assert_allclose(out, [0.0, 0.25, -0.5, -0.25], atol=1e-6)

    def test_rfftfreq(self) -> None:
        out = lucid.fft.rfftfreq(4).numpy()
        # For n=4 rfftfreq → [0, 0.25, 0.5].
        np.testing.assert_allclose(out, [0.0, 0.25, 0.5], atol=1e-6)
