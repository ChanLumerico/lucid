"""Unit tests for lucid.fft — exercises every public function with closed-form
expectations.  No reference framework dependency; checks numerical identities
that any correct DFT implementation must satisfy."""

import math
import numpy as np
import pytest
import lucid
import lucid.fft as F

# ── Round-trip identities ────────────────────────────────────────────────────


class TestRoundTrip:
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    def test_fft_ifft_1d(self, norm):
        x = lucid.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        out = F.ifft(F.fft(x, norm=norm), norm=norm)
        np.testing.assert_allclose(out.numpy().real, x.numpy(), atol=1e-5)

    @pytest.mark.parametrize("norm", [None, "ortho"])
    def test_fft2_ifft2(self, norm):
        x = lucid.arange(0, 12, 1, dtype=lucid.float32).reshape(3, 4)
        out = F.ifft2(F.fft2(x, norm=norm), norm=norm)
        np.testing.assert_allclose(out.numpy().real, x.numpy(), atol=1e-4)

    @pytest.mark.parametrize("norm", [None, "ortho", "forward"])
    def test_fftn_ifftn(self, norm):
        x = lucid.arange(0, 24, 1, dtype=lucid.float32).reshape(2, 3, 4)
        out = F.ifftn(F.fftn(x, norm=norm), norm=norm)
        np.testing.assert_allclose(out.numpy().real, x.numpy(), atol=1e-4)

    @pytest.mark.parametrize("norm", [None, "ortho", "forward"])
    def test_rfft_irfft(self, norm):
        x = lucid.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        out = F.irfft(F.rfft(x, norm=norm), n=6, norm=norm)
        np.testing.assert_allclose(out.numpy(), x.numpy(), atol=1e-5)

    @pytest.mark.parametrize("norm", [None, "ortho"])
    def test_rfft2_irfft2(self, norm):
        x = lucid.arange(0, 16, 1, dtype=lucid.float32).reshape(4, 4)
        out = F.irfft2(F.rfft2(x, norm=norm), s=[4, 4], norm=norm)
        np.testing.assert_allclose(out.numpy(), x.numpy(), atol=1e-4)

    def test_hfft_ihfft_roundtrip(self):
        x = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        out = F.hfft(F.ihfft(x), n=4)
        np.testing.assert_allclose(out.numpy(), x.numpy(), atol=1e-5)


# ── Numerical correctness (vs. closed-form expected values) ──────────────────


class TestKnownValues:
    def test_fft_impulse(self):
        """FFT of impulse [1,0,0,0] is all-ones."""
        x = lucid.tensor([1.0, 0.0, 0.0, 0.0])
        y = F.fft(x).numpy()
        np.testing.assert_allclose(y, np.ones(4, dtype=np.complex64), atol=1e-6)

    def test_ifft_constant(self):
        """IFFT of a non-zero DC component (rest zero) is constant."""
        X = lucid.tensor([4.0 + 0j, 0.0, 0.0, 0.0], dtype=lucid.complex64)
        y = F.ifft(X).numpy()
        np.testing.assert_allclose(y, np.full(4, 1.0, dtype=np.complex64), atol=1e-6)

    def test_fft_sum_dc(self):
        """The 0-th FFT bin equals sum of input."""
        x = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        y = F.fft(x).numpy()
        assert abs(y[0] - 10.0) < 1e-5

    def test_rfft_length(self):
        """rfft(x) of length n returns n//2+1 bins."""
        x = lucid.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        y = F.rfft(x)
        assert y.shape == (5,)

    def test_irfft_default_length(self):
        """irfft default reconstructs to 2*(n_input-1)."""
        x = lucid.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # rfft of length-8
        y = F.irfft(F.rfft(lucid.arange(0, 8, 1, dtype=lucid.float32)))
        assert y.shape == (8,)


# ── Norm conventions ─────────────────────────────────────────────────────────


class TestNorm:
    def test_ortho_unitary(self):
        """ortho-normalised fft is its own conjugate transpose: |fft(x)|^2 sum
        equals |x|^2 sum (Parseval)."""
        x = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        y = F.fft(x, norm="ortho").numpy()
        # |x|^2 vs |y|^2
        np.testing.assert_allclose(
            (y.real**2 + y.imag**2).sum(), (x.numpy() ** 2).sum(), atol=1e-4
        )

    def test_forward_scales_input(self):
        """Under norm='forward', fft scales by 1/N."""
        x = lucid.tensor([4.0, 0.0, 0.0, 0.0])
        y_default = F.fft(x).numpy()
        y_forward = F.fft(x, norm="forward").numpy()
        np.testing.assert_allclose(y_forward * 4.0, y_default, atol=1e-5)


# ── Shifts and freqs ────────────────────────────────────────────────────────


class TestShifts:
    def test_fftshift_even(self):
        x = lucid.arange(0, 8, 1, dtype=lucid.float32)
        y = F.fftshift(x).numpy()
        expected = np.array([4, 5, 6, 7, 0, 1, 2, 3], dtype=np.float32)
        np.testing.assert_array_equal(y, expected)

    def test_fftshift_odd(self):
        x = lucid.arange(0, 7, 1, dtype=lucid.float32)
        y = F.fftshift(x).numpy()
        expected = np.array([4, 5, 6, 0, 1, 2, 3], dtype=np.float32)
        np.testing.assert_array_equal(y, expected)

    def test_ifftshift_inverse(self):
        for n in (4, 5, 6, 7, 8):
            x = lucid.arange(0, n, 1, dtype=lucid.float32)
            y = F.ifftshift(F.fftshift(x)).numpy()
            np.testing.assert_array_equal(y, x.numpy())


class TestFreqs:
    def test_fftfreq_even(self):
        y = F.fftfreq(8, d=1.0).numpy()
        expected = np.fft.fftfreq(8, 1.0).astype(y.dtype)
        np.testing.assert_allclose(y, expected, atol=1e-7)

    def test_fftfreq_odd(self):
        y = F.fftfreq(7, d=1.0).numpy()
        expected = np.fft.fftfreq(7, 1.0).astype(y.dtype)
        np.testing.assert_allclose(y, expected, atol=1e-7)

    def test_rfftfreq(self):
        y = F.rfftfreq(8, d=2.0).numpy()
        expected = np.fft.rfftfreq(8, 2.0).astype(y.dtype)
        np.testing.assert_allclose(y, expected, atol=1e-7)


# ── Autograd ────────────────────────────────────────────────────────────────


class TestAutograd:
    def test_fft_forward_shape(self):
        """fft preserves shape; output is C64 for real input."""
        x = lucid.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        y = F.fft(x)
        assert y.shape == (4,)
        assert y.dtype == lucid.complex64

    def test_rfft_irfft_grad_chain(self):
        """grad through rfft -> irfft round-trip should be identity (within
        numerical tolerance) on real inputs."""
        x = lucid.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        y = F.irfft(F.rfft(x), n=4)
        # y == x; sum() backward gives all-ones grad for x
        y.sum().backward()
        np.testing.assert_allclose(
            x.grad.numpy(), np.ones(4, dtype=np.float32), atol=1e-4
        )

    def test_fftshift_grad_passes_through(self):
        x = lucid.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        y = F.fftshift(x)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), np.ones(4, dtype=np.float32))


# ── API surface ─────────────────────────────────────────────────────────────


class TestAPISurface:
    def test_all_22_functions_exist(self):
        expected = {
            "fft",
            "ifft",
            "fft2",
            "ifft2",
            "fftn",
            "ifftn",
            "rfft",
            "irfft",
            "rfft2",
            "irfft2",
            "rfftn",
            "irfftn",
            "hfft",
            "ihfft",
            "hfft2",
            "ihfft2",
            "hfftn",
            "ihfftn",
            "fftfreq",
            "rfftfreq",
            "fftshift",
            "ifftshift",
        }
        assert expected.issubset(set(F.__all__))
        for name in expected:
            assert callable(getattr(F, name)), f"{name} not callable"

    def test_lucid_fft_subpackage(self):
        assert lucid.fft is F

    def test_invalid_norm_rejected(self):
        x = lucid.tensor([1.0, 2.0])
        with pytest.raises(ValueError):
            F.fft(x, norm="bogus")
