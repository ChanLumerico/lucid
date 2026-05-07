"""Unit tests for ``lucid.signal.windows``.

Each window is checked for:

* Correct output length.
* Symmetric profile (``w[i] == w[-1-i]`` for ``sym=True``).
* Endpoint values (the "boundary signature" of each family).
* ``sym=False`` vs ``sym=True`` length agreement (both produce length M).
"""

import math

import numpy as np
import pytest

import lucid
import lucid.signal.windows as W


def _is_symmetric(arr: np.ndarray, atol: float = 1e-5) -> bool:
    return np.allclose(arr, arr[::-1], atol=atol)


class TestLengths:
    @pytest.mark.parametrize(
        "name, kwargs",
        [
            ("hann", {}),
            ("hamming", {}),
            ("blackman", {}),
            ("nuttall", {}),
            ("bartlett", {}),
            ("cosine", {}),
            ("gaussian", {}),
            ("general_hamming", {"alpha": 0.54}),
            ("general_cosine", {"a": [0.5, 0.5]}),
            ("general_gaussian", {}),
            ("exponential", {}),
            ("kaiser", {}),
        ],
    )
    def test_length_matches_M(self, name, kwargs) -> None:
        for M in (1, 2, 7, 16, 65):
            fn = getattr(W, name)
            assert fn(M, **kwargs).shape == (M,)
            # ``exponential`` requires an explicit centre when sym=False;
            # other windows have a canonical periodic form.
            extra = {"center": (M - 1) / 2.0} if name == "exponential" else {}
            assert fn(M, sym=False, **kwargs, **extra).shape == (M,)


class TestSymmetry:
    @pytest.mark.parametrize(
        "name, kwargs",
        [
            ("hann", {}),
            ("hamming", {}),
            ("blackman", {}),
            ("nuttall", {}),
            ("bartlett", {}),
            ("cosine", {}),
            ("gaussian", {}),
            ("general_hamming", {"alpha": 0.54}),
            ("general_cosine", {"a": [0.5, 0.5]}),
            ("general_gaussian", {}),
            ("kaiser", {}),
        ],
    )
    def test_sym_window_is_symmetric(self, name, kwargs) -> None:
        fn = getattr(W, name)
        for M in (5, 8, 16):
            arr = fn(M, sym=True, **kwargs).numpy()
            assert _is_symmetric(arr), f"{name}(M={M}) not symmetric"


class TestKnownValues:
    def test_hann_endpoints_zero(self) -> None:
        # Hann starts and ends at zero (within float epsilon).
        arr = W.hann(16).numpy()
        assert abs(arr[0]) < 1e-5
        assert abs(arr[-1]) < 1e-5

    def test_hamming_endpoint_alpha(self) -> None:
        # Hamming endpoints equal `1 - alpha = 0.46` … wait, the formula is
        # `alpha - (1-alpha) cos(0)` = `alpha - (1-alpha)` = `2*alpha - 1` = 0.08
        arr = W.hamming(16).numpy()
        assert abs(arr[0] - 0.08) < 1e-4
        assert abs(arr[-1] - 0.08) < 1e-4

    def test_bartlett_zero_endpoints_one_centre(self) -> None:
        arr = W.bartlett(9).numpy()
        assert abs(arr[0]) < 1e-6
        assert abs(arr[-1]) < 1e-6
        assert abs(arr[4] - 1.0) < 1e-6  # exact centre for odd M

    def test_kaiser_centre_is_one(self) -> None:
        arr = W.kaiser(15, beta=8.0).numpy()
        assert abs(arr[7] - 1.0) < 1e-5

    def test_cosine_no_zero(self) -> None:
        # Cosine (half-sine) is strictly positive — no zeros at boundaries.
        arr = W.cosine(8).numpy()
        assert np.all(arr > 0)


class TestParameterised:
    def test_general_hamming_eq_hann(self) -> None:
        np.testing.assert_allclose(
            W.general_hamming(16, alpha=0.5).numpy(),
            W.hann(16).numpy(),
            atol=1e-6,
        )

    def test_general_hamming_eq_hamming(self) -> None:
        np.testing.assert_allclose(
            W.general_hamming(16, alpha=0.54).numpy(),
            W.hamming(16).numpy(),
            atol=1e-6,
        )

    def test_general_cosine_eq_blackman(self) -> None:
        np.testing.assert_allclose(
            W.general_cosine(16, [0.42, 0.50, 0.08]).numpy(),
            W.blackman(16).numpy(),
            atol=1e-6,
        )

    def test_general_gaussian_p1_eq_gaussian(self) -> None:
        # p=1 is the standard Gaussian.
        np.testing.assert_allclose(
            W.general_gaussian(16, p=1.0, sig=4.0).numpy(),
            W.gaussian(16, std=4.0).numpy(),
            atol=1e-6,
        )

    def test_exponential_default_centre_symmetric(self) -> None:
        arr = W.exponential(16, tau=2.0).numpy()
        assert _is_symmetric(arr)

    def test_exponential_periodic_requires_centre(self) -> None:
        with pytest.raises(ValueError):
            W.exponential(8, sym=False)


class TestEdgeCases:
    def test_M_zero(self) -> None:
        assert W.hann(0).shape == (0,)
        assert W.kaiser(0).shape == (0,)

    def test_M_one(self) -> None:
        # Length-1 windows are conventionally [1.0].
        assert W.hann(1).numpy().tolist() == [1.0]
        assert W.hamming(1).numpy().tolist() == [1.0]
        assert W.kaiser(1).numpy().tolist() == [1.0]

    def test_negative_M_rejected(self) -> None:
        with pytest.raises(ValueError):
            W.hann(-1)


class TestNamespacePolicy:
    def test_only_via_signal_windows(self) -> None:
        # Per H8: windows are accessed only through ``lucid.signal.windows``.
        # No top-level ``lucid.hann`` etc.
        for name in ("hann", "hamming", "blackman", "nuttall", "kaiser"):
            assert not hasattr(lucid, name), (
                f"lucid.{name} should not exist — H8 forbids window shortcuts"
            )

    def test_signal_subpackage_exists(self) -> None:
        assert callable(lucid.signal.windows.hann)
