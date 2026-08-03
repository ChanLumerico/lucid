"""Regression tests: functions that answered NaN where a number exists.

Found 2026-08-04 by the audit's ``stability`` axis, which feeds each op an
input scaled across fourteen orders of magnitude.  Its premise — "a
finite, in-domain input must not produce a NaN" — carries the whole check
in the words *in domain*, and the axis had no way to know where that was:
of its twenty-six findings, ten were ``acos`` and friends being asked for
a value outside [-1, 1] and correctly refusing.

The domain is a fact about the function, not a property of the code, so
it is measured rather than listed — see
``lucid.test.audit.tools.stability_contract``.  What survived that filter
is below, and every one of them is a case where the reference answers an
ordinary number.

The expected values here are the reference's, not hand-derived: these are
asymptotic regimes where writing the answer out by hand is how the error
gets in.
"""

import numpy as np
import pytest

import lucid
import lucid.special as sp

_LARGE = np.array([1.0, 2.0, 5.0, 10.0, 100.0, 1e4, 1e6, 1e12])


# ── the scaled special functions ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,fn,expected",
    [
        (
            "erfcx",
            sp.erfcx,
            [
                4.2758358e-01,
                2.5539568e-01,
                1.1070464e-01,
                5.6140993e-02,
                5.6416138e-03,
                5.6418958e-05,
                5.6418958e-07,
                5.6418958e-13,
            ],
        ),
        (
            "i0e",
            sp.i0e,
            [
                4.6575961e-01,
                3.0850832e-01,
                1.8354081e-01,
                1.2783334e-01,
                3.9944379e-02,
                3.9894727e-03,
                3.9894233e-04,
                3.9894228e-07,
            ],
        ),
        (
            "i1e",
            sp.i1e,
            [
                2.0791042e-01,
                2.1526929e-01,
                1.6397227e-01,
                1.2126268e-01,
                3.9744153e-02,
                3.9892732e-03,
                3.9894213e-04,
                3.9894228e-07,
            ],
        ),
        (
            "scaled_modified_bessel_k0",
            sp.scaled_modified_bessel_k0,
            [
                1.1444631e00,
                8.4156822e-01,
                5.4780756e-01,
                3.9163193e-01,
                1.2517562e-01,
                1.2532985e-02,
                1.2533140e-03,
                1.2533141e-06,
            ],
        ),
        (
            "scaled_modified_bessel_k1",
            sp.scaled_modified_bessel_k1,
            [
                1.6361535e00,
                1.0334768e00,
                6.0027386e-01,
                4.1076657e-01,
                1.2579995e-01,
                1.2533611e-02,
                1.2533146e-03,
                1.2533141e-06,
            ],
        ),
    ],
    # Measured against the reference, not written out by hand: these are
    # asymptotic regimes, and hand-typing the values is how the first
    # version of this test failed.
)
def test_scaled_special_functions_stay_finite(name, fn, expected) -> None:
    """Each was written as its own definition, which is what broke it.

    ``erfcx = exp(x²)·erfc(x)``, ``i0e = exp(-|x|)·i0(x)``,
    ``k0e = k0(x)·exp(x)`` — every one of them forms an infinity and a
    zero and multiplies them together, when the point of the scaled
    variant is that the answer is bounded.  ``erfcx(100)`` is 0.00564.
    """
    got = np.asarray(fn(lucid.tensor(_LARGE)).numpy(), dtype=np.float64)
    assert np.isfinite(got).all(), f"{name} went non-finite: {got}"
    assert np.allclose(got, expected, rtol=2e-5), f"{name}: {got}"


@pytest.mark.parametrize(
    "name,fn",
    [
        ("erfcx", sp.erfcx),
        ("i0e", sp.i0e),
        ("i1e", sp.i1e),
        ("k0e", sp.scaled_modified_bessel_k0),
        ("k1e", sp.scaled_modified_bessel_k1),
    ],
)
def test_scaled_special_function_gradients_stay_finite(name, fn) -> None:
    """The branch that loses still contributes to the gradient.

    ``where`` selects a value and masks a derivative, and a mask times an
    infinity is a NaN — so both branches have to be clamped onto their
    own domain, not merely selected between.
    """
    x = lucid.tensor(np.array([0.5, 2.0, 10.0, 1e4]), requires_grad=True)
    fn(x).sum().backward()
    grad = np.asarray(x.grad.numpy()).ravel()
    assert np.isfinite(grad).all(), f"{name} gradient: {grad}"


def test_i0e_and_i1e_agree_with_their_unscaled_forms() -> None:
    """Guard the instrument: the branches must compute, not just not-NaN."""
    x = np.array([0.1, 1.0, 3.0, 3.75, 4.0, 8.0])
    for scaled, plain in ((sp.i0e, lucid.i0), (sp.i1e, sp.i1)):
        got = np.asarray(scaled(lucid.tensor(x)).numpy(), dtype=np.float64)
        want = np.asarray(plain(lucid.tensor(x)).numpy(), dtype=np.float64) * np.exp(
            -np.abs(x)
        )
        assert np.allclose(got, want, rtol=1e-5), f"{got} != {want}"


# ── lgamma on the left half of the line ──────────────────────────────────────


def test_lgamma_covers_the_negative_half() -> None:
    """The Lanczos series is valid for ``x >= 0.5`` and nothing said so.

    ``lgamma(-0.5)`` came back NaN where the answer is 1.2655, and
    ``lgamma(1e-30)`` came back inf where it is 69.0776.  Γ has poles at
    the non-positive integers and is finite everywhere between them — the
    left half was not a domain error, it was the half that was missing.
    """
    x = np.array([-2.5, -1.5, -0.5, 1e-30, 1e-6, 0.5, 1.5, 5.0, 100.0])
    expected = [
        -5.624372e-02,
        8.600470e-01,
        1.265512e00,
        6.907755e01,
        1.381551e01,
        5.723649e-01,
        -1.207822e-01,
        3.178054e00,
        3.591342e02,
    ]
    got = np.asarray(lucid.lgamma(lucid.tensor(x)).numpy(), dtype=np.float64)
    assert np.isfinite(got).all(), got
    assert np.allclose(got, expected, rtol=1e-6), got


def test_lgamma_is_infinite_at_its_poles() -> None:
    """Γ has poles at 0 and the negative integers; lgamma is +inf there."""
    got = np.asarray(
        lucid.lgamma(lucid.tensor(np.array([0.0, -1.0, -2.0]))).numpy(),
        dtype=np.float64,
    )
    assert np.isinf(got).all() and (got > 0).all(), got


def test_multigammaln_follows_lgamma_onto_the_left_half() -> None:
    """It sums ``lgamma(a + (1-i)/2)``, so half its terms land there."""
    x = np.array([1e-30, 1e-6, 1.0, 10.0])
    got = np.asarray(sp.multigammaln(lucid.tensor(x), 2).numpy(), dtype=np.float64)
    assert np.allclose(got, [70.91543, 15.65339, 1.14473, 25.06353], rtol=1e-6), got


# ── log1p ────────────────────────────────────────────────────────────────────


def test_log1p_keeps_its_digits_below_the_epsilon() -> None:
    """``log(1 + x)`` loses everything under ~1e-8, which is the point.

    ``1 + 1e-30`` rounds to exactly 1, so the composed form answered 0
    where the value is 1e-30 — and that is the entire reason ``log1p``
    exists as a function of its own rather than as an abbreviation.
    """
    x = np.array([1e-30, 1e-20, 1e-16, 1e-8, 1e-3, 1.0, -0.5])
    got = np.asarray(lucid.log1p(lucid.tensor(x)).numpy(), dtype=np.float64)
    want = np.log1p(x)
    assert np.allclose(got, want, rtol=1e-12), f"{got} != {want}"


def test_log1p_is_negative_infinity_at_minus_one() -> None:
    got = float(np.asarray(lucid.log1p(lucid.tensor(np.array([-1.0]))).numpy())[0])
    assert np.isneginf(got), got


def test_xlog1py_follows_log1p() -> None:
    """It was 0 where the reference is 1e-60, for the same reason."""
    got = np.asarray(
        sp.xlog1py(
            lucid.tensor(np.array([1e-30, 1.0])), lucid.tensor(np.array([1e-30, 2.0]))
        ).numpy(),
        dtype=np.float64,
    )
    assert np.allclose(got, [1e-60, 1.0986122886681098], rtol=1e-9), got
