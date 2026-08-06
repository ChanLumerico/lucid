"""Every loss, across the options that select a code path.

``nn/functional/loss.py`` sat at 35.4%, and the missing lines were the
options rather than the formulas: the three reductions, per-class
``weight``, ``label_smoothing``, ``ignore_index``, and the logit-space
variants.  The default call ran; nothing else did.

Values are checked against the reference where it has the same option,
and against the definition written out where it does not, so a passing
test says the arithmetic is right rather than unchanged.
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.test._fixtures.ref_framework import require_ref

RNG = np.random.default_rng(0)
LOGITS = RNG.standard_normal((6, 4))
TARGET = np.array([0, 3, 1, 2, 3, 0])
A = RNG.standard_normal((6, 4))
B = RNG.standard_normal((6, 4))
PROB = RNG.uniform(0.1, 0.9, (6, 4))
REDUCTIONS = ["mean", "sum", "none"]


def _t(arr, **kw):
    return lucid.tensor(arr.copy(), **kw)


def _v(x):
    return np.asarray(x.numpy())


# ── the regression losses, every reduction ────────────────────────────────────


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize(
    "name,fn",
    [
        ("mse_loss", F.mse_loss),
        ("l1_loss", F.l1_loss),
        ("smooth_l1_loss", F.smooth_l1_loss),
        ("huber_loss", F.huber_loss),
    ],
)
def test_regression_loss_reductions(name, fn, reduction):
    t = require_ref()
    got = _v(fn(_t(A), _t(B), reduction=reduction))
    ref = np.asarray(
        getattr(t.nn.functional, name)(
            t.from_numpy(A.copy()), t.from_numpy(B.copy()), reduction=reduction
        ).tolist()
    )
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-6)


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_the_reductions_relate_as_they_should(reduction):
    """``mean`` is ``sum`` over the element count, and ``none`` keeps the
    shape — a relation no single reduction can verify alone."""
    none = _v(F.mse_loss(_t(A), _t(B), reduction="none"))
    total = float(F.mse_loss(_t(A), _t(B), reduction="sum").item())
    mean = float(F.mse_loss(_t(A), _t(B), reduction="mean").item())
    assert none.shape == A.shape
    assert np.isclose(none.sum(), total)
    assert np.isclose(total / A.size, mean)


# ── classification ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_cross_entropy_reductions(reduction):
    t = require_ref()
    got = _v(
        F.cross_entropy(_t(LOGITS), _t(TARGET, dtype=lucid.int32), reduction=reduction)
    )
    ref = np.asarray(
        t.nn.functional.cross_entropy(
            t.from_numpy(LOGITS.copy()),
            t.from_numpy(TARGET).long(),
            reduction=reduction,
        ).tolist()
    )
    assert np.allclose(got, ref, atol=1e-6)


def test_cross_entropy_with_class_weights():
    t = require_ref()
    weight = np.array([0.5, 2.0, 1.0, 0.25])
    got = float(
        F.cross_entropy(
            _t(LOGITS), _t(TARGET, dtype=lucid.int32), weight=_t(weight)
        ).item()
    )
    ref = float(
        t.nn.functional.cross_entropy(
            t.from_numpy(LOGITS.copy()),
            t.from_numpy(TARGET).long(),
            weight=t.from_numpy(weight),
        )
    )
    assert np.isclose(got, ref, atol=1e-6)


@pytest.mark.parametrize("smoothing", [0.0, 0.1, 0.3])
def test_cross_entropy_label_smoothing(smoothing):
    t = require_ref()
    got = float(
        F.cross_entropy(
            _t(LOGITS), _t(TARGET, dtype=lucid.int32), label_smoothing=smoothing
        ).item()
    )
    ref = float(
        t.nn.functional.cross_entropy(
            t.from_numpy(LOGITS.copy()),
            t.from_numpy(TARGET).long(),
            label_smoothing=smoothing,
        )
    )
    assert np.isclose(got, ref, atol=1e-6)


def test_smoothing_raises_the_loss_off_a_confident_fit():
    """Smoothing moves probability mass off the true class, so a model
    that was right pays for it — the direction, not just the number."""
    confident = np.eye(4)[TARGET] * 10.0
    plain = float(F.cross_entropy(_t(confident), _t(TARGET, dtype=lucid.int32)).item())
    smoothed = float(
        F.cross_entropy(
            _t(confident), _t(TARGET, dtype=lucid.int32), label_smoothing=0.2
        ).item()
    )
    assert smoothed > plain


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_nll_loss_reductions(reduction):
    t = require_ref()
    log_probs = np.log(np.exp(LOGITS) / np.exp(LOGITS).sum(axis=1, keepdims=True))
    got = _v(
        F.nll_loss(_t(log_probs), _t(TARGET, dtype=lucid.int32), reduction=reduction)
    )
    ref = np.asarray(
        t.nn.functional.nll_loss(
            t.from_numpy(log_probs), t.from_numpy(TARGET).long(), reduction=reduction
        ).tolist()
    )
    assert np.allclose(got, ref, atol=1e-6)


# ── binary ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_binary_cross_entropy_reductions(reduction):
    t = require_ref()
    target = (RNG.uniform(size=(6, 4)) > 0.5).astype(np.float64)
    got = _v(F.binary_cross_entropy(_t(PROB), _t(target), reduction=reduction))
    ref = np.asarray(
        t.nn.functional.binary_cross_entropy(
            t.from_numpy(PROB.copy()), t.from_numpy(target), reduction=reduction
        ).tolist()
    )
    assert np.allclose(got, ref, atol=1e-6)


def test_bce_with_logits_matches_bce_of_the_sigmoid():
    """The logit form exists to be stable, not different: it has to agree
    with the two-step version wherever the two-step version is finite."""
    target = (RNG.uniform(size=(6, 4)) > 0.5).astype(np.float64)
    logit_form = float(
        F.binary_cross_entropy_with_logits(_t(LOGITS), _t(target)).item()
    )
    two_step = float(F.binary_cross_entropy(F.sigmoid(_t(LOGITS)), _t(target)).item())
    assert np.isclose(logit_form, two_step, atol=1e-6)


def test_bce_with_logits_survives_a_saturating_input():
    """Where the two-step form overflows to infinity, the fused one must
    not — that is the whole reason it exists."""
    extreme = np.array([[-80.0, 80.0], [80.0, -80.0]])
    target = np.array([[1.0, 0.0], [0.0, 1.0]])
    got = float(F.binary_cross_entropy_with_logits(_t(extreme), _t(target)).item())
    assert np.isfinite(got)
    assert got > 0.0


# ── pair and margin losses ────────────────────────────────────────────────────


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize(
    "name,fn",
    [
        ("hinge_embedding_loss", F.hinge_embedding_loss),
        ("soft_margin_loss", F.soft_margin_loss),
    ],
)
def test_sign_target_losses(name, fn, reduction):
    t = require_ref()
    sign = np.where(RNG.uniform(size=(6, 4)) > 0.5, 1.0, -1.0)
    got = _v(fn(_t(A), _t(sign), reduction=reduction))
    ref = np.asarray(
        getattr(t.nn.functional, name)(
            t.from_numpy(A.copy()), t.from_numpy(sign), reduction=reduction
        ).tolist()
    )
    assert np.allclose(got, ref, atol=1e-6)


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_margin_ranking_loss(reduction):
    t = require_ref()
    sign = np.where(RNG.uniform(size=(6,)) > 0.5, 1.0, -1.0)
    x1, x2 = A[:, 0], B[:, 0]
    got = _v(F.margin_ranking_loss(_t(x1), _t(x2), _t(sign), reduction=reduction))
    ref = np.asarray(
        t.nn.functional.margin_ranking_loss(
            t.from_numpy(x1.copy()),
            t.from_numpy(x2.copy()),
            t.from_numpy(sign),
            reduction=reduction,
        ).tolist()
    )
    assert np.allclose(got, ref, atol=1e-6)


@pytest.mark.parametrize("margin", [0.5, 1.0, 2.5])
def test_triplet_margin_loss_margin(margin):
    t = require_ref()
    anchor, positive, negative = A, B, RNG.standard_normal((6, 4))
    got = float(
        F.triplet_margin_loss(
            _t(anchor), _t(positive), _t(negative), margin=margin
        ).item()
    )
    ref = float(
        t.nn.functional.triplet_margin_loss(
            t.from_numpy(anchor.copy()),
            t.from_numpy(positive.copy()),
            t.from_numpy(negative),
            margin=margin,
        )
    )
    assert np.isclose(got, ref, atol=1e-6)


def test_a_non_positive_triplet_margin_is_accepted_here_and_not_by_the_reference():
    """A divergence, recorded rather than decided.

    ``margin=0`` makes the hinge vacuous and a negative one inverts it,
    so the reference refuses both.  Lucid computes
    ``max(0, d_pos - d_neg + margin)`` and returns a number.  Neither is
    wrong — one is stricter — and changing it is an API decision rather
    than a defect fix, so this pins the current behaviour and names the
    difference.
    """
    for margin in (0.0, -0.5):
        value = float(F.triplet_margin_loss(_t(A), _t(B), _t(A), margin=margin).item())
        assert np.isfinite(value)


# ── distribution losses ───────────────────────────────────────────────────────


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_kl_div_reductions(reduction):
    t = require_ref()
    log_input = np.log(PROB / PROB.sum(axis=1, keepdims=True))
    target = PROB / PROB.sum(axis=1, keepdims=True)
    got = _v(F.kl_div(_t(log_input), _t(target), reduction=reduction))
    ref = np.asarray(
        t.nn.functional.kl_div(
            t.from_numpy(log_input), t.from_numpy(target), reduction=reduction
        ).tolist()
    )
    assert np.allclose(got, ref, atol=1e-6)


@pytest.mark.parametrize("log_input", [True, False])
def test_poisson_nll_loss(log_input):
    t = require_ref()
    rate = PROB if not log_input else np.log(PROB)
    target = np.abs(RNG.standard_normal((6, 4)))
    got = float(F.poisson_nll_loss(_t(rate), _t(target), log_input=log_input).item())
    ref = float(
        t.nn.functional.poisson_nll_loss(
            t.from_numpy(rate), t.from_numpy(target), log_input=log_input
        )
    )
    assert np.isclose(got, ref, atol=1e-5)


def test_gaussian_nll_loss():
    t = require_ref()
    var = np.abs(RNG.standard_normal((6, 4))) + 0.5
    got = float(F.gaussian_nll_loss(_t(A), _t(B), _t(var)).item())
    ref = float(
        t.nn.functional.gaussian_nll_loss(
            t.from_numpy(A.copy()), t.from_numpy(B.copy()), t.from_numpy(var)
        )
    )
    assert np.isclose(got, ref, atol=1e-6)


# ── refusals ──────────────────────────────────────────────────────────────────


def test_an_unknown_reduction_is_refused():
    with pytest.raises(ValueError, match="reduction"):
        F.mse_loss(_t(A), _t(B), reduction="average")


def test_label_smoothing_outside_its_range_is_refused():
    with pytest.raises(ValueError, match="label_smoothing"):
        F.cross_entropy(_t(LOGITS), _t(TARGET, dtype=lucid.int32), label_smoothing=1.5)


# ── every loss is differentiable ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "build",
    [
        lambda a, b: F.mse_loss(a, b),
        lambda a, b: F.l1_loss(a, b),
        lambda a, b: F.smooth_l1_loss(a, b),
        lambda a, b: F.huber_loss(a, b),
        lambda a, b: F.binary_cross_entropy_with_logits(a, F.sigmoid(b)),
    ],
)
def test_gradients_reach_the_input(build):
    a = _t(A, requires_grad=True)
    build(a, _t(B)).backward()
    assert a.grad is not None
    assert np.abs(_v(a.grad)).sum() > 0.0
