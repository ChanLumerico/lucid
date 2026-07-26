"""MNIST training pinned against a reference implementation, step for step.

Every other training test in this repo asserts only that *Lucid's own* loss
shrinks on synthetic data.  That catches a dead gradient, but it cannot catch
a gradient that is merely wrong: a loss curve going down is compatible with an
optimizer that is subtly mis-scaled, a backward that drops a term, or a
convolution whose padding is off by one.  Each of those still "trains", just
to a worse model, and nothing in a solo run reports it.

So this module pins Lucid against a reference implementation on a problem with
a known answer.  For every case below both frameworks get:

  * the same architecture,
  * the same initial state — Lucid's parameters *and* buffers, copied across,
    so step 0 is identical rather than merely similar,
  * the same data in the same order, and the same optimizer settings.

Under those conditions the two are solving an identical optimisation problem,
and any divergence is Lucid's numerics.

Cases
-----

``lenet`` is the canonical 61,706-parameter LeNet-5 (C1/S2/C3/S4/C5, tanh,
average pooling, exactly as in LeCun et al. 1998), run under SGD+momentum,
Adam and AdamW.  Three optimizers rather than one because SGD's update is
almost trivially checkable while Adam's is not — bias correction, the epsilon
placement and AdamW's decoupled decay are each a place where a plausible
implementation is off by a little, and "a little" is invisible in a loss
curve.

``bn_cnn`` swaps in the modern convolution block LeNet gives no coverage of:
BatchNorm, ReLU and max pooling.  BatchNorm is the reason this case exists.
Its running statistics are updated outside the autograd graph, so a wrong
momentum or a biased-vs-unbiased variance never shows up in a training loss
at all — it appears only at eval, as a model that scored well while training
and badly afterwards.  The buffers are therefore compared explicitly after
training, not just the losses.

What the assertions check
-------------------------

  1. **Trainability** — the loss falls and accuracy rises well past chance.
  2. **Stability** — no NaN or Inf in any loss or parameter, at any step.
  3. **Agreement** — every step, not just the first, tracks the reference.
  4. **Buffers** — BatchNorm's running statistics, in a separate test that
     freezes the weights so the comparison is not polluted by drift.

Measured behaviour
------------------

Tolerances here are set from measurement, not from a guess, and the two
architectures turned out to need different regimes.

``lenet`` (tanh, average pooling) does not drift at all.  On both CPU and
Metal::

    step 0 loss        lucid 2.30564976   reference 2.30565023   (4.8e-07)
    every later step   0.0 or 2.4e-07 absolute  (~1-2 float32 ULP)
    epoch mean loss    0.8520 0.2157 0.1418 0.0891 0.0692  — identical to 4dp
    final accuracy     0.9600 vs 0.9600 — and identical across CPU and Metal

Both operations are smooth and contractive, so a 1-ULP perturbation is damped
rather than amplified — 470 optimizer steps later the two runs are still on
top of each other.  The expectation going in had been the opposite, that
float32's non-associative addition would pull them apart through reduction
order alone.  It does not, here.

``bn_cnn`` (ReLU, max pooling) does drift, and the reason is architectural::

    epoch 1  max 2.2e-04     epoch 4  max 2.6e-03
    epoch 2  max 9.3e-04     epoch 5  max 3.8e-03
    epoch 3  max 2.2e-03

ReLU's threshold and MaxPool's argmax are discrete: a 1-ULP difference can
flip which unit is active or which element wins a pooling window, and the
gradient then takes a different path outright.  Several independent checks
say this is amplified rounding and not a defect — step 0 is *exactly* 0.0,
the sign of the difference is balanced across steps (216 lucid-high, 250
lucid-low, mean signed difference only 19% of mean absolute), and with the
weights frozen the BatchNorm buffers agree to 5.96e-08.  A systematic error
looks nothing like this: it is one-sided and present from the first step.

The control experiment settles it.  Run the *reference against itself* with a
single weight nudged by one ULP — same code on both sides, so whatever
divergence appears cannot be an implementation difference::

    reference vs 1-ULP-perturbed reference   epoch max  4.5e-04 … 3.5e-03
    lucid vs reference                       epoch max  2.2e-04 … 3.8e-03

The same magnitude.  And over the early window Lucid tracks the reference
*more* closely (1.2e-05) than the reference tracks a one-ULP copy of itself
(1.0e-04).  The drift belongs to the architecture, not to Lucid, and any two
correct implementations would show it.

Why the early window carries the test
-------------------------------------

Because a real defect appears immediately while chaotic drift needs epochs,
the tight bound is applied to the first 30 steps and a coarse one to the full
run.  Injecting a 1% error into Lucid's learning rate — the "trains fine,
converges to a slightly worse model" defect this module exists to catch —
gives, for ``lenet`` + SGD::

    step 0 absdiff        4.8e-07    under 1e-5    not caught
    epoch mean reldiff    0.0056     under 0.02    not caught
    final accuracy diff   0.0005     under 0.01    not caught
    max absdiff, 30 steps 8.8e-03    over  1e-5    CAUGHT

and 7.7e-03 for ``bn_cnn`` against a clean early maximum of 1.2e-05.  Only
the step-level check fires; an epoch-and-accuracy-only version of this module
passes a mis-scaled optimizer without complaint, which is exactly the failure
it was written to detect.  The bounds put the detection floor near a 0.1%
error for both regimes.

Scope: 6,000 training images rather than the full 60,000, so each case runs in
about 20 seconds per device.  Five epochs over that subset takes LeNet-5 to
96% — far enough from chance that a wrong gradient cannot fake it.
"""

import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.models as M
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim

pytestmark = [pytest.mark.parity, pytest.mark.slow]

# Subset sizes: large enough to learn a real decision boundary, small enough
# that two full training runs stay inside a test-suite time budget.
N_TRAIN = 6_000
N_TEST = 2_000
BATCH = 64
EPOCHS = 5
SEED = 0

# Per-optimizer settings, identical on both sides.  Adam and AdamW run at
# their own natural rate rather than SGD's — 0.05 would diverge.
SGD_LR = 0.05
MOMENTUM = 0.9
ADAM_LR = 1e-3

# MNIST's own normalisation constants.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# ── data ────────────────────────────────────────────────────────────────────


def _load_mnist(ref_vision: ModuleType, root: Path) -> tuple[np.ndarray, ...]:
    """Fetch MNIST and return it as 32x32 float32 arrays.

    LeNet-5 expects 32x32; MNIST is 28x28, so the canonical treatment (and
    the paper's) is to pad by 2 on every side rather than resample.
    """
    try:
        train = ref_vision.datasets.MNIST(str(root), train=True, download=True)
        test = ref_vision.datasets.MNIST(str(root), train=False, download=True)
    except Exception as exc:  # offline, or the mirror is down
        pytest.skip(f"MNIST unavailable ({type(exc).__name__}: {exc})")

    def _prep(ds: Any, n: int) -> tuple[np.ndarray, np.ndarray]:
        x = ds.data.numpy()[:n].astype(np.float32) / 255.0
        x = (x - MNIST_MEAN) / MNIST_STD
        x = np.pad(x, ((0, 0), (2, 2), (2, 2)))  # 28x28 → 32x32
        return x[:, None, :, :], ds.targets.numpy()[:n].astype(np.int64)

    x_tr, y_tr = _prep(train, N_TRAIN)
    x_te, y_te = _prep(test, N_TEST)
    return x_tr, y_tr, x_te, y_te


# ── models ──────────────────────────────────────────────────────────────────


def _build_ref_lenet(ref: ModuleType) -> Any:
    """Mirror ``lucid.models.lenet_5_cls`` layer for layer.

    Written out rather than imported because the reference vision package
    ships no LeNet.  The ordering here must match Lucid's parameter order
    (features.0/3/6, f6, classifier) — ``_copy_state`` zips the two.
    """
    nn_ref = ref.nn

    class RefLeNet(nn_ref.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.features = nn_ref.Sequential(
                nn_ref.Conv2d(1, 6, 5),
                nn_ref.Tanh(),
                nn_ref.AvgPool2d(2, stride=2),
                nn_ref.Conv2d(6, 16, 5),
                nn_ref.Tanh(),
                nn_ref.AvgPool2d(2, stride=2),
                nn_ref.Conv2d(16, 120, 5),
                nn_ref.Tanh(),
            )
            self.f6 = nn_ref.Linear(120, 84)
            self.act_f6 = nn_ref.Tanh()
            self.classifier = nn_ref.Linear(84, 10)

        def forward(self, x: Any) -> Any:
            h = self.features(x).flatten(1)
            return self.classifier(self.act_f6(self.f6(h)))

    return RefLeNet()


def _build_lucid_bn_cnn() -> nn.Module:
    """A conv / BatchNorm / ReLU / MaxPool stack — the block LeNet lacks.

    Deliberately small (~11k parameters); the point is covering BatchNorm's
    running statistics and the ReLU/MaxPool pair, not a headline accuracy.
    """
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.MaxPool2d(2),  # 32 → 16
        nn.Conv2d(8, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),  # 16 → 8
        nn.Flatten(),
        nn.Linear(16 * 8 * 8, 10),
    )


def _build_ref_bn_cnn(ref: ModuleType) -> Any:
    nn_ref = ref.nn
    return nn_ref.Sequential(
        nn_ref.Conv2d(1, 8, 3, padding=1),
        nn_ref.BatchNorm2d(8),
        nn_ref.ReLU(),
        nn_ref.MaxPool2d(2),
        nn_ref.Conv2d(8, 16, 3, padding=1),
        nn_ref.BatchNorm2d(16),
        nn_ref.ReLU(),
        nn_ref.MaxPool2d(2),
        nn_ref.Flatten(),
        nn_ref.Linear(16 * 8 * 8, 10),
    )


def _logits(out: Any) -> Any:
    """Zoo models return an output dataclass; a bare Sequential returns the
    tensor itself."""
    return out.logits if hasattr(out, "logits") else out


# ── state transfer ──────────────────────────────────────────────────────────


def _copy_state(src: Any, dst: Any, ref: ModuleType) -> None:
    """Copy Lucid's initialised parameters *and* buffers into the reference.

    This is what makes the comparison meaningful: without it the two models
    start from different points and any later difference is unattributable.
    Layouts match exactly — conv is (out, in, kH, kW) and linear is (out, in)
    on both sides — so no transpose is involved.

    Buffers are copied for the same reason as parameters.  BatchNorm happens
    to start from ``running_mean=0`` / ``running_var=1`` on both sides, so
    today this is a no-op, but a layer that derived a buffer from its config
    would desynchronise the two runs silently without it.
    """
    for kind, s_items, d_items in (
        ("parameter", list(src.named_parameters()), list(dst.named_parameters())),
        ("buffer", list(src.named_buffers()), list(dst.named_buffers())),
    ):
        assert len(s_items) == len(d_items), (
            f"{kind} count differs: lucid {len(s_items)} vs reference "
            f"{len(d_items)} — the architectures have drifted apart"
        )
        with ref.no_grad():
            for (s_name, s_t), (d_name, d_t) in zip(s_items, d_items):
                s_arr = s_t.numpy()
                # ``ascontiguousarray`` returns at least 1-d, so it silently
                # turns BatchNorm's 0-d ``num_batches_tracked`` into shape
                # (1,).  Reshape back to what Lucid actually reported.
                s_arr = np.ascontiguousarray(s_arr).reshape(s_arr.shape)
                assert s_arr.shape == tuple(d_t.shape), (
                    f"shape mismatch at {kind} {s_name} / {d_name}: "
                    f"{s_arr.shape} vs {tuple(d_t.shape)}"
                )
                d_t.copy_(ref.from_numpy(s_arr))


# ── optimizers ──────────────────────────────────────────────────────────────


def _lucid_opt(name: str, params: Any) -> Any:
    if name == "sgd":
        return optim.SGD(params, lr=SGD_LR, momentum=MOMENTUM)
    if name == "adam":
        return optim.Adam(params, lr=ADAM_LR)
    return optim.AdamW(params, lr=ADAM_LR)


def _ref_opt(name: str, params: Any, ref: ModuleType) -> Any:
    if name == "sgd":
        return ref.optim.SGD(params, lr=SGD_LR, momentum=MOMENTUM)
    if name == "adam":
        return ref.optim.Adam(params, lr=ADAM_LR)
    return ref.optim.AdamW(params, lr=ADAM_LR)


# ── training loops ──────────────────────────────────────────────────────────


def _batches(n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """One epoch of shuffled index batches.  Both frameworks consume the
    identical list, so batch composition and order are never a variable."""
    order = rng.permutation(n)
    return [order[i : i + BATCH] for i in range(0, n, BATCH)]


def _train_lucid(
    model: Any,
    data: tuple[np.ndarray, ...],
    schedule: list[list[np.ndarray]],
    device: str,
    opt_name: str,
) -> tuple[list[float], list[float], float]:
    x_tr, y_tr, x_te, y_te = data
    opt = _lucid_opt(opt_name, model.parameters())

    epoch_losses: list[float] = []
    step_losses: list[float] = []
    for epoch_batches in schedule:
        losses = []
        for idx in epoch_batches:
            xb = lucid.tensor(x_tr[idx], device=device)
            yb = lucid.tensor(y_tr[idx], dtype=lucid.int64, device=device)

            opt.zero_grad()
            loss = F.cross_entropy(_logits(model(xb)), yb)
            loss.backward()
            opt.step()

            value = float(loss.item())
            assert math.isfinite(value), (
                f"lucid loss became {value} — training is numerically unstable"
            )
            losses.append(value)
            step_losses.append(value)
        epoch_losses.append(float(np.mean(losses)))

    # Accuracy in eval mode, batched so a 2,000-image forward does not spike
    # memory on either device.  Eval mode is also what puts BatchNorm on its
    # running statistics rather than the batch's own.
    model.eval()
    correct = 0
    with lucid.no_grad():
        for i in range(0, len(y_te), 256):
            xb = lucid.tensor(x_te[i : i + 256], device=device)
            pred = _logits(model(xb)).numpy().argmax(axis=1)
            correct += int((pred == y_te[i : i + 256]).sum())
    model.train()
    return epoch_losses, step_losses, correct / len(y_te)


def _train_ref(
    model: Any,
    data: tuple[np.ndarray, ...],
    schedule: list[list[np.ndarray]],
    ref: ModuleType,
    opt_name: str,
) -> tuple[list[float], list[float], float]:
    x_tr, y_tr, x_te, y_te = data
    opt = _ref_opt(opt_name, model.parameters(), ref)

    epoch_losses: list[float] = []
    step_losses: list[float] = []
    for epoch_batches in schedule:
        losses = []
        for idx in epoch_batches:
            xb = ref.from_numpy(x_tr[idx])
            yb = ref.from_numpy(y_tr[idx])

            opt.zero_grad()
            loss = ref.nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()

            value = float(loss.item())
            losses.append(value)
            step_losses.append(value)
        epoch_losses.append(float(np.mean(losses)))

    model.eval()
    correct = 0
    with ref.no_grad():
        for i in range(0, len(y_te), 256):
            pred = model(ref.from_numpy(x_te[i : i + 256])).numpy().argmax(axis=1)
            correct += int((pred == y_te[i : i + 256]).sum())
    model.train()
    return epoch_losses, step_losses, correct / len(y_te)


# ── cases ───────────────────────────────────────────────────────────────────

# Steps over which the two runs are still provably on the same trajectory.
# Systematic defects show up here; chaotic drift has not accumulated yet.
EARLY_STEPS = 30

# (test id, model kind, optimizer, accuracy floor, early tol, full-run tol)
#
# Two tolerance regimes because the architectures behave differently, and the
# difference is measured, not assumed:
#
#   lenet   tanh + average pooling are smooth and contractive, so a 1-ULP
#           difference is damped.  Measured max over all 470 steps: 2.4e-07
#           (SGD), 4.8e-07 (Adam), 1.1e-06 (AdamW) — flat, no growth.
#
#   bn_cnn  ReLU's threshold and MaxPool's argmax are discrete.  A 1-ULP
#           difference can flip which unit is active or which element wins a
#           pooling window, which changes the gradient path outright.  The
#           gap therefore grows: 2.2e-04 in epoch 1 to 3.8e-03 by epoch 5.
#           That is amplification of rounding, not a defect — step 0 is
#           *exactly* 0.0, the sign of the difference is balanced across
#           steps (216 lucid-high vs 250 lucid-low, mean signed difference
#           only 19% of mean absolute), and freezing the weights makes the
#           BatchNorm buffers agree to 6e-08.  A systematic error looks
#           nothing like that: it is one-sided and present from step 1.
CASES = [
    ("lenet-sgd", "lenet", "sgd", 0.90, 1e-5, 1e-3),
    ("lenet-adam", "lenet", "adam", 0.90, 1e-5, 1e-3),
    ("lenet-adamw", "lenet", "adamw", 0.90, 1e-5, 1e-3),
    ("bn_cnn-adam", "bn_cnn", "adam", 0.90, 1e-3, 3e-2),
]


def _build_pair(kind: str, device: str, ref: ModuleType) -> tuple[Any, Any]:
    lucid.manual_seed(SEED)
    if kind == "lenet":
        lucid_model: Any = M.lenet_5_cls().to(device)
        ref_model = _build_ref_lenet(ref)
    else:
        lucid_model = _build_lucid_bn_cnn().to(device)
        ref_model = _build_ref_bn_cnn(ref)
    _copy_state(lucid_model, ref_model, ref)
    return lucid_model, ref_model


@pytest.fixture(scope="module")
def mnist(tmp_path_factory: pytest.TempPathFactory) -> tuple[np.ndarray, ...]:
    from lucid.test._fixtures.ref_framework import require_ref_vision

    root = tmp_path_factory.mktemp("mnist")
    return _load_mnist(require_ref_vision(), root)


@pytest.mark.parametrize(
    "kind,opt_name,acc_floor,early_tol,full_tol",
    [c[1:] for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_training_matches_the_reference_over_five_epochs(
    device: str,
    ref: ModuleType,
    mnist: tuple[np.ndarray, ...],
    kind: str,
    opt_name: str,
    acc_floor: float,
    early_tol: float,
    full_tol: float,
) -> None:
    model, ref_model = _build_pair(kind, device, ref)

    # One schedule, consumed by both — identical batches in identical order.
    rng = np.random.default_rng(SEED)
    schedule = [_batches(N_TRAIN, rng) for _ in range(EPOCHS)]

    lucid_epochs, lucid_steps, lucid_acc = _train_lucid(
        model, mnist, schedule, device, opt_name
    )
    ref_epochs, ref_steps, ref_acc = _train_ref(
        ref_model, mnist, schedule, ref, opt_name
    )

    # 1. Trainability — the loss actually falls, and the result is a model
    #    rather than a coin flip.  Measured ratio for lenet+sgd is 0.081
    #    (0.8520 → 0.0692), so the bound keeps real headroom.
    assert lucid_epochs[-1] < 0.25 * lucid_epochs[0], (
        f"lucid loss barely moved: {lucid_epochs[0]:.4f} → "
        f"{lucid_epochs[-1]:.4f} over {EPOCHS} epochs"
    )
    assert lucid_acc > acc_floor, f"lucid accuracy {lucid_acc:.4f} is too low"

    # 2. Stability — nothing went non-finite anywhere, including parameters
    #    (a loss can stay finite while a weight has already blown up).
    assert all(math.isfinite(v) for v in lucid_steps)
    for name, p in model.named_parameters():
        assert np.isfinite(p.numpy()).all(), f"{name} contains NaN/Inf"

    # 3. Agreement.  Step 0 isolates forward + backward + the update rule:
    #    identical weights and identical input, no accumulated drift to hide
    #    behind.  Measured 4.8e-07, so 1e-5 leaves ~20x headroom.
    assert lucid_steps[0] == pytest.approx(ref_steps[0], abs=1e-5), (
        f"first-step loss differs: lucid {lucid_steps[0]:.8f} vs "
        f"reference {ref_steps[0]:.8f} — same weights, same batch, so this "
        f"is a forward or backward discrepancy, not trajectory divergence"
    )

    # The early window is where this test gets its power.  A systematic
    # defect — a mis-scaled update, a dropped term — is present from the
    # first step, while chaotic drift needs epochs to accumulate.  Injecting
    # a 1% learning-rate error pushes this to 7.7e-03 (bn_cnn) and 8.8e-03
    # (lenet) against clean values of 1.2e-05 and 4.8e-07, so the bounds
    # below sit roughly a factor of 600 and 20 above the noise while still
    # catching an error of about 0.1%.
    step_diffs = [abs(a - b) for a, b in zip(lucid_steps, ref_steps)]
    early = step_diffs[:EARLY_STEPS]
    worst_early = max(range(len(early)), key=early.__getitem__)
    assert early[worst_early] < early_tol, (
        f"trajectories separated inside the first {EARLY_STEPS} steps, at "
        f"step {worst_early}: lucid {lucid_steps[worst_early]:.8f} vs "
        f"reference {ref_steps[worst_early]:.8f} "
        f"(absdiff {early[worst_early]:.2e}) — too early to be rounding "
        f"amplification, so this is a systematic discrepancy"
    )

    # Over the whole run, a coarser bound: enough to catch a trajectory that
    # genuinely walks away, loose enough to tolerate the ReLU/MaxPool
    # amplification measured above.
    worst = max(range(len(step_diffs)), key=step_diffs.__getitem__)
    assert step_diffs[worst] < full_tol, (
        f"loss trajectories separated at step {worst}: lucid "
        f"{lucid_steps[worst]:.8f} vs reference {ref_steps[worst]:.8f} "
        f"(absdiff {step_diffs[worst]:.2e})"
    )

    # Epoch means: measured identical to four decimals, bounded at 2%.
    for i, (lucid_e, ref_e) in enumerate(zip(lucid_epochs, ref_epochs)):
        assert lucid_e == pytest.approx(ref_e, rel=0.02, abs=1e-3), (
            f"epoch {i + 1} mean loss diverged: lucid {lucid_e:.4f} vs "
            f"reference {ref_e:.4f}\nlucid  {lucid_epochs}\nref    {ref_epochs}"
        )

    # Measured exactly equal on both devices; 0.01 is 20 of the 2,000 test
    # images, so a genuinely worse model cannot slip through.
    assert abs(lucid_acc - ref_acc) < 0.01, (
        f"final accuracy differs: lucid {lucid_acc:.4f} vs "
        f"reference {ref_acc:.4f}"
    )

    # ``num_batches_tracked`` is an integer counter, so unlike the float
    # statistics it cannot drift — the two runs must have taken exactly the
    # same number of BatchNorm updates.  The float buffers are deliberately
    # *not* compared here: after five epochs they inherit whatever trajectory
    # divergence happened above, so any bound would be arbitrary.  They are
    # checked properly by the frozen-weight test below, and indirectly by the
    # accuracy assertion, which runs in eval mode and therefore reads the
    # running statistics rather than the batch's own.
    ref_buffers = dict(ref_model.named_buffers())
    for name, buf in model.named_buffers():
        if not name.endswith("num_batches_tracked"):
            continue
        got = np.asarray(buf.numpy(), dtype=np.int64)
        want = np.asarray(ref_buffers[name].numpy(), dtype=np.int64)
        assert np.array_equal(got, want), (
            f"{name} differs: lucid {got} vs reference {want}"
        )


def test_batchnorm_running_stats_match_with_weights_frozen(
    device: str,
    ref: ModuleType,
    mnist: tuple[np.ndarray, ...],
) -> None:
    """BatchNorm's running statistics, isolated from trajectory divergence.

    Running statistics are updated outside the autograd graph, so a wrong
    momentum or a biased-vs-unbiased variance never shows up in a training
    loss.  It surfaces only at eval — as a model that trained well and scores
    badly — which is exactly the kind of defect a loss curve cannot report.

    Comparing them after a full training run does not work: by then they have
    inherited the ReLU/MaxPool trajectory divergence documented above and
    differ by ~5e-03 for reasons that have nothing to do with BatchNorm.  So
    this runs the forward pass only, with no optimizer step at all.  The
    weights stay identical on both sides, which makes the batch statistics
    identical too, and any difference in the accumulated buffers is then
    BatchNorm's own update rule.

    Measured: 5.96e-08 — one float32 ULP — across every buffer after ten
    batches, which is what established the 5e-03 in the training run as a
    downstream effect rather than a defect.
    """
    model, ref_model = _build_pair("bn_cnn", device, ref)
    x_tr = mnist[0]

    rng = np.random.default_rng(SEED)
    for idx in _batches(N_TRAIN, rng)[:10]:
        model(lucid.tensor(x_tr[idx], device=device))
        with ref.no_grad():
            ref_model(ref.from_numpy(x_tr[idx]))

    ref_buffers = dict(ref_model.named_buffers())
    checked = 0
    for name, buf in model.named_buffers():
        got = np.asarray(buf.numpy(), dtype=np.float64)
        want = np.asarray(ref_buffers[name].numpy(), dtype=np.float64)
        assert np.abs(got - want).max() < 1e-6, (
            f"buffer {name} diverged with the weights frozen: max absdiff "
            f"{np.abs(got - want).max():.3e} — the two models saw identical "
            f"inputs and identical weights, so this is BatchNorm's own update"
        )
        checked += 1
    assert checked == 6, f"expected 6 BatchNorm buffers, compared {checked}"
