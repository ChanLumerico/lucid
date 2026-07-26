"""LeNet-5 on MNIST — the smallest honest end-to-end proof that Lucid learns.

Every other training test in this repo asserts only that *Lucid's own* loss
shrinks on synthetic data.  That catches a dead gradient, but it cannot catch
a gradient that is merely wrong: a loss curve going down is compatible with an
optimizer that is subtly mis-scaled, a backward that drops a term, or a
convolution whose padding is off by one.  Each of those still "trains", just
to a worse model, and nothing in a solo run reports it.

So this test pins Lucid against a reference implementation on a problem with
a known answer.  Both frameworks get:

  * the same architecture (the canonical 61,706-parameter LeNet-5 — C1/S2/C3/
    S4/C5 with tanh and average pooling, exactly as in LeCun et al. 1998),
  * the same initial weights — Lucid's, copied across, so step 0 is identical
    rather than merely similar,
  * the same data in the same order, and the same optimizer settings.

Under those conditions the two are solving an identical optimisation problem,
and any divergence is Lucid's numerics.  What the assertions then check:

  1. **Trainability** — the loss falls and accuracy rises well past chance.
  2. **Stability** — no NaN or Inf in any loss, parameter or gradient, at any
     step of any epoch.
  3. **Agreement** — every step, not just the first, tracks the reference.

On tolerances — these are set from measurement, not from a guess.  The
expectation going in was that only step 0 could be held tight, since the two
runs are separate trajectories through a non-convex landscape and float32
addition is not associative, so reduction order alone should pull them apart.
That turned out to be wrong here.  Measured over the full run, on both CPU
and Metal::

    step 0 loss        lucid 2.30564976   reference 2.30565023   (4.8e-07)
    every later step   0.0 or 2.4e-07 absolute  (~1-2 float32 ULP)
    epoch mean loss    0.8520 0.2157 0.1418 0.0891 0.0692  — identical to 4dp
    final accuracy     0.9600 vs 0.9600 — and identical across CPU and Metal

The trajectories do not diverge because tanh saturates, which is contractive:
it damps a 1-ULP perturbation instead of amplifying it.  So the bounds below
assert per-step agreement across all 470 steps, which is a far stronger claim
than an epoch-level one — with roughly three orders of magnitude of headroom
over the observed values, since a single machine and one reference version is
thin evidence for pinning a bound at what it happens to measure today.

That per-step bound is doing the work.  Injecting a 1% error into Lucid's
learning rate — the "trains fine, converges to a slightly worse model" defect
this test exists to catch — gives::

    step 0 absdiff        4.8e-07    under 1e-5    not caught
    epoch mean reldiff    0.0056     under 0.02    not caught
    final accuracy diff   0.0005     under 0.01    not caught
    max per-step absdiff  1.3e-02    over  1e-3    CAUGHT at step 26

Only the per-step check fires.  An epoch-and-accuracy-only version of this
test passes a mis-scaled optimizer without complaint, which is exactly the
failure it was written to detect.  The 1e-3 bound puts the detection floor at
roughly a 0.1% error.

Scope: 6,000 training images rather than the full 60,000, so the whole test
runs in about 45 seconds.  Five epochs over that subset takes LeNet-5 to 96%
— far enough from chance that a wrong gradient cannot fake it.
"""

import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.models as M
import lucid.nn.functional as F
import lucid.optim as optim

pytestmark = [pytest.mark.parity, pytest.mark.slow]

# Subset sizes: large enough to learn a real decision boundary, small enough
# that two full training runs stay inside a test-suite time budget.
N_TRAIN = 6_000
N_TEST = 2_000
BATCH = 64
EPOCHS = 5
LR = 0.05
MOMENTUM = 0.9
SEED = 0

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


# ── the reference model ─────────────────────────────────────────────────────


def _build_ref_lenet(ref: ModuleType) -> Any:
    """Mirror ``lucid.models.lenet_5_cls`` layer for layer.

    Written out rather than imported because the reference vision package
    ships no LeNet.  The ordering here must match Lucid's parameter order
    (features.0/3/6, f6, classifier) — ``_copy_weights`` zips the two.
    """
    nn = ref.nn

    class RefLeNet(nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 6, 5),
                nn.Tanh(),
                nn.AvgPool2d(2, stride=2),
                nn.Conv2d(6, 16, 5),
                nn.Tanh(),
                nn.AvgPool2d(2, stride=2),
                nn.Conv2d(16, 120, 5),
                nn.Tanh(),
            )
            self.f6 = nn.Linear(120, 84)
            self.act_f6 = nn.Tanh()
            self.classifier = nn.Linear(84, 10)

        def forward(self, x: Any) -> Any:
            h = self.features(x).flatten(1)
            return self.classifier(self.act_f6(self.f6(h)))

    return RefLeNet()


def _copy_weights(src: Any, dst: Any, ref: ModuleType) -> None:
    """Copy Lucid's initialised weights into the reference model.

    This is what makes the comparison meaningful: without it the two models
    start from different points and any later difference is unattributable.
    Layouts match exactly — conv is (out, in, kH, kW) and linear is
    (out, in) on both sides — so no transpose is involved.
    """
    src_params = list(src.named_parameters())
    dst_params = list(dst.named_parameters())
    assert len(src_params) == len(dst_params), (
        f"parameter count differs: lucid {len(src_params)} vs "
        f"reference {len(dst_params)} — the architectures have drifted apart"
    )
    with ref.no_grad():
        for (s_name, s_p), (d_name, d_p) in zip(src_params, dst_params):
            s_arr = s_p.numpy()
            assert s_arr.shape == tuple(d_p.shape), (
                f"shape mismatch at {s_name} / {d_name}: "
                f"{s_arr.shape} vs {tuple(d_p.shape)}"
            )
            d_p.copy_(ref.from_numpy(s_arr))


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
) -> tuple[list[float], list[float], float]:
    x_tr, y_tr, x_te, y_te = data
    opt = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    epoch_losses: list[float] = []
    step_losses: list[float] = []
    for epoch_batches in schedule:
        losses = []
        for idx in epoch_batches:
            xb = lucid.tensor(x_tr[idx], device=device)
            yb = lucid.tensor(y_tr[idx], dtype=lucid.int64, device=device)

            opt.zero_grad()
            loss = F.cross_entropy(model(xb).logits, yb)
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
    # memory on either device.
    model.eval()
    correct = 0
    with lucid.no_grad():
        for i in range(0, len(y_te), 256):
            xb = lucid.tensor(x_te[i : i + 256], device=device)
            pred = model(xb).logits.numpy().argmax(axis=1)
            correct += int((pred == y_te[i : i + 256]).sum())
    model.train()
    return epoch_losses, step_losses, correct / len(y_te)


def _train_ref(
    model: Any,
    data: tuple[np.ndarray, ...],
    schedule: list[list[np.ndarray]],
    ref: ModuleType,
) -> tuple[list[float], list[float], float]:
    x_tr, y_tr, x_te, y_te = data
    opt = ref.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

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


# ── the test ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def mnist(tmp_path_factory: pytest.TempPathFactory) -> tuple[np.ndarray, ...]:
    from lucid.test._fixtures.ref_framework import require_ref_vision

    root = tmp_path_factory.mktemp("mnist")
    return _load_mnist(require_ref_vision(), root)


def test_lenet_mnist_matches_the_reference_over_five_epochs(
    device: str,
    ref: ModuleType,
    mnist: tuple[np.ndarray, ...],
) -> None:
    lucid.manual_seed(SEED)
    model = M.lenet_5_cls().to(device)

    ref_model = _build_ref_lenet(ref)
    _copy_weights(model, ref_model, ref)

    # One schedule, consumed by both — identical batches in identical order.
    rng = np.random.default_rng(SEED)
    schedule = [_batches(N_TRAIN, rng) for _ in range(EPOCHS)]

    lucid_epochs, lucid_steps, lucid_acc = _train_lucid(
        model, mnist, schedule, device
    )
    ref_epochs, ref_steps, ref_acc = _train_ref(ref_model, mnist, schedule, ref)

    # 1. Trainability — the loss actually falls, and the result is a model
    #    rather than a coin flip.  Measured ratio is 0.081 (0.8520 → 0.0692)
    #    and accuracy 0.96, so both bounds keep real headroom.
    assert lucid_epochs[-1] < 0.25 * lucid_epochs[0], (
        f"lucid loss barely moved: {lucid_epochs[0]:.4f} → "
        f"{lucid_epochs[-1]:.4f} over {EPOCHS} epochs"
    )
    assert lucid_acc > 0.90, f"lucid accuracy {lucid_acc:.4f} is too low"

    # 2. Stability — nothing went non-finite anywhere, including parameters
    #    (a loss can stay finite while a weight has already blown up).
    assert all(math.isfinite(v) for v in lucid_steps)
    for name, p in model.named_parameters():
        arr = p.numpy()
        assert np.isfinite(arr).all(), f"{name} contains NaN/Inf after training"

    # 3. Agreement.  Step 0 isolates forward + backward + the update rule:
    #    identical weights and identical input, no accumulated drift to hide
    #    behind.  Measured 4.8e-07, so 1e-5 leaves ~20x headroom.
    assert lucid_steps[0] == pytest.approx(ref_steps[0], abs=1e-5), (
        f"first-step loss differs: lucid {lucid_steps[0]:.8f} vs "
        f"reference {ref_steps[0]:.8f} — same weights, same batch, so this "
        f"is a forward or backward discrepancy, not trajectory divergence"
    )

    # And every step after it.  Measured max is 2.4e-07 (~2 float32 ULP) over
    # all 470 steps; the bound is deliberately far looser than that, because
    # a different reference version or host could legitimately shift reduction
    # order.  Even so it is ~100x tighter than any real defect would produce.
    step_diffs = [abs(a - b) for a, b in zip(lucid_steps, ref_steps)]
    worst = max(range(len(step_diffs)), key=step_diffs.__getitem__)
    assert step_diffs[worst] < 1e-3, (
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
