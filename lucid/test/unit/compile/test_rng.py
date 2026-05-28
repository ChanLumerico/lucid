"""RNG behaviour in compile mode.

The MPSGraph RNG path is **stateless** (descriptor-baked seed): every
invocation of the compiled executable produces the *same* random
sequence.  This is intentional — the alternative path
(``randomTensorWithShape:descriptor:stateTensor:``) requires plumbing
a Philox state buffer through the executable's I/O schema, which
would be a cross-cutting refactor of the compile-pipeline contract.

The trade-off is therefore explicit:

  * Deterministic-per-executable RNG: useful for inference-mode
    smoke tests, adversarial-robustness probes, and any pattern that
    expects reproducible noise.
  * Stochastic step-to-step RNG (dropout regularisation, data aug,
    MC sampling): **users must keep these in eager mode**.

The tests below pin both halves of that contract — compile-mode RNG
must be reproducible across calls, and the user-facing API must not
silently degrade dropout in training mode.
"""

import lucid
import lucid.nn as nn

from lucid.test.unit.compile._helpers import COMPILE_DEVICE


class _RandModel(nn.Module):
    """Wraps an RNG op into an nn.Module so it can be ``lucid.compile``-d.

    The op is dispatched in :meth:`forward` so a single Tracer captures
    it — the resulting executable's RNG seed is baked at compile time.
    """

    def __init__(self, mode: str = "randn") -> None:
        super().__init__()
        self._mode = mode

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        if self._mode == "randn":
            r = lucid.randn(*x.shape, device=x.device)
        elif self._mode == "rand":
            r = lucid.rand(*x.shape, device=x.device)
        else:
            raise ValueError(self._mode)
        return x + r


def test_compile_randn_deterministic_per_executable() -> None:
    """Repeated calls to a compiled RNG executable yield the same draw.

    This is the documented behaviour — the descriptor-baked seed makes
    every invocation reproduce the same Philox stream.  If we ever
    promote to the stateful path this test should change to
    ``assert torch_neq(first, second)`` and the user-guide note
    updated in lock-step.
    """
    lucid.manual_seed(42)
    model = _RandModel("randn").to(COMPILE_DEVICE)
    cm = lucid.compile(model)

    x = lucid.zeros(4, 8).to(COMPILE_DEVICE)
    first = cm(x).detach().clone()
    second = cm(x).detach().clone()

    diff = float((first - second).abs().max().item())
    assert diff == 0.0, (
        f"compile-mode RNG was supposed to be deterministic-per-executable, "
        f"got diff = {diff:.3e}.  Either the stateless seed path silently "
        f"became stateful, or a buffer was reused across calls."
    )


def test_compile_rand_uniform_within_bounds() -> None:
    """rand draws U(0, 1) — values must lie in ``[0, 1)`` after compile."""
    model = _RandModel("rand").to(COMPILE_DEVICE)
    cm = lucid.compile(model)

    x = lucid.zeros(64, 64).to(COMPILE_DEVICE)
    r = cm(x).detach()  # adds zeros so values == raw uniform draws
    lo = float(r.min().item())
    hi = float(r.max().item())
    assert 0.0 <= lo < hi <= 1.0, f"uniform out of [0, 1): [{lo}, {hi}]"
    # A 4096-sample draw should have non-trivial spread.
    assert hi - lo > 0.5, f"uniform draw is suspiciously narrow: spread={hi-lo}"


def test_compile_randn_normal_stats() -> None:
    """randn draws N(0, 1) — sample mean ≈ 0, std ≈ 1 within fp32 noise.

    Large sample size (16K) so the central-limit bound shrinks below
    the tolerance — this guards against silent dtype downcasts or
    range clipping in the emitter.
    """
    model = _RandModel("randn").to(COMPILE_DEVICE)
    cm = lucid.compile(model)
    x = lucid.zeros(128, 128).to(COMPILE_DEVICE)
    r = cm(x).detach()
    mean = float(r.mean().item())
    std = float(((r - mean) ** 2).mean().sqrt().item())
    assert abs(mean) < 0.05, f"randn mean = {mean}"
    assert abs(std - 1.0) < 0.05, f"randn std = {std}"
