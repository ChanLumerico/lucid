"""Dropout policy in compile mode.

Contract (matches the emitter logic in
``lucid/_C/compile/OpEmitters/nn/Dropout.mm``):

  * **eval mode** or **p == 0** — identity passthrough.  Bit-exact with
    eager because both reduce to a clone.
  * **training mode** with **p > 0** — emitter returns nullptr, the
    builder aborts and the trace is marked eager-only.  The result
    is still correct (it's just eager); the user's mental model of
    "compile mode = compiled" is wrong for that signature.

The policy is deliberate: the RNG path is deterministic-per-executable
(see ``OpEmitters/special/Random.mm``).  If we ran dropout through
that path the model would apply the *same* mask every step — silently
breaking dropout's regularising effect.  Eager fallback preserves
training correctness at the cost of compile-mode speedup for these
signatures.

Tests pin both halves of the contract — eval-mode dropout compiles
cleanly, training-mode dropout falls back without losing the math.
"""

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid.test.unit.compile._helpers import (
    COMPILE_DEVICE,
    assert_compile_parity,
    metal_tensor,
)


def _dropout_model(p: float) -> nn.Module:
    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.drop = nn.Dropout(p=p)
            self.fc2 = nn.Linear(16, 4)

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            return self.fc2(self.drop(self.fc1(x).relu()))

    return _M().to(COMPILE_DEVICE)


def test_dropout_eval_mode_compiles_clean() -> None:
    """In ``.eval()`` dropout is identity; trace must compile fully."""
    model = _dropout_model(p=0.5)
    model.eval()
    x = metal_tensor(4, 8)
    assert_compile_parity(model, x, atol=1e-4, rtol=1e-5)

    # Make sure no eager-only fallback was inserted.
    cm = lucid.compile(model.eval())
    cm(x)
    info = cm.cache_info()
    assert info["entries"] >= 1, "dropout-in-eval should compile"


def test_dropout_zero_prob_compiles_clean() -> None:
    """``p == 0`` is identity even in training mode — must compile."""
    model = _dropout_model(p=0.0)
    model.train()
    x = metal_tensor(4, 8)
    # Compile mode + eager should both be deterministic + identical.
    assert_compile_parity(model, x, atol=1e-4, rtol=1e-5)


def test_dropout_training_falls_back_to_eager() -> None:
    """``p > 0`` in training mode triggers eager fallback (RNG-driven mask).

    The compiled cache should NOT carry a successful executable for
    this signature — it should be in the eager-only set so future
    calls skip the recompile attempt.
    """
    model = _dropout_model(p=0.5)
    model.train()
    x = metal_tensor(4, 8)
    cm = lucid.compile(model)
    cm(x)
    cm(x)  # second call to ensure the fallback set is stable
    info = cm.cache_info()
    # Either nothing was cached (trace aborted before producing a
    # successful executable) or the signature was blacklisted as
    # eager-only.  Both indicate correct fallback behaviour.
    assert info["entries"] == 0 or len(info["eager_only"]) > 0, (
        f"expected eager fallback for training-mode dropout; " f"cache_info={info}"
    )


def test_dropout_training_produces_random_outputs() -> None:
    """Training-mode dropout must still randomise (even on eager path).

    Two calls with the same input must produce *different* outputs —
    if the compile pipeline silently routed training-mode dropout
    through the deterministic stateless RNG path, both calls would
    return identical tensors and this assertion would fail.
    """
    model = _dropout_model(p=0.5)
    model.train()
    cm = lucid.compile(model)
    x = metal_tensor(64, 8)  # large enough that random masks differ
    a = cm(x).detach().clone()
    b = cm(x).detach().clone()
    diff = float((a - b).abs().max().item())
    assert diff > 0.0, (
        "training-mode dropout produced identical outputs across two "
        "calls — RNG state is stuck.  Either the eager fallback "
        "regressed or compile took a deterministic RNG path."
    )
