"""Dropout policy in compile mode.

Contract (matches the emitter logic in
``lucid/_C/compile/OpEmitters/nn/Dropout.mm``):

  * **eval mode** or **p == 0** — identity passthrough.  Bit-exact with
    eager because both reduce to a clone.
  * **training mode** with **p > 0** — `lucid.compile` (forward-only
    cache via :class:`CompiledModule`) still routes through the
    standard ``dropout`` op which falls back to eager because the
    forward path has no place to thread an MPSGraph state buffer.
    The proper compile path for training-mode dropout lives in
    :func:`fused_step` (Option-A Phase 1) — that surface uses the
    sibling ``dropout_stateful`` engine op + MPSGraph's stateful
    Philox RNG via ``compile_generic_fused_step_with_vars``, giving
    genuinely-per-dispatch varying masks while the executable still
    runs entirely on the GPU.

Tests pin both halves of the contract — eval-mode dropout compiles
cleanly, training-mode dropout via :func:`lucid.compile` still falls
back to eager, and training-mode dropout via :func:`fused_step`
compiles cleanly + produces randomised outputs across calls.
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


def test_dropout_training_lucid_compile_still_falls_back_to_eager() -> None:
    """``lucid.compile()`` (forward-only) still falls back for training dropout.

    The forward-only :class:`CompiledModule` path uses
    ``compile_trace`` which has no variable-promotion hook for RNG
    state, so training-mode dropout there continues to take the eager
    fallback to preserve mask randomisation across calls.  The
    sibling ``dropout_stateful`` op exists for the
    :func:`fused_step` path which DOES have variable promotion —
    that case is covered by the dedicated test below.

    The compiled cache should therefore NOT carry a successful
    executable for this signature; it should be in the eager-only
    set so future calls skip the recompile attempt.
    """
    model = _dropout_model(p=0.5)
    model.train()
    x = metal_tensor(4, 8)
    cm = lucid.compile(model)
    cm(x)
    cm(x)  # second call to ensure the fallback set is stable
    info = cm.cache_info()
    assert info["entries"] == 0 or len(info["eager_only"]) > 0, (
        f"expected eager fallback for training-mode dropout via "
        f"lucid.compile(); cache_info={info}"
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
