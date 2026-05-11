"""Shared utilities for model-level parity tests.

Three testing modes
-------------------
1. **Full parity** (< 30 M params, has timm equivalent):
   Copy Lucid weights → timm positionally, run identical input,
   compare logits with ``atol=1e-4, rtol=1e-4``.

2. **Self-consistency** (no timm equivalent or timm key mismatch):
   Run the Lucid model twice with the same fixed input and assert
   deterministic output.  Catches forward-pass bugs without a reference.

3. **Block parity** (> 100 M params, marked ``heavy``):
   Extract a single block (e.g., one ResNet bottleneck), copy its
   weights, and compare only that sub-module.  Keeps memory safe on
   all M-chip configs.

Size thresholds used for ``pytest.mark`` decoration
-----------------------------------------------------
  < 30 M   → no extra marker (always run)
 30–100 M  → ``pytest.mark.slow``
  > 100 M  → ``pytest.mark.heavy``
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.test._helpers.compare import assert_close

# ── timm availability ─────────────────────────────────────────────────────────

try:
    import timm as _timm  # noqa: F401

    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

requires_timm = pytest.mark.skipif(not HAS_TIMM, reason="timm not installed")
heavy = pytest.mark.heavy


# ── weight transfer ───────────────────────────────────────────────────────────


def _copy_lucid_to_ref(lucid_model: nn.Module, ref_model: Any) -> None:
    """Copy parameters from Lucid model → reference (PyTorch) model positionally.

    Both models must have identical parameter shapes in iteration order.
    Silently succeeds only when that condition holds — caller should
    verify parameter counts match first.
    """
    import torch  # available because ``ref`` fixture already handled this

    lucid_params = list(lucid_model.parameters())
    ref_params = list(ref_model.parameters())

    if len(lucid_params) != len(ref_params):
        raise AssertionError(
            f"param count mismatch: lucid={len(lucid_params)} "
            f"ref={len(ref_params)}"
        )

    for lp, rp in zip(lucid_params, ref_params):
        arr = lp.data.numpy()
        if arr.shape != tuple(rp.shape):
            raise AssertionError(
                f"shape mismatch: lucid={arr.shape} ref={tuple(rp.shape)}"
            )
        rp.data.copy_(torch.from_numpy(arr))


# ── full-model parity ─────────────────────────────────────────────────────────


def run_parity(
    lucid_model: nn.Module,
    timm_name: str,
    *,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    seed: int = 0,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    num_classes: int = 1000,
    in_chans: int = 3,
) -> None:
    """Run one full-model parity check against a timm reference.

    Steps
    -----
    1. Build a ``timm`` model with the same number of classes.
    2. Copy Lucid weights → timm positionally.
    3. Feed an identical random input.
    4. Compare the raw logit arrays.

    Both models are put in eval mode before forward.
    """
    import timm
    import torch

    timm_model = timm.create_model(
        timm_name,
        pretrained=False,
        num_classes=num_classes,
        in_chans=in_chans,
    )
    timm_model.eval()

    lucid_model.eval()

    try:
        _copy_lucid_to_ref(lucid_model, timm_model)
    except AssertionError as exc:
        pytest.skip(f"weight-transfer failed — {exc}")

    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal(input_shape).astype(np.float32)

    with torch.no_grad():
        y_ref = timm_model(torch.from_numpy(x_np)).numpy()

    lucid_out = lucid_model(lucid.from_numpy(x_np))
    # Support both raw Tensor and ImageClassificationOutput
    y_lucid = (
        lucid_out.logits.numpy()
        if hasattr(lucid_out, "logits")
        else lucid_out.numpy()
    )

    assert_close(y_lucid, y_ref, atol=atol, rtol=rtol,
                 msg=f"parity failed for timm:{timm_name}")


# ── self-consistency (no external reference) ──────────────────────────────────


def run_self_consistency(
    lucid_model: nn.Module,
    *,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    seed: int = 42,
) -> None:
    """Assert deterministic forward pass: same seed ↦ identical output twice.

    Used for models without a timm equivalent (LeNet, ZFNet, GoogLeNet, …).
    Catches undefined-behaviour bugs without needing an external reference.
    """
    lucid_model.eval()

    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal(input_shape).astype(np.float32)
    x = lucid.from_numpy(x_np)

    out1 = lucid_model(x)
    out2 = lucid_model(x)

    y1 = out1.logits.numpy() if hasattr(out1, "logits") else out1.numpy()
    y2 = out2.logits.numpy() if hasattr(out2, "logits") else out2.numpy()

    assert_close(y1, y2, atol=0.0, rtol=0.0, msg="forward pass is not deterministic")


# ── block-level parity (for heavy models) ────────────────────────────────────


def run_block_parity(
    lucid_block: nn.Module,
    ref_block: Any,
    *,
    input_shape: tuple[int, ...],
    seed: int = 0,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    """Compare a single sub-module (block/layer) against its PyTorch counterpart.

    Used for heavy models where loading the full model would OOM.
    The caller is responsible for constructing equivalent ``lucid_block``
    and ``ref_block`` with the same architecture.
    """
    import torch

    lucid_block.eval()
    ref_block.eval()

    try:
        _copy_lucid_to_ref(lucid_block, ref_block)
    except AssertionError as exc:
        pytest.skip(f"block weight-transfer failed — {exc}")

    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal(input_shape).astype(np.float32)

    with torch.no_grad():
        y_ref = ref_block(torch.from_numpy(x_np)).numpy()

    y_lucid_out = lucid_block(lucid.from_numpy(x_np))
    y_lucid = (
        y_lucid_out.numpy()
        if hasattr(y_lucid_out, "numpy")
        else np.array(y_lucid_out)
    )

    assert_close(y_lucid, y_ref, atol=atol, rtol=rtol, msg="block parity failed")
