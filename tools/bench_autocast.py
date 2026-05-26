"""Focused autocast (F16) vs F32 bench for fused_step.

Companion to ``tools/bench_compile_vs_eager.py``.  After the X4.4
autocast threading + BN F16 emit + Arith mixed-dtype cast fixes
landed (2026-05-27), `fused_step` + `with autocast(F16)` works on
LN-based / MLP / BN1d / BN2d-Conv workloads.

This script measures the F16 speedup on those workloads vs the F32
baseline.  Skips workloads that hit the still-open
ResNet-residual + autocast bug (BN VJP needs mixed-dtype reconciliation —
documented in [[arch-compile-parity-vs-torch-mps]] §6).

Usage::

    python3 -m tools.bench_autocast
    python3 -m tools.bench_autocast --csv ~/Desktop/bench_autocast.csv

Output: CSV with columns
    workload, precision, batch, fused_step_warm_ms, speedup_vs_f32, note
"""

import argparse
import csv
import os
import statistics
import sys
import time

import lucid
import lucid.amp as amp
import lucid.metal as metal
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import fused_step


COMPILE_DEVICE = "metal"


# ── Workloads that work with fused_step + autocast ──────────────────


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc3(self.fc2(self.fc1(x).relu()).relu())


class _LNMLP(nn.Module):
    """LayerNorm-based MLP — transformer-shape pattern."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc3(self.ln2(self.fc2(self.ln1(self.fc1(x))).relu()))


class _BN1DMLP(nn.Module):
    """BatchNorm1d-MLP — exercises the BN F16 fix."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc3(self.bn2(self.fc2(self.bn1(self.fc1(x)).relu())))


class _GPT2Block(nn.Module):
    """Single transformer block — d=768, 12 heads, seq=128.

    Same shape as bench_compile_vs_eager.py's GPT-2 workload.
    """

    def __init__(self) -> None:
        super().__init__()
        d_model = 768
        n_heads = 12
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + a
        return x + self.mlp(self.ln2(x))


WORKLOADS = [
    ("mlp", _MLP, lambda bs: lucid.randn(bs, 64), lambda bs: lucid.randn(bs, 10)),
    ("ln_mlp", _LNMLP, lambda bs: lucid.randn(bs, 64), lambda bs: lucid.randn(bs, 10)),
    ("bn1d_mlp", _BN1DMLP, lambda bs: lucid.randn(bs, 64), lambda bs: lucid.randn(bs, 10)),
    # gpt2_block + autocast currently fails on GELU's F32 constants
    # vs F16 activations (constantWithScalar:dataType: in the GELU
    # emitter defaults to F32 without matching the input).  Tracked
    # alongside the BN VJP mixed-dtype issue in
    # arch-compile-parity-vs-torch-mps §6.  Re-enable when the
    # systemic mixed-dtype fix lands.
]


def _loss_fn(pred: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
    """MSE in F32 — cast pred up from F16 to match target dtype."""
    return F.mse_loss(pred.to(target.dtype), target)


def _bench_fused_step(
    mk_model, mk_input, mk_target, bs: int, *, autocast: bool, n_iter: int = 10
) -> tuple[float, float, float]:
    """Run fused_step N times, return (cold_ms, warm_median_ms, p95_ms)."""
    lucid.manual_seed(0)
    model = mk_model().to(COMPILE_DEVICE)
    model.train()
    inputs = (mk_input(bs).to(COMPILE_DEVICE),)
    target = mk_target(bs).to(COMPILE_DEVICE)
    opt = optim.SGD(list(model.parameters()), lr=1e-3)
    step = fused_step(model, _loss_fn, opt)

    def call() -> object:
        if autocast:
            with amp.autocast(dtype=lucid.float16):
                loss = step(*inputs, target)
        else:
            loss = step(*inputs, target)
        # Force eval to drain MLX deferred work — see
        # perf-compile-vs-eager-2026-05-26 §"Open questions" §3.
        _ = float(loss.item())
        return loss

    metal.synchronize()
    t0 = time.perf_counter()
    call()
    metal.synchronize()
    cold = (time.perf_counter() - t0) * 1000.0

    # warmup
    for _ in range(3):
        call()
    metal.synchronize()

    samples: list[float] = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        call()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    n = len(samples)
    warm = statistics.median(samples)
    p95 = samples[min(n - 1, int(n * 0.95))]
    return cold, warm, p95


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=str, default="8,32,128", help="Comma-separated BS list")
    ap.add_argument("--csv", type=str, default="out/bench_autocast.csv")
    ap.add_argument("--iter", type=int, default=10)
    args = ap.parse_args()

    batches = [int(b) for b in args.batch.split(",")]
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)

    rows: list[dict[str, object]] = []

    print(f"# Autocast bench — {args.csv}")
    print(f"# {len(WORKLOADS)} workloads × {len(batches)} batches × 2 precisions = "
          f"{len(WORKLOADS) * len(batches) * 2} rows")

    for name, mk_model, mk_input, mk_target in WORKLOADS:
        for bs in batches:
            for prec_label, autocast in [("f32", False), ("autocast_f16", True)]:
                sys.stdout.write(f"  {name:14s} {prec_label:14s} BS={bs:4d}  ...  ")
                sys.stdout.flush()
                try:
                    cold, warm, p95 = _bench_fused_step(
                        mk_model, mk_input, mk_target, bs,
                        autocast=autocast, n_iter=args.iter,
                    )
                    sys.stdout.write(f"warm={warm:.2f}ms cold={cold:.1f}ms\n")
                    note = "ok"
                except Exception as e:  # pragma: no cover
                    cold = warm = p95 = float("nan")
                    note = f"error:{type(e).__name__}:{str(e)[:60]}"
                    sys.stdout.write(f"FAIL: {note}\n")
                sys.stdout.flush()
                rows.append({
                    "workload": name,
                    "precision": prec_label,
                    "batch": bs,
                    "cold_ms": cold,
                    "warm_ms_median": warm,
                    "warm_ms_p95": p95,
                    "note": note,
                })

    # Compute speedup: autocast_f16 vs f32 for the same workload/batch.
    f32_lookup: dict[tuple[str, int], float] = {}
    for r in rows:
        if r["precision"] == "f32" and isinstance(r["warm_ms_median"], float) and r["warm_ms_median"] == r["warm_ms_median"]:
            f32_lookup[(str(r["workload"]), int(r["batch"]))] = float(r["warm_ms_median"])

    for r in rows:
        if r["precision"] == "autocast_f16":
            f32_warm = f32_lookup.get((str(r["workload"]), int(r["batch"])))
            warm = r["warm_ms_median"]
            if (f32_warm is not None and isinstance(warm, float) and warm == warm and warm > 0):
                r["speedup_vs_f32"] = f32_warm / warm
            else:
                r["speedup_vs_f32"] = float("nan")
        else:
            r["speedup_vs_f32"] = 1.0

    # Write CSV.
    fields = ["workload", "precision", "batch", "cold_ms", "warm_ms_median",
              "warm_ms_p95", "speedup_vs_f32", "note"]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            cleaned = {k: ("" if (isinstance(v, float) and v != v) else v)
                       for k, v in r.items() if k in fields}
            for k in fields:
                cleaned.setdefault(k, "")
            writer.writerow(cleaned)
    print(f"\n# CSV written to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
