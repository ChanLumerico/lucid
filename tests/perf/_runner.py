"""
Lucid performance benchmark runner.

Measures median wall-clock latency (µs) and throughput (GFLOPs) across four
backends:
  lucid-CPU  — engine on Device.CPU  (Apple Accelerate / LAPACK)
  lucid-GPU  — engine on Device.GPU  (MLX / Metal)
  torch-CPU  — PyTorch on cpu
  torch-MPS  — PyTorch on mps        (skip when unavailable)

Usage (from repo root):
  python tests/perf/_runner.py              # default shape set
  python tests/perf/_runner.py --md         # print markdown table
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from lucid._C import engine as E  # noqa: E402

try:
    import mlx.core as _mx
    def _gpu_sync() -> None:
        _mx.synchronize()
except ImportError:
    def _gpu_sync() -> None:
        pass


# --------------------------------------------------------------------------- #
# Timing primitive
# --------------------------------------------------------------------------- #

def _timeit(fn: Callable, n_warmup: int = 8, n_iter: int = 50,
            after: Callable | None = None) -> float:
    """Return median latency in microseconds.

    `after` is called after each fn() call to force GPU synchronisation
    before stopping the clock. Pass `mlx.core.synchronize` or
    `torch.mps.synchronize` for GPU backends.
    """
    for _ in range(n_warmup):
        fn()
        if after:
            after()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        if after:
            after()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e6


def _mps_available() -> bool:
    try:
        return torch.backends.mps.is_available()
    except Exception:
        return False


def _gpu_available() -> bool:
    try:
        t = E.TensorImpl(np.ones((2, 2), dtype="float32"), E.Device.GPU, False)
        _ = t.shape
        return True
    except Exception:
        return False


MPS_OK = _mps_available()
GPU_OK = _gpu_available()


# --------------------------------------------------------------------------- #
# BenchSpec — one benchmark case
# --------------------------------------------------------------------------- #

@dataclass
class TimingResult:
    backend: str
    latency_us: float          # median latency in µs
    flops: float | None        # peak FLOPs (None → N/A)

    @property
    def gflops(self) -> float | None:
        if self.flops is None or self.latency_us <= 0:
            return None
        return self.flops / (self.latency_us * 1e-6) / 1e9


@dataclass
class BenchSpec:
    """One benchmark: an op measured across backends."""
    name: str
    flops: float | None = None   # theoretical FLOPs per call (for throughput)

    # Callable factories: fn(device_str) -> Callable[[], None]
    # device_str is one of: "lucid-cpu", "lucid-gpu", "torch-cpu", "torch-mps"
    # Returns a zero-arg callable that executes one forward pass.
    factory: Callable[[str], Callable[[], None] | None] = field(repr=False, default=None)

    def run(self, n_warmup: int = 8, n_iter: int = 50) -> list[TimingResult]:
        results = []
        for backend in ["lucid-cpu", "lucid-gpu", "torch-cpu", "torch-mps"]:
            if backend == "lucid-gpu" and not GPU_OK:
                continue
            if backend == "torch-mps" and not MPS_OK:
                continue
            fn = self.factory(backend)
            if fn is None:
                continue
            gc.collect()
            if backend == "lucid-gpu":
                after = _gpu_sync
            elif backend == "torch-mps":
                after = torch.mps.synchronize
            else:
                after = None
            try:
                us = _timeit(fn, n_warmup, n_iter, after=after)
            except Exception as exc:
                print(f"  [{self.name} | {backend}] ERROR: {exc}", file=sys.stderr)
                continue
            results.append(TimingResult(backend, us, self.flops))
        return results


# --------------------------------------------------------------------------- #
# Helpers for building inputs on each backend
# --------------------------------------------------------------------------- #

def lucid_tensor(arr: np.ndarray, device: E.Device) -> E.TensorImpl:
    return E.TensorImpl(np.ascontiguousarray(arr), device, False)


def torch_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    t = torch.from_numpy(arr.copy())
    if device != "cpu":
        t = t.to(device)
    return t


# --------------------------------------------------------------------------- #
# Pretty-print
# --------------------------------------------------------------------------- #

_COL_W = 14

def _fmt_us(v: float) -> str:
    if v < 1000:
        return f"{v:.1f} µs"
    return f"{v/1000:.2f} ms"


def _fmt_gflops(r: TimingResult) -> str:
    g = r.gflops
    if g is None:
        return "—"
    return f"{g:.1f}"


def _pct_of_torch(r: TimingResult | None, ref: TimingResult | None) -> str:
    """Format latency as % of torch-cpu (100% = same speed, <100% = faster)."""
    if r is None or ref is None or ref.latency_us <= 0:
        return "—"
    pct = r.latency_us / ref.latency_us * 100
    return f"{pct:.0f}%"


def print_table(specs_results: list[tuple[BenchSpec, list[TimingResult]]],
                markdown: bool = False) -> None:
    backends = ["lucid-cpu", "lucid-gpu", "torch-cpu", "torch-mps"]
    active = [b for b in backends
              if (b != "lucid-gpu" or GPU_OK)
              and (b != "torch-mps" or MPS_OK)]

    # Columns: latency per backend + GFLOPs(lucid-cpu) + %(lucid-cpu vs torch-cpu)
    lat_cols = active
    extra_cols = ["GFLOPs", "% of torch"]
    all_cols = lat_cols + extra_cols

    if markdown:
        hdr = ["Op"] + all_cols
        print("| " + " | ".join(f"{h}" for h in hdr) + " |")
        print("| " + " | ".join(["---"] * len(hdr)) + " |")
    else:
        header = f"{'Op':<32}" + "".join(f"{b:>{_COL_W}}" for b in all_cols)
        print(header)
        print("-" * len(header))

    for spec, results in specs_results:
        by_backend = {r.backend: r for r in results}
        ref = by_backend.get("torch-cpu")
        lcpu = by_backend.get("lucid-cpu")

        row_us = [_fmt_us(by_backend[b].latency_us) if b in by_backend else "—"
                  for b in active]
        gflops = _fmt_gflops(lcpu) if lcpu else "—"
        pct = _pct_of_torch(lcpu, ref)
        all_vals = row_us + [gflops, pct]

        if markdown:
            cells = [spec.name] + all_vals
            print("| " + " | ".join(cells) + " |")
        else:
            print(f"{spec.name:<32}" + "".join(f"{v:>{_COL_W}}" for v in all_vals))


# --------------------------------------------------------------------------- #
# CLI entry-point (also imported by individual bench_*.py files)
# --------------------------------------------------------------------------- #

def run_suite(specs: list[BenchSpec], n_warmup: int = 8, n_iter: int = 50,
              markdown: bool = False) -> None:
    all_results = []
    for spec in specs:
        print(f"  {spec.name}...", end=" ", flush=True)
        results = spec.run(n_warmup, n_iter)
        all_results.append((spec, results))
        # Quick inline summary
        for r in results:
            print(f"[{r.backend}: {_fmt_us(r.latency_us)}]", end=" ", flush=True)
        print()

    print()
    print_table(all_results, markdown=markdown)
