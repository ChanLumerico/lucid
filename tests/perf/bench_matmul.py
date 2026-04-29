"""Matmul benchmarks: lucid-CPU / lucid-GPU / torch-CPU / torch-MPS."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from lucid._C import engine as E

from _runner import BenchSpec, lucid_tensor, torch_tensor


def _make_matmul(M: int, K: int, N: int) -> BenchSpec:
    A_np = np.random.randn(M, K).astype("float32")
    B_np = np.random.randn(K, N).astype("float32")
    flops = 2.0 * M * K * N

    def factory(backend: str):
        if backend == "lucid-cpu":
            a = lucid_tensor(A_np, E.Device.CPU)
            b = lucid_tensor(B_np, E.Device.CPU)
            return lambda: E.matmul(a, b)
        if backend == "lucid-gpu":
            a = lucid_tensor(A_np, E.Device.GPU)
            b = lucid_tensor(B_np, E.Device.GPU)
            return lambda: E.matmul(a, b)
        if backend == "torch-cpu":
            a = torch_tensor(A_np, "cpu")
            b = torch_tensor(B_np, "cpu")
            return lambda: torch.matmul(a, b)
        if backend == "torch-mps":
            a = torch_tensor(A_np, "mps")
            b = torch_tensor(B_np, "mps")
            return lambda: torch.matmul(a, b)
        return None

    return BenchSpec(name=f"matmul [{M},{K}]x[{K},{N}]", flops=flops, factory=factory)


SPECS = [
    _make_matmul(128,  128,  128),
    _make_matmul(512,  512,  512),
    _make_matmul(1024, 1024, 1024),
    _make_matmul(2048, 2048, 2048),
    _make_matmul(64,   4096, 4096),   # wide K — bandwidth-bound
]


if __name__ == "__main__":
    import argparse, sys
    from _runner import run_suite

    parser = argparse.ArgumentParser()
    parser.add_argument("--md", action="store_true")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    print("=== Matmul benchmarks ===")
    run_suite(SPECS, n_iter=args.n, markdown=args.md)
