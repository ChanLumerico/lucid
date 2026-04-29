"""Batch norm and layer norm benchmarks."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from lucid._C import engine as E

from _runner import BenchSpec, lucid_tensor, torch_tensor


def _make_batchnorm(N: int, C: int, H: int, W: int) -> BenchSpec:
    x_np = np.random.randn(N, C, H, W).astype("float32")
    g_np = (np.random.randn(C) * 0.1 + 1.0).astype("float32")
    b_np = np.zeros(C, dtype="float32")
    flops = float(N * C * H * W * 4)

    def factory(backend: str):
        if backend == "lucid-cpu":
            x = lucid_tensor(x_np, E.Device.CPU)
            g = lucid_tensor(g_np, E.Device.CPU)
            bi = lucid_tensor(b_np, E.Device.CPU)
            return lambda: E.nn.batch_norm(x, g, bi, 1e-5)
        if backend == "lucid-gpu":
            x = lucid_tensor(x_np, E.Device.GPU)
            g = lucid_tensor(g_np, E.Device.GPU)
            bi = lucid_tensor(b_np, E.Device.GPU)
            return lambda: E.nn.batch_norm(x, g, bi, 1e-5)
        if backend == "torch-cpu":
            x = torch_tensor(x_np, "cpu")
            g = torch_tensor(g_np, "cpu")
            bi = torch_tensor(b_np, "cpu")
            return lambda: F.batch_norm(x, None, None, g, bi, training=True, eps=1e-5)
        if backend == "torch-mps":
            x = torch_tensor(x_np, "mps")
            g = torch_tensor(g_np, "mps")
            bi = torch_tensor(b_np, "mps")
            return lambda: F.batch_norm(x, None, None, g, bi, training=True, eps=1e-5)
        return None

    return BenchSpec(name=f"batch_norm [{N},{C},{H},{W}]", flops=flops, factory=factory)


def _make_layernorm(N: int, T: int, D: int) -> BenchSpec:
    x_np = np.random.randn(N, T, D).astype("float32")
    g_np = (np.random.randn(D) * 0.1 + 1.0).astype("float32")
    b_np = np.zeros(D, dtype="float32")
    flops = float(N * T * D * 5)

    def factory(backend: str):
        if backend == "lucid-cpu":
            x = lucid_tensor(x_np, E.Device.CPU)
            g = lucid_tensor(g_np, E.Device.CPU)
            bi = lucid_tensor(b_np, E.Device.CPU)
            return lambda: E.nn.layer_norm(x, g, bi, 1e-5)
        if backend == "lucid-gpu":
            x = lucid_tensor(x_np, E.Device.GPU)
            g = lucid_tensor(g_np, E.Device.GPU)
            bi = lucid_tensor(b_np, E.Device.GPU)
            return lambda: E.nn.layer_norm(x, g, bi, 1e-5)
        if backend == "torch-cpu":
            x = torch_tensor(x_np, "cpu")
            g = torch_tensor(g_np, "cpu")
            bi = torch_tensor(b_np, "cpu")
            return lambda: F.layer_norm(x, (D,), g, bi, eps=1e-5)
        if backend == "torch-mps":
            x = torch_tensor(x_np, "mps")
            g = torch_tensor(g_np, "mps")
            bi = torch_tensor(b_np, "mps")
            return lambda: F.layer_norm(x, (D,), g, bi, eps=1e-5)
        return None

    return BenchSpec(name=f"layer_norm [{N},{T},{D}]", flops=flops, factory=factory)


SPECS = [
    _make_batchnorm(4,  64,  56,  56),
    _make_batchnorm(4, 256,  28,  28),
    _make_batchnorm(4, 512,  14,  14),
    _make_layernorm(16,  64,  512),
    _make_layernorm(4,  512, 1024),
]


if __name__ == "__main__":
    import argparse
    from _runner import run_suite

    parser = argparse.ArgumentParser()
    parser.add_argument("--md", action="store_true")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    print("=== Norm benchmarks ===")
    run_suite(SPECS, n_iter=args.n, markdown=args.md)
