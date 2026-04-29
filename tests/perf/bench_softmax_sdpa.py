"""Softmax and Scaled Dot-Product Attention benchmarks."""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from lucid._C import engine as E

from _runner import BenchSpec, lucid_tensor, torch_tensor


def _make_softmax(B: int, T: int, D: int) -> BenchSpec:
    x_np = np.random.randn(B, T, D).astype("float32")
    flops = float(B * T * D * 5)  # exp + sum + div ≈ 5 ops/element

    def factory(backend: str):
        if backend == "lucid-cpu":
            x = lucid_tensor(x_np, E.Device.CPU)
            return lambda: E.softmax(x, -1)
        if backend == "lucid-gpu":
            x = lucid_tensor(x_np, E.Device.GPU)
            return lambda: E.softmax(x, -1)
        if backend == "torch-cpu":
            x = torch_tensor(x_np, "cpu")
            return lambda: F.softmax(x, dim=-1)
        if backend == "torch-mps":
            x = torch_tensor(x_np, "mps")
            return lambda: F.softmax(x, dim=-1)
        return None

    return BenchSpec(name=f"softmax [{B},{T},{D}]", flops=flops, factory=factory)


def _make_sdpa(B: int, H: int, T: int, D: int) -> BenchSpec:
    scale = 1.0 / math.sqrt(D)
    q_np = np.random.randn(B, H, T, D).astype("float32") * scale
    k_np = np.random.randn(B, H, T, D).astype("float32") * scale
    v_np = np.random.randn(B, H, T, D).astype("float32")
    # FLOPs: 2 QK^T matmuls per head
    flops = 2.0 * 2.0 * B * H * T * T * D  # QK^T + AV

    def factory(backend: str):
        if backend == "lucid-cpu":
            q = lucid_tensor(q_np, E.Device.CPU)
            k = lucid_tensor(k_np, E.Device.CPU)
            v = lucid_tensor(v_np, E.Device.CPU)
            return lambda: E.nn.scaled_dot_product_attention(q, k, v, None, scale, False)
        if backend == "lucid-gpu":
            q = lucid_tensor(q_np, E.Device.GPU)
            k = lucid_tensor(k_np, E.Device.GPU)
            v = lucid_tensor(v_np, E.Device.GPU)
            return lambda: E.nn.scaled_dot_product_attention(q, k, v, None, scale, False)
        if backend == "torch-cpu":
            q = torch_tensor(q_np, "cpu")
            k = torch_tensor(k_np, "cpu")
            v = torch_tensor(v_np, "cpu")
            return lambda: F.scaled_dot_product_attention(q, k, v, scale=scale)
        if backend == "torch-mps":
            q = torch_tensor(q_np, "mps")
            k = torch_tensor(k_np, "mps")
            v = torch_tensor(v_np, "mps")
            return lambda: F.scaled_dot_product_attention(q, k, v, scale=scale)
        return None

    return BenchSpec(name=f"sdpa [{B},{H},{T},{D}]", flops=flops, factory=factory)


SPECS = [
    _make_softmax(4,  512,  512),
    _make_softmax(32, 128, 1024),
    _make_sdpa(2, 8,  64, 64),
    _make_sdpa(2, 8, 128, 64),
    _make_sdpa(2, 16, 256, 64),
]


if __name__ == "__main__":
    import argparse
    from _runner import run_suite

    parser = argparse.ArgumentParser()
    parser.add_argument("--md", action="store_true")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    print("=== Softmax + SDPA benchmarks ===")
    run_suite(SPECS, n_iter=args.n, markdown=args.md)
