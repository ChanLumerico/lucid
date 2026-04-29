"""Conv2d benchmarks: typical ResNet / ViT shapes."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from lucid._C import engine as E

from _runner import BenchSpec, lucid_tensor, torch_tensor


def _make_conv2d(N: int, C_in: int, C_out: int, H: int, W: int,
                 kH: int = 3, kW: int = 3,
                 stride: int = 1, pad: int = 1) -> BenchSpec:
    x_np = np.random.randn(N, C_in, H, W).astype("float32") * 0.1
    w_np = np.random.randn(C_out, C_in, kH, kW).astype("float32") * 0.01
    b_np = np.zeros(C_out, dtype="float32")
    # FLOPs = 2 * N * C_out * H_out * W_out * C_in * kH * kW
    H_out = (H + 2 * pad - kH) // stride + 1
    W_out = (W + 2 * pad - kW) // stride + 1
    flops = 2.0 * N * C_out * H_out * W_out * C_in * kH * kW

    def factory(backend: str):
        if backend == "lucid-cpu":
            x = lucid_tensor(x_np, E.Device.CPU)
            w = lucid_tensor(w_np, E.Device.CPU)
            bi = lucid_tensor(b_np, E.Device.CPU)
            return lambda: E.nn.conv2d(x, w, bi, stride, stride, pad, pad, 1, 1, 1)
        if backend == "lucid-gpu":
            x = lucid_tensor(x_np, E.Device.GPU)
            w = lucid_tensor(w_np, E.Device.GPU)
            bi = lucid_tensor(b_np, E.Device.GPU)
            return lambda: E.nn.conv2d(x, w, bi, stride, stride, pad, pad, 1, 1, 1)
        if backend == "torch-cpu":
            x = torch_tensor(x_np, "cpu")
            w = torch_tensor(w_np, "cpu")
            bi = torch_tensor(b_np, "cpu")
            return lambda: F.conv2d(x, w, bi, stride=stride, padding=pad)
        if backend == "torch-mps":
            x = torch_tensor(x_np, "mps")
            w = torch_tensor(w_np, "mps")
            bi = torch_tensor(b_np, "mps")
            return lambda: F.conv2d(x, w, bi, stride=stride, padding=pad)
        return None

    label = f"conv2d [{N},{C_in},{H},{W}] → {C_out}×{kH}×{kW} s{stride}p{pad}"
    return BenchSpec(name=label, flops=flops, factory=factory)


SPECS = [
    _make_conv2d(1,   3,   64,  224, 224, 7, 7, stride=2, pad=3),  # ResNet stem
    _make_conv2d(4,   64,  64,   56,  56),                           # ResNet layer1
    _make_conv2d(4,  256, 512,   28,  28),                           # ResNet layer3
    _make_conv2d(16,   3,  64,   32,  32),                           # small batch
]


if __name__ == "__main__":
    import argparse
    from _runner import run_suite

    parser = argparse.ArgumentParser()
    parser.add_argument("--md", action="store_true")
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    print("=== Conv2d benchmarks ===")
    run_suite(SPECS, n_iter=args.n, markdown=args.md)
