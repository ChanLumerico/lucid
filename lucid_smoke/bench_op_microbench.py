#!/usr/bin/env python3
"""bench_op_microbench.py — Phase 0 baseline for Lucid 3.4 MPSGraph dispatch.

Measures Lucid (MLX backend) vs PyTorch (MPS backend) per-op latency on the
candidate set for MPSGraph promotion.  The output JSON drives the Phase 0.4
shortlist decision: any op where ``lucid / torch >= 1.3`` graduates.

Designed to run on the canonical M4 Max Mac Studio host
(`/Users/chanlee/lucid_smoke/` after macstudio-bench deploy).

Examples
--------
    python bench_op_microbench.py --out /tmp/baseline.json
    python bench_op_microbench.py --backend lucid --op conv2d --dtype f32
    python bench_op_microbench.py --filter rn18 --measure 80
"""

import argparse
import gc
import json
import os
import platform
import socket
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Backend abstraction — lazy imports so we don't pull both stacks needlessly.
# ---------------------------------------------------------------------------


class LucidBackend:
    name = "lucid"

    def __init__(self) -> None:
        import lucid
        import lucid.nn.functional as F
        from lucid._C import engine as _C_engine

        self.lucid = lucid
        self.F = F
        self._C_engine = _C_engine
        self.version = getattr(lucid, "__version__", "unknown")
        self.device = "metal"
        lucid.manual_seed(0)

    def dtype(self, name: str):
        return self.lucid.float32 if name == "f32" else self.lucid.float16

    def randn(self, shape, dtype, *, requires_grad: bool = False):
        # lucid.randn only supports F32/F64 — for F16 we generate F32 then cast.
        target = self.dtype(dtype)
        if dtype == "f32":
            return self.lucid.randn(
                *shape, dtype=target, device=self.device, requires_grad=requires_grad
            )
        t = self.lucid.randn(*shape, dtype=self.lucid.float32, device=self.device).to(dtype=target)
        if requires_grad:
            t.requires_grad = True
        return t

    def randint_idx(self, low: int, high: int, shape):
        return self.lucid.randint(low, high, shape, device=self.device)

    def ones_1d(self, n: int, dtype, *, requires_grad: bool = False):
        target = self.dtype(dtype)
        if dtype == "f32":
            return self.lucid.ones(
                n, dtype=target, device=self.device, requires_grad=requires_grad
            )
        t = self.lucid.ones(n, dtype=self.lucid.float32, device=self.device).to(dtype=target)
        if requires_grad:
            t.requires_grad = True
        return t

    def zeros_1d(self, n: int, dtype, *, requires_grad: bool = False):
        target = self.dtype(dtype)
        if dtype == "f32":
            return self.lucid.zeros(
                n, dtype=target, device=self.device, requires_grad=requires_grad
            )
        t = self.lucid.zeros(n, dtype=self.lucid.float32, device=self.device).to(dtype=target)
        if requires_grad:
            t.requires_grad = True
        return t

    def sync(self, t) -> None:
        if t is None:
            return
        impl = getattr(t, "impl", None)
        if impl is None:
            return
        self._C_engine.eval_gpu(impl)

    def sync_many(self, tensors) -> None:
        impls = []
        for t in tensors:
            if t is None:
                continue
            impl = getattr(t, "impl", None)
            if impl is not None:
                impls.append(impl)
        if impls:
            self._C_engine.eval_tensors(impls)

    def zero_grad(self, t) -> None:
        if t is not None:
            t.grad = None


class TorchBackend:
    name = "torch"

    def __init__(self) -> None:
        import torch
        import torch.nn.functional as F

        if not torch.backends.mps.is_available():
            raise RuntimeError("torch.backends.mps is not available on this host")
        self.torch = torch
        self.F = F
        self.version = torch.__version__
        self.device = "mps"
        torch.manual_seed(0)

    def dtype(self, name: str):
        return self.torch.float32 if name == "f32" else self.torch.float16

    def randn(self, shape, dtype, *, requires_grad: bool = False):
        return self.torch.randn(
            *shape,
            dtype=self.dtype(dtype),
            device=self.device,
            requires_grad=requires_grad,
        )

    def randint_idx(self, low: int, high: int, shape):
        return self.torch.randint(
            low, high, tuple(shape), device=self.device, dtype=self.torch.int64
        )

    def ones_1d(self, n: int, dtype, *, requires_grad: bool = False):
        t = self.torch.ones(n, dtype=self.dtype(dtype), device=self.device)
        if requires_grad:
            t.requires_grad_(True)
        return t

    def zeros_1d(self, n: int, dtype, *, requires_grad: bool = False):
        t = self.torch.zeros(n, dtype=self.dtype(dtype), device=self.device)
        if requires_grad:
            t.requires_grad_(True)
        return t

    def sync(self, t) -> None:  # torch MPS sync is global
        self.torch.mps.synchronize()

    def sync_many(self, tensors) -> None:  # one global sync covers all
        self.torch.mps.synchronize()

    def zero_grad(self, t) -> None:
        if t is not None and t.grad is not None:
            t.grad = None


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------


def _perf_ns() -> int:
    return time.perf_counter_ns()


def _stats_ms(samples_ns):
    if not samples_ns:
        return {"median": float("nan"), "p95": float("nan"), "min": float("nan"), "iters": 0}
    samples_ms = sorted(s / 1e6 for s in samples_ns)
    n = len(samples_ms)
    p95_idx = max(0, min(n - 1, int(round(0.95 * (n - 1)))))
    return {
        "median": statistics.median(samples_ms),
        "p95": samples_ms[p95_idx],
        "min": samples_ms[0],
        "iters": n,
    }


def time_forward(backend, build_fwd, warmup: int, measure: int):
    for _ in range(warmup):
        out = build_fwd()
        backend.sync(out)
    samples = []
    for _ in range(measure):
        t0 = _perf_ns()
        out = build_fwd()
        backend.sync(out)
        t1 = _perf_ns()
        samples.append(t1 - t0)
    return _stats_ms(samples)


def time_backward(backend, build_loss, learnables, warmup: int, measure: int):
    # build_loss is a 0-arg callable returning a scalar loss; learnables is the
    # list of requires_grad inputs whose .grad we sync + reset each iter.
    grads = lambda: [getattr(p, "grad", None) for p in learnables]
    for _ in range(warmup):
        loss = build_loss()
        backend.sync(loss)
        loss.backward()
        backend.sync_many(grads())
        for p in learnables:
            backend.zero_grad(p)
    samples = []
    for _ in range(measure):
        loss = build_loss()
        backend.sync(loss)
        t0 = _perf_ns()
        loss.backward()
        backend.sync_many(grads())
        t1 = _perf_ns()
        for p in learnables:
            backend.zero_grad(p)
        samples.append(t1 - t0)
    return _stats_ms(samples)


# ---------------------------------------------------------------------------
# Per-op runners.  Each takes (backend, args_dict, dtype, warmup, measure) and
# returns {"fwd_ms": {...}, "bwd_ms": {...}} or {"fwd_ms": {...}} for fwd-only.
# ---------------------------------------------------------------------------


def _run_conv2d(backend, args, dtype, warmup, measure):
    N, Cin, H, W = args["N"], args["Cin"], args["H"], args["W"]
    Cout, K, S, P, groups = args["Cout"], args["K"], args["S"], args["P"], args["groups"]
    x_fwd = backend.randn((N, Cin, H, W), dtype)
    W_fwd = backend.randn((Cout, Cin // groups, K, K), dtype)
    b_fwd = backend.randn((Cout,), dtype)
    fwd_stats = time_forward(
        backend,
        lambda: backend.F.conv2d(x_fwd, W_fwd, b_fwd, stride=S, padding=P, groups=groups),
        warmup,
        measure,
    )
    x_b = backend.randn((N, Cin, H, W), dtype, requires_grad=True)
    W_b = backend.randn((Cout, Cin // groups, K, K), dtype, requires_grad=True)
    b_b = backend.randn((Cout,), dtype, requires_grad=True)

    def loss_fn():
        y = backend.F.conv2d(x_b, W_b, b_b, stride=S, padding=P, groups=groups)
        return y.sum()

    bwd_stats = time_backward(backend, loss_fn, [x_b, W_b, b_b], warmup, measure)
    return {"fwd_ms": fwd_stats, "bwd_ms": bwd_stats}


def _bn_buffers(backend, C: int, dtype):
    # Running stats must be valid floats (no NaN) but require no grad.
    rm = backend.zeros_1d(C, dtype)
    rv = backend.ones_1d(C, dtype)
    return rm, rv


def _run_batch_norm_train(backend, args, dtype, warmup, measure):
    N, C, H, W = args["N"], args["C"], args["H"], args["W"]
    x_fwd = backend.randn((N, C, H, W), dtype)
    rm_f, rv_f = _bn_buffers(backend, C, dtype)
    g_f = backend.ones_1d(C, dtype)
    b_f = backend.zeros_1d(C, dtype)
    fwd_stats = time_forward(
        backend,
        lambda: backend.F.batch_norm(x_fwd, rm_f, rv_f, g_f, b_f, training=True),
        warmup,
        measure,
    )
    x_b = backend.randn((N, C, H, W), dtype, requires_grad=True)
    g_b = backend.ones_1d(C, dtype, requires_grad=True)
    b_b = backend.zeros_1d(C, dtype, requires_grad=True)
    rm_b, rv_b = _bn_buffers(backend, C, dtype)

    def loss_fn():
        y = backend.F.batch_norm(x_b, rm_b, rv_b, g_b, b_b, training=True)
        return y.sum()

    bwd_stats = time_backward(backend, loss_fn, [x_b, g_b, b_b], warmup, measure)
    return {"fwd_ms": fwd_stats, "bwd_ms": bwd_stats}


def _run_batch_norm_eval(backend, args, dtype, warmup, measure):
    N, C, H, W = args["N"], args["C"], args["H"], args["W"]
    x_fwd = backend.randn((N, C, H, W), dtype)
    rm, rv = _bn_buffers(backend, C, dtype)
    g = backend.ones_1d(C, dtype)
    b = backend.zeros_1d(C, dtype)
    fwd_stats = time_forward(
        backend,
        lambda: backend.F.batch_norm(x_fwd, rm, rv, g, b, training=False),
        warmup,
        measure,
    )
    # Eval mode rarely needs grad — measure forward only.
    return {"fwd_ms": fwd_stats}


def _run_max_pool2d(backend, args, dtype, warmup, measure):
    N, C, H, W, K, S, P = args["N"], args["C"], args["H"], args["W"], args["K"], args["S"], args["P"]
    x_fwd = backend.randn((N, C, H, W), dtype)
    fwd_stats = time_forward(
        backend,
        lambda: backend.F.max_pool2d(x_fwd, kernel_size=K, stride=S, padding=P),
        warmup,
        measure,
    )
    x_b = backend.randn((N, C, H, W), dtype, requires_grad=True)

    def loss_fn():
        y = backend.F.max_pool2d(x_b, kernel_size=K, stride=S, padding=P)
        return y.sum()

    bwd_stats = time_backward(backend, loss_fn, [x_b], warmup, measure)
    return {"fwd_ms": fwd_stats, "bwd_ms": bwd_stats}


def _run_avg_pool2d(backend, args, dtype, warmup, measure):
    N, C, H, W, K, S, P = args["N"], args["C"], args["H"], args["W"], args["K"], args["S"], args["P"]
    x_fwd = backend.randn((N, C, H, W), dtype)
    fwd_stats = time_forward(
        backend,
        lambda: backend.F.avg_pool2d(x_fwd, kernel_size=K, stride=S, padding=P),
        warmup,
        measure,
    )
    x_b = backend.randn((N, C, H, W), dtype, requires_grad=True)

    def loss_fn():
        y = backend.F.avg_pool2d(x_b, kernel_size=K, stride=S, padding=P)
        return y.sum()

    bwd_stats = time_backward(backend, loss_fn, [x_b], warmup, measure)
    return {"fwd_ms": fwd_stats, "bwd_ms": bwd_stats}


def _run_adaptive_avg_pool2d(backend, args, dtype, warmup, measure):
    N, C, H, W = args["N"], args["C"], args["H"], args["W"]
    target = args["target"]
    x_fwd = backend.randn((N, C, H, W), dtype)
    fwd_stats = time_forward(
        backend,
        lambda: backend.F.adaptive_avg_pool2d(x_fwd, output_size=target),
        warmup,
        measure,
    )
    x_b = backend.randn((N, C, H, W), dtype, requires_grad=True)

    def loss_fn():
        y = backend.F.adaptive_avg_pool2d(x_b, output_size=target)
        return y.sum()

    bwd_stats = time_backward(backend, loss_fn, [x_b], warmup, measure)
    return {"fwd_ms": fwd_stats, "bwd_ms": bwd_stats}


def _run_linear(backend, args, dtype, warmup, measure):
    N, IN, OUT = args["N"], args["in"], args["out"]
    x_fwd = backend.randn((N, IN), dtype)
    W_fwd = backend.randn((OUT, IN), dtype)
    b_fwd = backend.randn((OUT,), dtype)
    fwd_stats = time_forward(
        backend, lambda: backend.F.linear(x_fwd, W_fwd, b_fwd), warmup, measure
    )
    x_b = backend.randn((N, IN), dtype, requires_grad=True)
    W_b = backend.randn((OUT, IN), dtype, requires_grad=True)
    b_b = backend.randn((OUT,), dtype, requires_grad=True)

    def loss_fn():
        y = backend.F.linear(x_b, W_b, b_b)
        return y.sum()

    bwd_stats = time_backward(backend, loss_fn, [x_b, W_b, b_b], warmup, measure)
    return {"fwd_ms": fwd_stats, "bwd_ms": bwd_stats}


def _run_matmul(backend, args, dtype, warmup, measure):
    # Generic A @ B with shapes args["a"], args["b"].
    A_shape = tuple(args["a"])
    B_shape = tuple(args["b"])
    a_fwd = backend.randn(A_shape, dtype)
    b_fwd = backend.randn(B_shape, dtype)
    fwd_stats = time_forward(backend, lambda: a_fwd @ b_fwd, warmup, measure)
    a_b = backend.randn(A_shape, dtype, requires_grad=True)
    b_b = backend.randn(B_shape, dtype, requires_grad=True)

    def loss_fn():
        return (a_b @ b_b).sum()

    bwd_stats = time_backward(backend, loss_fn, [a_b, b_b], warmup, measure)
    return {"fwd_ms": fwd_stats, "bwd_ms": bwd_stats}


def _run_activation(act_name: str):
    def runner(backend, args, dtype, warmup, measure):
        shape = tuple(args["shape"])
        x_fwd = backend.randn(shape, dtype)
        fn = getattr(backend.F, act_name)
        fwd_stats = time_forward(backend, lambda: fn(x_fwd), warmup, measure)
        x_b = backend.randn(shape, dtype, requires_grad=True)

        def loss_fn():
            return fn(x_b).sum()

        bwd_stats = time_backward(backend, loss_fn, [x_b], warmup, measure)
        return {"fwd_ms": fwd_stats, "bwd_ms": bwd_stats}

    return runner


def _run_softmax(backend, args, dtype, warmup, measure):
    shape = tuple(args["shape"])
    dim = args["dim"]
    x_fwd = backend.randn(shape, dtype)
    fwd_stats = time_forward(
        backend, lambda: backend.F.softmax(x_fwd, dim=dim), warmup, measure
    )
    x_b = backend.randn(shape, dtype, requires_grad=True)

    def loss_fn():
        return backend.F.softmax(x_b, dim=dim).sum()

    bwd_stats = time_backward(backend, loss_fn, [x_b], warmup, measure)
    return {"fwd_ms": fwd_stats, "bwd_ms": bwd_stats}


def _run_embedding(backend, args, dtype, warmup, measure):
    B, L, V, E = args["B"], args["L"], args["V"], args["E"]
    idx = backend.randint_idx(0, V, (B, L))
    W_fwd = backend.randn((V, E), dtype)
    fwd_stats = time_forward(
        backend, lambda: backend.F.embedding(idx, W_fwd), warmup, measure
    )
    # Backward — scatter_add into weight.grad is the hot path.
    W_b = backend.randn((V, E), dtype, requires_grad=True)

    def loss_fn():
        return backend.F.embedding(idx, W_b).sum()

    bwd_stats = time_backward(backend, loss_fn, [W_b], warmup, measure)
    return {"fwd_ms": fwd_stats, "bwd_ms": bwd_stats}


def _run_layer_norm(backend, args, dtype, warmup, measure):
    shape = tuple(args["shape"])
    normalized = tuple(args["normalized_shape"])
    x_fwd = backend.randn(shape, dtype)
    g = backend.ones_1d(normalized[-1], dtype)
    b = backend.zeros_1d(normalized[-1], dtype)
    fwd_stats = time_forward(
        backend,
        lambda: backend.F.layer_norm(x_fwd, normalized, g, b),
        warmup,
        measure,
    )
    x_b = backend.randn(shape, dtype, requires_grad=True)
    g_b = backend.ones_1d(normalized[-1], dtype, requires_grad=True)
    b_b = backend.zeros_1d(normalized[-1], dtype, requires_grad=True)

    def loss_fn():
        return backend.F.layer_norm(x_b, normalized, g_b, b_b).sum()

    bwd_stats = time_backward(backend, loss_fn, [x_b, g_b, b_b], warmup, measure)
    return {"fwd_ms": fwd_stats, "bwd_ms": bwd_stats}


def _run_group_norm(backend, args, dtype, warmup, measure):
    N, C, H, W = args["N"], args["C"], args["H"], args["W"]
    G = args["groups"]
    x_fwd = backend.randn((N, C, H, W), dtype)
    g = backend.ones_1d(C, dtype)
    b = backend.zeros_1d(C, dtype)
    fwd_stats = time_forward(
        backend,
        lambda: backend.F.group_norm(x_fwd, G, g, b),
        warmup,
        measure,
    )
    x_b = backend.randn((N, C, H, W), dtype, requires_grad=True)
    g_b = backend.ones_1d(C, dtype, requires_grad=True)
    b_b = backend.zeros_1d(C, dtype, requires_grad=True)

    def loss_fn():
        return backend.F.group_norm(x_b, G, g_b, b_b).sum()

    bwd_stats = time_backward(backend, loss_fn, [x_b, g_b, b_b], warmup, measure)
    return {"fwd_ms": fwd_stats, "bwd_ms": bwd_stats}


def _run_rms_norm(backend, args, dtype, warmup, measure):
    shape = tuple(args["shape"])
    normalized = tuple(args["normalized_shape"])
    x_fwd = backend.randn(shape, dtype)
    g = backend.ones_1d(normalized[-1], dtype)
    fwd_stats = time_forward(
        backend,
        lambda: backend.F.rms_norm(x_fwd, normalized, g),
        warmup,
        measure,
    )
    x_b = backend.randn(shape, dtype, requires_grad=True)
    g_b = backend.ones_1d(normalized[-1], dtype, requires_grad=True)

    def loss_fn():
        return backend.F.rms_norm(x_b, normalized, g_b).sum()

    bwd_stats = time_backward(backend, loss_fn, [x_b, g_b], warmup, measure)
    return {"fwd_ms": fwd_stats, "bwd_ms": bwd_stats}


# ---------------------------------------------------------------------------
# Shape catalogue.  Labels keep the JSON readable and let `--filter` target
# subsets like "rn18" or "transformer".
# ---------------------------------------------------------------------------


CONV2D_SHAPES = [
    {"label": "rn18_first_3x3",     "N": 32, "Cin": 3,   "H": 32,  "W": 32,  "Cout": 64,  "K": 3, "S": 1, "P": 1, "groups": 1},
    {"label": "rn18_layer1_3x3",    "N": 32, "Cin": 64,  "H": 32,  "W": 32,  "Cout": 64,  "K": 3, "S": 1, "P": 1, "groups": 1},
    {"label": "rn18_layer2_ds_3x3", "N": 32, "Cin": 64,  "H": 32,  "W": 32,  "Cout": 128, "K": 3, "S": 2, "P": 1, "groups": 1},
    {"label": "rn18_layer2_3x3",    "N": 32, "Cin": 128, "H": 16,  "W": 16,  "Cout": 128, "K": 3, "S": 1, "P": 1, "groups": 1},
    {"label": "rn18_layer3_ds_3x3", "N": 32, "Cin": 128, "H": 16,  "W": 16,  "Cout": 256, "K": 3, "S": 2, "P": 1, "groups": 1},
    {"label": "rn18_layer3_3x3",    "N": 32, "Cin": 256, "H": 8,   "W": 8,   "Cout": 256, "K": 3, "S": 1, "P": 1, "groups": 1},
    {"label": "rn18_layer4_ds_3x3", "N": 32, "Cin": 256, "H": 8,   "W": 8,   "Cout": 512, "K": 3, "S": 2, "P": 1, "groups": 1},
    {"label": "rn18_layer4_3x3",    "N": 32, "Cin": 512, "H": 4,   "W": 4,   "Cout": 512, "K": 3, "S": 1, "P": 1, "groups": 1},
    {"label": "imagenet_stem_7x7",  "N": 16, "Cin": 3,   "H": 224, "W": 224, "Cout": 64,  "K": 7, "S": 2, "P": 3, "groups": 1},
    {"label": "ds_1x1",             "N": 32, "Cin": 128, "H": 16,  "W": 16,  "Cout": 128, "K": 1, "S": 1, "P": 0, "groups": 1},
    {"label": "depthwise_3x3",      "N": 32, "Cin": 128, "H": 16,  "W": 16,  "Cout": 128, "K": 3, "S": 1, "P": 1, "groups": 128},
]


BN_SHAPES = [
    {"label": "rn18_first",  "N": 32, "C": 64,  "H": 32, "W": 32},
    {"label": "rn18_l2",     "N": 32, "C": 128, "H": 16, "W": 16},
    {"label": "rn18_l3",     "N": 32, "C": 256, "H": 8,  "W": 8},
    {"label": "rn18_l4",     "N": 32, "C": 512, "H": 4,  "W": 4},
    {"label": "large_acts",  "N": 32, "C": 64,  "H": 112,"W": 112},
]


MAXPOOL_SHAPES = [
    {"label": "imagenet_stem",  "N": 32, "C": 64,  "H": 112,"W": 112, "K": 3, "S": 2, "P": 1},
    {"label": "vgg_block_a",    "N": 32, "C": 128, "H": 56, "W": 56,  "K": 2, "S": 2, "P": 0},
]


AVGPOOL_SHAPES = [
    {"label": "rn18_global",    "N": 32, "C": 512, "H": 4,  "W": 4,  "K": 4, "S": 4, "P": 0},
    {"label": "imagenet_global","N": 16, "C": 2048,"H": 7,  "W": 7,  "K": 7, "S": 7, "P": 0},
]


ADAPTIVE_AVGPOOL_SHAPES = [
    {"label": "rn18_to_1x1",    "N": 32, "C": 512, "H": 4,  "W": 4,  "target": (1, 1)},
    {"label": "imagenet_to_1x1","N": 16, "C": 2048,"H": 7,  "W": 7,  "target": (1, 1)},
]


LINEAR_SHAPES = [
    {"label": "rn18_head",    "N": 32, "in": 512,   "out": 10},
    {"label": "imagenet_head","N": 16, "in": 2048,  "out": 1000},
    {"label": "gpt_qkv",      "N": 32 * 128, "in": 768,  "out": 2304},  # B*L flattened
    {"label": "gpt_ffn_up",   "N": 32 * 128, "in": 768,  "out": 3072},
    {"label": "gpt_ffn_down", "N": 32 * 128, "in": 3072, "out": 768},
    {"label": "llama_ffn_up", "N": 16 * 256, "in": 1024, "out": 4096},
]


MATMUL_SHAPES = [
    {"label": "attn_qk_b32_h12_l64_d64", "a": [32, 12, 64, 64], "b": [32, 12, 64, 64]},
    {"label": "attn_qk_b16_h8_l256_d64", "a": [16, 8, 256, 64], "b": [16, 8, 64, 256]},
    {"label": "bgemm_b16_M256_K64_N256", "a": [16, 256, 64],     "b": [16, 64, 256]},
    {"label": "gemm_M4096_K1024_N1024",  "a": [4096, 1024],       "b": [1024, 1024]},
]


ACTIVATION_SHAPES = [
    {"label": "rn18_l1_acts",     "shape": [32, 64,  32, 32]},
    {"label": "rn18_l2_acts",     "shape": [32, 128, 16, 16]},
    {"label": "transformer_acts", "shape": [32, 128, 768]},
    {"label": "ffn_acts_big",     "shape": [32, 128, 3072]},
]


SOFTMAX_SHAPES = [
    {"label": "ce_imagenet",     "shape": [32, 1000],          "dim": -1},
    {"label": "ce_gpt2_logits",  "shape": [32 * 128, 50257],   "dim": -1},
    {"label": "attn_softmax",    "shape": [32, 12, 64, 64],    "dim": -1},
]


EMBEDDING_SHAPES = [
    {"label": "gpt2_input",      "B": 32, "L": 128, "V": 50257, "E": 768},
    {"label": "bert_input",      "B": 16, "L": 256, "V": 30522, "E": 768},
    {"label": "small_emb",       "B": 64, "L": 64,  "V": 10000, "E": 256},
]


LAYERNORM_SHAPES = [
    {"label": "gpt2_layer",      "shape": [32, 128, 768],  "normalized_shape": [768]},
    {"label": "vit_large_layer", "shape": [16, 256, 1024], "normalized_shape": [1024]},
    {"label": "llama_layer",     "shape": [16, 256, 4096], "normalized_shape": [4096]},
]


GROUPNORM_SHAPES = [
    {"label": "convnext_block", "N": 32, "C": 256, "H": 8,  "W": 8,  "groups": 32},
    {"label": "midnet_block",   "N": 16, "C": 512, "H": 14, "W": 14, "groups": 32},
]


RMSNORM_SHAPES = [
    {"label": "gpt2_layer",     "shape": [32, 128, 768],  "normalized_shape": [768]},
    {"label": "llama_layer",    "shape": [16, 256, 4096], "normalized_shape": [4096]},
]


OPS = {
    "conv2d":              {"shapes": CONV2D_SHAPES,           "run": _run_conv2d},
    "batch_norm_train":    {"shapes": BN_SHAPES,               "run": _run_batch_norm_train},
    "batch_norm_eval":     {"shapes": BN_SHAPES,               "run": _run_batch_norm_eval},
    "max_pool2d":          {"shapes": MAXPOOL_SHAPES,          "run": _run_max_pool2d},
    "avg_pool2d":          {"shapes": AVGPOOL_SHAPES,          "run": _run_avg_pool2d},
    "adaptive_avg_pool2d": {"shapes": ADAPTIVE_AVGPOOL_SHAPES, "run": _run_adaptive_avg_pool2d},
    "linear":              {"shapes": LINEAR_SHAPES,           "run": _run_linear},
    "matmul":              {"shapes": MATMUL_SHAPES,           "run": _run_matmul},
    "relu":                {"shapes": ACTIVATION_SHAPES,       "run": _run_activation("relu")},
    "gelu":                {"shapes": ACTIVATION_SHAPES,       "run": _run_activation("gelu")},
    "silu":                {"shapes": ACTIVATION_SHAPES,       "run": _run_activation("silu")},
    "softmax":             {"shapes": SOFTMAX_SHAPES,          "run": _run_softmax},
    "embedding":           {"shapes": EMBEDDING_SHAPES,        "run": _run_embedding},
    "layer_norm":          {"shapes": LAYERNORM_SHAPES,        "run": _run_layer_norm},
    "group_norm":          {"shapes": GROUPNORM_SHAPES,        "run": _run_group_norm},
    "rms_norm":             {"shapes": RMSNORM_SHAPES,          "run": _run_rms_norm},
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _machine_info() -> dict:
    info = {
        "hostname": socket.gethostname(),
        "arch": platform.machine(),
        "platform": platform.platform(),
        "mac_ver": platform.mac_ver()[0],
        "python": platform.python_version(),
    }
    # Best-effort: CPU brand on macOS.
    try:
        import subprocess

        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        info["cpu"] = brand
    except Exception:
        pass
    return info


def _ratio(lucid_v, torch_v):
    if lucid_v is None or torch_v is None:
        return None
    try:
        if torch_v <= 0:
            return None
        return lucid_v / torch_v
    except (TypeError, ZeroDivisionError):
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Lucid 3.4 Phase 0 per-op microbench")
    parser.add_argument(
        "--backend",
        choices=["lucid", "torch", "both"],
        default="both",
        help="Which backend(s) to measure",
    )
    parser.add_argument(
        "--op",
        choices=list(OPS.keys()) + ["all"],
        default="all",
        help="Op (default: every op in the catalogue)",
    )
    parser.add_argument(
        "--dtype",
        choices=["f32", "f16", "both"],
        default="both",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Substring match on shape labels (e.g. 'rn18', 'gpt', 'attn')",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measure", type=int, default=50)
    parser.add_argument(
        "--out",
        type=str,
        default="-",
        help="JSON output path ('-' = stdout, default)",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print per-row progress to stderr",
    )
    args = parser.parse_args(argv)

    backend_names = ["lucid", "torch"] if args.backend == "both" else [args.backend]
    op_names = list(OPS.keys()) if args.op == "all" else [args.op]
    dtypes = ["f32", "f16"] if args.dtype == "both" else [args.dtype]

    backends = {}
    versions = {}
    for name in backend_names:
        try:
            backends[name] = LucidBackend() if name == "lucid" else TorchBackend()
            versions[name] = backends[name].version
        except Exception as e:
            print(f"[init] {name} backend failed: {e}", file=sys.stderr)
            return 2

    results = []
    total_rows = 0
    for op_name in op_names:
        for shape in OPS[op_name]["shapes"]:
            label = shape["label"]
            if args.filter and args.filter not in label:
                continue
            shape_args = {k: v for k, v in shape.items() if k != "label"}
            for dt in dtypes:
                total_rows += 1
                row = {
                    "op": op_name,
                    "shape_label": label,
                    "shape_args": shape_args,
                    "dtype": dt,
                }
                for bname, b in backends.items():
                    try:
                        timings = OPS[op_name]["run"](
                            b, shape_args, dt, args.warmup, args.measure
                        )
                        row[bname] = {"ok": True, **timings}
                    except Exception as e:  # broad — we want to record skips
                        row[bname] = {
                            "ok": False,
                            "error": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc(limit=4),
                        }
                # Compute ratios if both succeeded.
                if "lucid" in row and "torch" in row and row["lucid"].get("ok") and row["torch"].get("ok"):
                    ratio = {}
                    for key in ("fwd_ms", "bwd_ms"):
                        lv = row["lucid"].get(key, {}).get("median") if isinstance(row["lucid"].get(key), dict) else None
                        tv = row["torch"].get(key, {}).get("median") if isinstance(row["torch"].get(key), dict) else None
                        r = _ratio(lv, tv)
                        if r is not None:
                            ratio[key] = r
                    row["ratio_lucid_over_torch"] = ratio
                results.append(row)
                if args.progress:
                    parts = [f"{op_name}/{label}/{dt}"]
                    for bname in backend_names:
                        ent = row.get(bname, {})
                        if ent.get("ok"):
                            f = ent.get("fwd_ms", {}).get("median")
                            bw = ent.get("bwd_ms", {}).get("median")
                            parts.append(
                                f"{bname}=fwd{f:.3f}"
                                + (f"/bwd{bw:.3f}" if bw is not None else "")
                            )
                        else:
                            parts.append(f"{bname}=SKIP")
                    print("  ".join(parts), file=sys.stderr)
                # Release between rows so GPU memory doesn't blow up.
                gc.collect()

    output = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine": _machine_info(),
        "versions": versions,
        "config": {
            "warmup": args.warmup,
            "measure": args.measure,
            "backends": backend_names,
            "ops": op_names,
            "dtypes": dtypes,
            "filter": args.filter,
        },
        "results": results,
    }
    blob = json.dumps(output, indent=2, default=str)
    if args.out == "-":
        print(blob)
    else:
        with open(args.out, "w") as f:
            f.write(blob)
        print(
            f"wrote {args.out} — {len(results)} rows, "
            f"{sum(1 for r in results if all(r.get(b, {}).get('ok') for b in backend_names))} successful",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
