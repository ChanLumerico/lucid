#!/usr/bin/env python3
"""bench_transformer_step.py — GPT-style step-time canary for Wave A.

A single transformer block (embedding → LayerNorm → attention → LayerNorm →
GELU-FFN) trained on synthetic token IDs.  Exercises every Wave A op:
GELU exact (FFN), LayerNorm forward (pre/post), Embedding backward
(token + position lookup), Linear (Q/K/V/FFN).

Measures forward + backward + optimizer step wall-clock against the
matching PyTorch MPS implementation.

Run on Mac Studio:
    python bench_transformer_step.py --backend lucid
    python bench_transformer_step.py --backend torch
    python bench_transformer_step.py --backend both
"""

import argparse
import gc
import json
import os
import sys
import time


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


def build_lucid_model(vocab, seq_len, d_model, n_heads, d_ff, device):
    import lucid
    import lucid.nn as nn

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok_embed = nn.Embedding(vocab, d_model)
            self.pos_embed = nn.Embedding(seq_len, d_model)
            self.ln1 = nn.LayerNorm(d_model)
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.proj = nn.Linear(d_model, d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.ffn_up = nn.Linear(d_model, d_ff)
            self.ffn_down = nn.Linear(d_ff, d_model)
            self.head = nn.Linear(d_model, vocab)

        def forward(self, idx):
            import lucid
            import lucid.nn.functional as F

            B, L = idx.shape
            pos = lucid.arange(L, device=device, dtype=lucid.int64)
            x = self.tok_embed(idx) + self.pos_embed(pos)
            h = self.ln1(x)
            qkv = self.qkv(h)
            # Skip the attention compute — exercise GELU/LN/Linear/Embed only
            # to keep this a fair Wave A canary.  Pretend qkv → proj.
            h2 = self.proj(qkv[..., :d_model])
            x = x + h2
            h = self.ln2(x)
            h = F.gelu(self.ffn_up(h))
            h = self.ffn_down(h)
            x = x + h
            return self.head(x)

    return Block()


def build_torch_model(vocab, seq_len, d_model, n_heads, d_ff):
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok_embed = nn.Embedding(vocab, d_model)
            self.pos_embed = nn.Embedding(seq_len, d_model)
            self.ln1 = nn.LayerNorm(d_model)
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.proj = nn.Linear(d_model, d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.ffn_up = nn.Linear(d_model, d_ff)
            self.ffn_down = nn.Linear(d_ff, d_model)
            self.head = nn.Linear(d_model, vocab)

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(L, device=idx.device)
            x = self.tok_embed(idx) + self.pos_embed(pos)
            h = self.ln1(x)
            qkv = self.qkv(h)
            h2 = self.proj(qkv[..., :d_model])
            x = x + h2
            h = self.ln2(x)
            h = torch.nn.functional.gelu(self.ffn_up(h))
            h = self.ffn_down(h)
            x = x + h
            return self.head(x)

    return Block()


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def time_lucid_steps(vocab, seq_len, d_model, n_heads, d_ff, batch, n_warmup, n_measure):
    import lucid
    import lucid.optim as optim
    import lucid.nn.functional as F
    from lucid._C import engine as _C_engine

    device = "metal"
    model = build_lucid_model(vocab, seq_len, d_model, n_heads, d_ff, device)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=3e-4)

    def step(idx, target):
        opt.zero_grad()
        logits = model(idx)
        loss = F.cross_entropy(logits.reshape(-1, vocab), target.reshape(-1))
        loss.backward()
        opt.step()
        impls = [p.grad.impl for p in model.parameters() if p.grad is not None]
        if impls:
            _C_engine.eval_tensors(impls)
        return float(loss.item())

    idx = lucid.randint(0, vocab, (batch, seq_len), device=device)
    target = lucid.randint(0, vocab, (batch, seq_len), device=device)

    for _ in range(n_warmup):
        step(idx, target)
    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter_ns()
        step(idx, target)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    times.sort()
    return times[len(times) // 2] / 1e6, times[int(0.95 * len(times))] / 1e6


def time_torch_steps(vocab, seq_len, d_model, n_heads, d_ff, batch, n_warmup, n_measure):
    import torch
    import torch.nn.functional as F

    device = "mps"
    model = build_torch_model(vocab, seq_len, d_model, n_heads, d_ff).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    def step(idx, target):
        opt.zero_grad()
        logits = model(idx)
        loss = F.cross_entropy(logits.reshape(-1, vocab), target.reshape(-1))
        loss.backward()
        opt.step()
        torch.mps.synchronize()
        return float(loss.item())

    idx = torch.randint(0, vocab, (batch, seq_len), device=device)
    target = torch.randint(0, vocab, (batch, seq_len), device=device)

    for _ in range(n_warmup):
        step(idx, target)
    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter_ns()
        step(idx, target)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    times.sort()
    return times[len(times) // 2] / 1e6, times[int(0.95 * len(times))] / 1e6


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["lucid", "torch", "both"], default="both")
    p.add_argument("--vocab", type=int, default=50257, help="GPT-2 vocab")
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--d-ff", type=int, default=3072)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--measure", type=int, default=20)
    args = p.parse_args(argv)

    cfg = dict(
        vocab=args.vocab, seq_len=args.seq, d_model=args.d_model,
        n_heads=args.n_heads, d_ff=args.d_ff, batch=args.batch,
        n_warmup=args.warmup, n_measure=args.measure,
    )
    print(f"Transformer step (GPT-2-base: vocab={args.vocab}, seq={args.seq}, "
          f"d_model={args.d_model}, d_ff={args.d_ff}, batch={args.batch})")
    print(f"warmup={args.warmup}, measure={args.measure}")
    print()

    if args.backend in ("lucid", "both"):
        try:
            med, p95 = time_lucid_steps(**cfg)
            print(f"  Lucid (3.4-dev):  median = {med:7.2f} ms   p95 = {p95:7.2f} ms")
        except Exception as e:
            print(f"  Lucid: FAILED — {type(e).__name__}: {e}")

    if args.backend in ("torch", "both"):
        try:
            med, p95 = time_torch_steps(**cfg)
            print(f"  PyTorch MPS:      median = {med:7.2f} ms   p95 = {p95:7.2f} ms")
        except Exception as e:
            print(f"  PyTorch: FAILED — {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
