#!/usr/bin/env python3
"""
LeNet-5 on MNIST — Lucid framework
Usage: python train_lenet_mnist.py [--device cpu|metal] [--epochs 5] [--batch 64]
"""

import argparse
import time
import numpy as np

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="metal", choices=["cpu", "metal"])
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

DEV = args.device
EPOCHS = args.epochs
BATCH = args.batch
LR = args.lr

# ── Data ──────────────────────────────────────────────────────────────────────

print("Loading MNIST...", flush=True)
try:
    import sklearn.datasets

    mnist = sklearn.datasets.fetch_openml(
        "mnist_784", version=1, as_frame=False, parser="auto"
    )
    X_all = mnist.data.astype(np.float32) / 255.0
    Y_all = mnist.target.astype(np.int32)
except Exception as e:
    raise RuntimeError(f"Could not load MNIST via sklearn: {e}")

X_train = X_all[:60000].reshape(-1, 1, 28, 28)
Y_train = Y_all[:60000]
X_test = X_all[60000:].reshape(-1, 1, 28, 28)
Y_test = Y_all[60000:]
print(f"  train={len(X_train):,}  test={len(X_test):,}", flush=True)

# ── Model ─────────────────────────────────────────────────────────────────────


class LeNet5(nn.Module):
    """Original LeNet-5: (N,1,28,28) → 10 logits."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # → (N,6,28,28)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # → (N,16,10,10)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.avg_pool2d(F.relu(self.conv1(x)), kernel_size=2)  # → (N,6,14,14)
        x = F.avg_pool2d(F.relu(self.conv2(x)), kernel_size=2)  # → (N,16,5,5)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Seed for reproducibility (weight init + data shuffle)
lucid.manual_seed(0)
np.random.seed(0)

model = LeNet5().to(DEV)
opt = optim.Adam(model.parameters(), lr=LR)
crit = nn.CrossEntropyLoss()
rng = np.random.default_rng(0)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nLeNet-5  |  params={total_params:,}  |  device={DEV}  |  lr={LR}", flush=True)
print("=" * 55, flush=True)

# ── Training ──────────────────────────────────────────────────────────────────


def run_epoch(epoch_idx):
    model.train()
    idx = rng.permutation(len(X_train))
    tot_loss = 0.0
    n_batches = 0
    t0 = time.time()

    for start in range(0, len(X_train), BATCH):
        bi = idx[start : start + BATCH]
        xb = lucid.tensor(X_train[bi], device=DEV)
        yb = lucid.tensor(Y_train[bi], dtype=lucid.int32, device=DEV)

        opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()   # auto-flushes forward graph on Metal before running
        opt.step()        # auto-flushes param updates on Metal after running

        tot_loss += float(loss.item())
        n_batches += 1

    elapsed = time.time() - t0
    avg_loss = tot_loss / n_batches
    speed = len(X_train) / elapsed

    print(
        f"Epoch {epoch_idx+1}/{EPOCHS}  "
        f"loss={avg_loss:.4f}  "
        f"time={elapsed:.1f}s  "
        f"({speed:.0f} samples/s)",
        flush=True,
    )
    return avg_loss


def evaluate():
    model.eval()
    correct = 0
    with lucid.no_grad():
        for start in range(0, len(X_test), 256):
            xb = lucid.tensor(X_test[start : start + 256], device=DEV)
            preds = model(xb).numpy().argmax(axis=1)
            correct += (preds == Y_test[start : start + 256]).sum()
    return correct / len(Y_test) * 100


# ── Main loop ─────────────────────────────────────────────────────────────────

t_total = time.time()
for epoch in range(EPOCHS):
    run_epoch(epoch)

print("=" * 55)
print("Evaluating on test set...", flush=True)
acc = evaluate()
print(f"Test accuracy: {acc:.2f}%")
print(f"Total time:    {time.time()-t_total:.1f}s")
