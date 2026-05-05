#!/usr/bin/env python3
"""
LeNet-5 MNIST benchmark: Lucid (Metal/MLX) vs PyTorch (MPS)

Three axes:
  • Speed      — samples/sec per epoch, total wall-clock time
  • Accuracy   — per-epoch train loss + final test accuracy
  • Memory     — peak GPU allocation during training (MiB)

Methodology for fairness:
  1. Weights initialised identically (PyTorch first → numpy → Lucid)
  2. Data shuffled with the same numpy seed; same batches seen in the same order
  3. Same optimiser (Adam, lr=1e-3), same loss (CrossEntropy), same batch size
  4. Both run on the same device (Metal / MPS)
"""

import time, gc, copy
import numpy as np
import sklearn.datasets

import torch
import torch.nn as tnn
import torch.nn.functional as tF
import torch.optim as topt

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as opt
import mlx.core as mx

# ── Config ────────────────────────────────────────────────────────────────────
SEED   = 0
EPOCHS = 5
BATCH  = 64
LR     = 1e-3
TORCH_DEV = torch.device("mps")
LUCID_DEV = "metal"

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading MNIST...", flush=True)
mnist   = sklearn.datasets.fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X_all   = mnist.data.astype(np.float32) / 255.0
Y_all   = mnist.target.astype(np.int64)

X_train_np = X_all[:60000].reshape(-1, 1, 28, 28)
Y_train_np = Y_all[:60000]
X_test_np  = X_all[60000:].reshape(-1, 1, 28, 28)
Y_test_np  = Y_all[60000:]

# Pre-generate identical batch indices for both frameworks
rng          = np.random.default_rng(SEED)
epoch_indices = [rng.permutation(len(X_train_np)) for _ in range(EPOCHS)]
print(f"  train={len(X_train_np):,}  test={len(X_test_np):,}", flush=True)

# ── Architectures ─────────────────────────────────────────────────────────────

class TorchLeNet5(tnn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = tnn.Conv2d(1, 6,  5, padding=2)
        self.conv2 = tnn.Conv2d(6, 16, 5)
        self.fc1   = tnn.Linear(16*5*5, 120)
        self.fc2   = tnn.Linear(120, 84)
        self.fc3   = tnn.Linear(84, 10)
    def forward(self, x):
        x = tF.avg_pool2d(tF.relu(self.conv1(x)), 2)
        x = tF.avg_pool2d(tF.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = tF.relu(self.fc1(x))
        x = tF.relu(self.fc2(x))
        return self.fc3(x)


class LucidLeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6,  5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
    def forward(self, x):
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2)
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ── Weight transfer: PyTorch → Lucid (via numpy) ──────────────────────────────

def copy_weights_to_lucid(torch_model, lucid_model):
    """Copy all parameters from a PyTorch model into the matching Lucid model."""
    t_params = dict(torch_model.named_parameters())
    for name, l_param in lucid_model.named_parameters():
        t_np = t_params[name].detach().cpu().numpy().astype(np.float32)
        l_impl = lucid_model
        parts  = name.split(".")
        for p in parts[:-1]:
            l_impl = getattr(l_impl, p)
        layer_param = getattr(l_impl, parts[-1])
        # Replace impl data in-place
        new_impl = lucid._C.engine.tensor_from_numpy(t_np, lucid._C.engine.Device.CPU, False)
        layer_param._impl = lucid._tensor.tensor._impl_with_grad(new_impl, layer_param._impl.requires_grad)
    return lucid_model

# ── Memory helpers ────────────────────────────────────────────────────────────

def reset_memory():
    """Reset GPU peak counters for both MPS and MLX."""
    torch.mps.empty_cache()
    mx.reset_peak_memory()

def peak_mib_torch():
    return torch.mps.driver_allocated_memory() / (1024**2)

def peak_mib_lucid():
    return mx.get_peak_memory() / (1024**2)

# ── Training helpers ──────────────────────────────────────────────────────────

def train_torch(model, optimizer, criterion, epoch_idx):
    model.train()
    idx      = epoch_indices[epoch_idx]
    tot_loss = 0.0; nb = 0; t0 = time.time()
    for start in range(0, len(X_train_np), BATCH):
        bi  = idx[start:start+BATCH]
        xb  = torch.tensor(X_train_np[bi]).to(TORCH_DEV)
        yb  = torch.tensor(Y_train_np[bi]).to(TORCH_DEV)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item(); nb += 1
    return tot_loss / nb, time.time() - t0


def eval_torch(model):
    model.eval(); correct = 0
    with torch.no_grad():
        for s in range(0, len(X_test_np), 256):
            xb    = torch.tensor(X_test_np[s:s+256]).to(TORCH_DEV)
            preds = model(xb).argmax(dim=1).cpu().numpy()
            correct += (preds == Y_test_np[s:s+256]).sum()
    return correct / len(Y_test_np) * 100


def train_lucid(model, optimizer, criterion, epoch_idx):
    model.train()
    idx      = epoch_indices[epoch_idx]
    tot_loss = 0.0; nb = 0; t0 = time.time()
    for start in range(0, len(X_train_np), BATCH):
        bi  = idx[start:start+BATCH]
        xb  = lucid.tensor(X_train_np[bi], device=LUCID_DEV)
        yb  = lucid.tensor(Y_train_np[bi].astype(np.int32), dtype=lucid.int32, device=LUCID_DEV)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        # Flush forward graph before backward so MLX evaluates a smaller
        # graph at each stage rather than one giant fused graph.
        loss.eval()
        loss.backward()
        optimizer.step()
        lucid.eval(*model.parameters())   # flush param updates
        tot_loss += float(loss.item()); nb += 1
    return tot_loss / nb, time.time() - t0


def eval_lucid(model):
    model.eval(); correct = 0
    with lucid.no_grad():
        for s in range(0, len(X_test_np), 256):
            xb    = lucid.tensor(X_test_np[s:s+256], device=LUCID_DEV)
            preds = model(xb).numpy().argmax(axis=1)
            correct += (preds == Y_test_np[s:s+256]).sum()
    return correct / len(Y_test_np) * 100

# ── Run experiments ───────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# 1. PyTorch
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  PyTorch (MPS)")
print("="*55)

torch.manual_seed(SEED)
torch_model = TorchLeNet5().to(TORCH_DEV)
torch_opt   = topt.Adam(torch_model.parameters(), lr=LR)
torch_crit  = tnn.CrossEntropyLoss()

reset_memory()
torch_epoch_times  = []
torch_epoch_losses = []
torch_total_t0     = time.time()

for ep in range(EPOCHS):
    loss, t = train_torch(torch_model, torch_opt, torch_crit, ep)
    torch_epoch_times.append(t)
    torch_epoch_losses.append(loss)
    speed = len(X_train_np) / t
    print(f"  Epoch {ep+1}/{EPOCHS}  loss={loss:.4f}  {t:.1f}s  ({speed:.0f} smp/s)", flush=True)

torch_total_time = time.time() - torch_total_t0
torch_peak_mem   = peak_mib_torch()
torch_acc        = eval_torch(torch_model)
print(f"  Test acc: {torch_acc:.2f}%  |  Peak GPU: {torch_peak_mem:.1f} MiB  |  Total: {torch_total_time:.1f}s")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Lucid  (same initial weights as PyTorch)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  Lucid (Metal/MLX)")
print("="*55)

lucid_model = LucidLeNet5()

# Transfer initial weights from the just-trained PyTorch model's INITIAL state?
# No — we want the SAME INITIAL weights, so we re-init PyTorch with same seed
# and copy those starting weights.
torch.manual_seed(SEED)
torch_ref   = TorchLeNet5()   # CPU, same seed → same init

t_params    = {n: p.detach().numpy().astype(np.float32)
               for n, p in torch_ref.named_parameters()}

# Inject into Lucid model layer by layer
from lucid._C import engine as _ce
from lucid._tensor.tensor import _impl_with_grad as _iwg

for name, lucid_param in lucid_model.named_parameters():
    arr = t_params[name]
    new_impl = _ce.TensorImpl(arr, _ce.Device.CPU, False)
    lucid_param._impl = _iwg(new_impl, lucid_param._impl.requires_grad)

lucid_model = lucid_model.to(LUCID_DEV)
lucid_opt   = opt.Adam(lucid_model.parameters(), lr=LR)
lucid_crit  = nn.CrossEntropyLoss()

reset_memory()
lucid_epoch_times  = []
lucid_epoch_losses = []
lucid_total_t0     = time.time()

for ep in range(EPOCHS):
    loss, t = train_lucid(lucid_model, lucid_opt, lucid_crit, ep)
    lucid_epoch_times.append(t)
    lucid_epoch_losses.append(loss)
    speed = len(X_train_np) / t
    print(f"  Epoch {ep+1}/{EPOCHS}  loss={loss:.4f}  {t:.1f}s  ({speed:.0f} smp/s)", flush=True)

lucid_total_time = time.time() - lucid_total_t0
lucid_peak_mem   = peak_mib_lucid()
lucid_acc        = eval_lucid(lucid_model)
print(f"  Test acc: {lucid_acc:.2f}%  |  Peak GPU: {lucid_peak_mem:.1f} MiB  |  Total: {lucid_total_time:.1f}s")

# ── Summary table ─────────────────────────────────────────────────────────────

def bar(val, ref, width=20, higher_is_better=True):
    ratio = val / ref if ref else 1.0
    filled = int(min(ratio, 2.0) * width / 2)
    color = "\033[92m" if (ratio >= 1 and higher_is_better) or (ratio <= 1 and not higher_is_better) else "\033[91m"
    reset = "\033[0m"
    return f"{color}{'█'*filled}{'░'*(width-filled)}{reset}"

print("\n\n" + "═"*65)
print("  BENCHMARK SUMMARY")
print("═"*65)

# Speed
avg_t  = np.mean(torch_epoch_times)
avg_l  = np.mean(lucid_epoch_times)
sps_t  = len(X_train_np) / avg_t
sps_l  = len(X_train_np) / avg_l
faster = (sps_t / sps_l - 1) * 100 if sps_l < sps_t else (sps_l / sps_t - 1) * 100
faster_who = "PyTorch" if sps_t > sps_l else "Lucid"
print(f"\n  ── SPEED (samples/sec, higher=better) ──")
print(f"  PyTorch  {sps_t:7.0f} smp/s  total {torch_total_time:.1f}s")
print(f"  Lucid    {sps_l:7.0f} smp/s  total {lucid_total_time:.1f}s")
print(f"  → {faster_who} is {faster:.1f}% faster")

# Accuracy
acc_diff = torch_acc - lucid_acc
print(f"\n  ── ACCURACY (test %, higher=better) ──")
print(f"  PyTorch  {torch_acc:.2f}%")
print(f"  Lucid    {lucid_acc:.2f}%")
print(f"  → {'PyTorch' if acc_diff > 0 else 'Lucid'} higher by {abs(acc_diff):.2f}pp")

# Memory
mem_diff = (lucid_peak_mem / torch_peak_mem - 1) * 100 if torch_peak_mem else 0
print(f"\n  ── MEMORY (peak GPU MiB, lower=better) ──")
print(f"  PyTorch  {torch_peak_mem:.1f} MiB   (MPS driver alloc)")
print(f"  Lucid    {lucid_peak_mem:.1f} MiB   (MLX metal peak)")
print(f"  Note: MPS and MLX report memory differently; absolute values")
print(f"        are not directly comparable (see methodology note below)")

# Per-epoch loss curve
print(f"\n  ── LOSS CURVE ──")
print(f"  {'Epoch':<6} {'PyTorch':>10} {'Lucid':>10} {'Δ':>8}")
print(f"  {'─'*36}")
for ep in range(EPOCHS):
    diff = lucid_epoch_losses[ep] - torch_epoch_losses[ep]
    sign = "+" if diff > 0 else ""
    print(f"  {ep+1:<6} {torch_epoch_losses[ep]:>10.4f} {lucid_epoch_losses[ep]:>10.4f} {sign}{diff:>7.4f}")

print("\n" + "═"*65)
print("  Methodology note")
print("  ─────────────────────────────────────────────────────────────")
print("  • Same architecture, same initial weights (PyTorch seed=0 → numpy → Lucid)")
print("  • Same batch order (pre-generated numpy indices, seed=0)")
print("  • Memory: torch.mps.driver_allocated_memory vs mx.metal.get_peak_memory")
print("    measure different things — driver total vs MLX heap peak.")
print("    Use as relative reference within each framework, not cross-framework.")
print("═"*65)
