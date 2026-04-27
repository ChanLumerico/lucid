"""Full Phase 6 coverage check — all nn modules + ops + namespaces (CPU+GPU)."""

from __future__ import annotations

import sys, os
# Make sure the worktree root is on sys.path[0] (overrides any site-packages
# `lucid` namespace that might be installed system-wide).
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor import Tensor
from lucid._C import engine as _E


_pass, _fail = 0, 0


def _check(name, ok, detail=""):
    global _pass, _fail
    tag = "PASS " if ok else "FAIL "
    if ok: _pass += 1
    else:  _fail += 1
    print(f"  {tag}  {name}: {detail}")


def gpu_t(arr, rg=False):
    return Tensor._wrap(_E.TensorImpl(arr, _E.Device.GPU, rg))


# === Top-level namespace exposure ===
print("=== top-level namespace ===")
_check("lucid.zeros",      hasattr(lucid, "zeros"))
_check("lucid.arange",     hasattr(lucid, "arange"))
_check("lucid.empty",      hasattr(lucid, "empty"))
_check("lucid.linspace",   hasattr(lucid, "linspace"))
_check("lucid.stack",      hasattr(lucid, "stack"))
_check("lucid.concatenate",hasattr(lucid, "concatenate"))
_check("lucid.matmul",     hasattr(lucid, "matmul"))
_check("lucid.exp",        hasattr(lucid, "exp"))
_check("lucid.sum",        hasattr(lucid, "sum"))
_check("lucid.linalg",     hasattr(lucid, "linalg"))
_check("lucid.random",     hasattr(lucid, "random"))
_check("lucid.einops",     hasattr(lucid, "einops"))
_check("lucid.optim",      hasattr(lucid, "optim"))
_check("lucid.optim.SGD",  hasattr(lucid.optim, "SGD"))
_check("lucid.optim.lr_scheduler.StepLR",
       hasattr(lucid.optim.lr_scheduler, "StepLR"))
_check("lucid.no_grad",    hasattr(lucid, "no_grad"))
_check("lucid.tensor",     hasattr(lucid, "tensor"))
_check("lucid.Float32",    hasattr(lucid, "Float32"))
_check("lucid.Tensor",     hasattr(lucid, "Tensor"))


# === Newly added ops ===
print("\n=== new random/einops ops ===")
lucid.random.seed(123)
p1 = lucid.random.permutation(10)
lucid.random.seed(123)
p2 = lucid.random.permutation(10)
_check("permutation reproducible (seed alias)",
       np.array_equal(p1.numpy(), p2.numpy()))

A = Tensor(np.random.randn(3, 4).astype(np.float32))
B = Tensor(np.random.randn(4, 5).astype(np.float32))
out = lucid.einops.einsum("ij,jk->ik", A, B)
_check("einsum ij,jk->ik",
       np.allclose(out.numpy(), A.numpy() @ B.numpy(), atol=1e-5))


# === nn modules instantiate + forward (CPU) ===
print("\n=== nn modules forward (CPU) ===")
xn = np.random.randn(2, 3, 8, 8).astype(np.float32)

# Conv chain
c = nn.Conv2d(3, 4, 3, padding=1)
y = c(Tensor(xn))
_check("Conv2d", y.shape == (2, 4, 8, 8))

ct = nn.ConvTranspose2d(3, 4, 2, stride=2)
y = ct(Tensor(xn))
_check("ConvTranspose2d", y.shape == (2, 4, 16, 16))

bn = nn.BatchNorm2d(3); bn.train()
y = bn(Tensor(xn))
_check("BatchNorm2d", y.shape == (2, 3, 8, 8))

ln = nn.LayerNorm([3, 8, 8])
y = ln(Tensor(xn))
_check("LayerNorm", y.shape == (2, 3, 8, 8))

gn = nn.GroupNorm(3, 3)
y = gn(Tensor(xn))
_check("GroupNorm", y.shape == (2, 3, 8, 8))

# Pool (default stride=1, output is 7x7 for 8x8 input + 2x2 kernel)
p = nn.MaxPool2d(2, stride=2)
y = p(Tensor(xn))
_check("MaxPool2d", y.shape == (2, 3, 4, 4))

ap = nn.AdaptiveAvgPool2d((4, 4))
y = ap(Tensor(xn))
_check("AdaptiveAvgPool2d", y.shape == (2, 3, 4, 4))

# Linear / Bilinear
flat = Tensor(np.random.randn(4, 16).astype(np.float32))
lin = nn.Linear(16, 8)
_check("Linear", lin(flat).shape == (4, 8))

bil = nn.Bilinear(8, 6, 4)
v1 = Tensor(np.random.randn(3, 8).astype(np.float32))
v2 = Tensor(np.random.randn(3, 6).astype(np.float32))
_check("Bilinear", bil(v1, v2).shape == (3, 4))

# Activation modules
for cls, n in [(nn.ReLU, "ReLU"), (nn.GELU, "GELU"), (nn.Sigmoid, "Sigmoid"),
               (nn.Tanh, "Tanh"), (nn.SELU, "SELU"), (nn.Mish, "Mish"),
               (nn.HardSigmoid, "HardSigmoid"), (nn.HardSwish, "HardSwish")]:
    m = cls(); _check(f"{n}", m(Tensor(xn)).shape == (2, 3, 8, 8))

# Dropout family
m = nn.Dropout(0.5); m.eval()
_check("Dropout eval", m(Tensor(xn)).shape == (2, 3, 8, 8))
m = nn.AlphaDropout(0.5); m.eval()
_check("AlphaDropout eval", m(Tensor(xn)).shape == (2, 3, 8, 8))

# Embedding / pos embedding
emb = nn.Embedding(20, 5, padding_idx=0)
ix = Tensor(np.array([[1, 2, 0, 3], [0, 5, 1, 2]], dtype=np.int64))
_check("Embedding", emb(ix).shape == (2, 4, 5))

# Attention
mha = nn.MultiHeadAttention(embed_dim=16, num_heads=4)
seq = Tensor(np.random.randn(2, 5, 16).astype(np.float32))
res = mha(seq, seq, seq)
out = res[0] if isinstance(res, tuple) else res
_check("MultiHeadAttention", out.shape == (2, 5, 16))

# RNN cells
rnn = nn.RNNCell(8, 16)
h = rnn(Tensor(np.random.randn(2, 8).astype(np.float32)))
_check("RNNCell", h.shape == (2, 16))

lstm = nn.LSTMCell(8, 16)
h, c_ = lstm(Tensor(np.random.randn(2, 8).astype(np.float32)))
_check("LSTMCell", h.shape == (2, 16) and c_.shape == (2, 16))

gru = nn.GRUCell(8, 16)
h = gru(Tensor(np.random.randn(2, 8).astype(np.float32)))
_check("GRUCell", h.shape == (2, 16))

# Loss
target = Tensor(np.random.randint(0, 4, size=(2,)).astype(np.int64))
logits = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
loss = F.cross_entropy(logits, target)
loss.backward()
_check("CE loss + backward", logits.grad.shape == (2, 4))


# === GPU smoke (forward only, just to confirm it doesn't crash) ===
print("\n=== GPU forward smoke ===")
xn_gpu = gpu_t(xn)
c_gpu = nn.Conv2d(3, 4, 3, padding=1)
# Move conv weights to GPU
c_gpu.weight._impl = _E.TensorImpl(c_gpu.weight.numpy(), _E.Device.GPU, True)
if c_gpu.bias is not None:
    c_gpu.bias._impl = _E.TensorImpl(c_gpu.bias.numpy(), _E.Device.GPU, True)
y_gpu = c_gpu(xn_gpu)
_check("Conv2d GPU forward", y_gpu.shape == (2, 4, 8, 8))


# === optim ===
print("\n=== optim smoke ===")
W = nn.Parameter(Tensor(np.random.randn(8, 4).astype(np.float32)))
opt = lucid.optim.SGD([W], lr=0.01)
_check("SGD instantiates", opt is not None)
opt2 = lucid.optim.Adam([W], lr=1e-3)
_check("Adam instantiates", opt2 is not None)
sch = lucid.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
_check("StepLR instantiates", sch is not None)


print(f"\n--- TOTAL: {_pass} passed, {_fail} failed ---")
import sys
sys.exit(0 if _fail == 0 else 1)
