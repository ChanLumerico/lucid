"""Phase 6-9 verifier: interpolate / one_hot / rotate / bilinear (CPU+GPU)."""

from __future__ import annotations

import math
import numpy as np

import lucid
from lucid._tensor import Tensor
from lucid._C import engine as _C_engine
from lucid.ops.bfunc import multiply
from lucid.ops.ufunc import sum as sum_op
import lucid.nn.functional as F


def gpu_t(arr: np.ndarray, requires_grad: bool = False) -> Tensor:
    return Tensor._wrap(_C_engine.TensorImpl(arr, _C_engine.Device.GPU, requires_grad))


_pass = 0
_fail = 0


def _check(name: str, ok: bool, detail: str = "") -> None:
    global _pass, _fail
    tag = "PASS " if ok else "FAIL "
    if ok:
        _pass += 1
    else:
        _fail += 1
    print(f"  {tag}  {name}: {detail}")


# --------------------------------------------------------------------- #
# interpolate (bilinear)
# --------------------------------------------------------------------- #
print("=== interpolate bilinear ===")
np.random.seed(0)
xn = np.random.randn(2, 3, 5, 7).astype(np.float32)
gout = np.random.randn(2, 3, 4, 6).astype(np.float32)
for align in (False, True):
    xc = Tensor(xn, requires_grad=True); xg = gpu_t(xn, True)
    oc = F.interpolate(xc, (4, 6), mode="bilinear", align_corners=align)
    og = F.interpolate(xg, (4, 6), mode="bilinear", align_corners=align)
    sum_op(multiply(oc, Tensor(gout))).backward()
    sum_op(multiply(og, gpu_t(gout))).backward()
    err_fwd = np.abs(oc.data - og.data).max()
    err_grad = np.abs(xc.grad - xg.grad).max()
    _check(f"bilinear align={align} fwd",
           err_fwd < 1e-4, f"err={err_fwd:.2e}")
    _check(f"bilinear align={align} dx",
           err_grad < 1e-4, f"err={err_grad:.2e}")

# Reference vs PyTorch-like formula (CPU only)
def ref_bilinear(x, H_out, W_out, align):
    N, C, H, W = x.shape
    if align:
        ys = np.linspace(0, H - 1, H_out, dtype=np.float32) if H_out > 1 else np.zeros(H_out, np.float32)
        xs = np.linspace(0, W - 1, W_out, dtype=np.float32) if W_out > 1 else np.zeros(W_out, np.float32)
    else:
        ys = (np.arange(H_out, dtype=np.float32) + 0.5) * H / H_out - 0.5
        xs = (np.arange(W_out, dtype=np.float32) + 0.5) * W / W_out - 0.5
    ys = np.clip(ys, 0, H - 1)
    xs = np.clip(xs, 0, W - 1)
    out = np.zeros((N, C, H_out, W_out), np.float32)
    for h, iy in enumerate(ys):
        y0 = int(np.floor(iy)); y1 = min(y0 + 1, H - 1); dy = iy - y0
        for w, ix in enumerate(xs):
            x0 = int(np.floor(ix)); x1 = min(x0 + 1, W - 1); dx = ix - x0
            out[:, :, h, w] = (
                x[:, :, y0, x0] * (1 - dy) * (1 - dx)
              + x[:, :, y0, x1] * (1 - dy) * dx
              + x[:, :, y1, x0] * dy * (1 - dx)
              + x[:, :, y1, x1] * dy * dx
            )
    return out

for align in (False, True):
    ref = ref_bilinear(xn, 4, 6, align)
    out = F.interpolate(Tensor(xn), (4, 6), mode="bilinear", align_corners=align)
    err = np.abs(out.data - ref).max()
    _check(f"bilinear align={align} vs ref", err < 1e-5, f"err={err:.2e}")

# --------------------------------------------------------------------- #
# interpolate (trilinear, 5-D)
# --------------------------------------------------------------------- #
print("\n=== interpolate trilinear ===")
xn5 = np.random.randn(1, 2, 3, 4, 5).astype(np.float32)
gout5 = np.random.randn(1, 2, 2, 3, 4).astype(np.float32)
for align in (False, True):
    xc = Tensor(xn5, requires_grad=True); xg = gpu_t(xn5, True)
    oc = F.interpolate(xc, (2, 3, 4), mode="trilinear", align_corners=align)
    og = F.interpolate(xg, (2, 3, 4), mode="trilinear", align_corners=align)
    sum_op(multiply(oc, Tensor(gout5))).backward()
    sum_op(multiply(og, gpu_t(gout5))).backward()
    _check(f"trilinear align={align} fwd",
           np.allclose(oc.data, og.data, atol=1e-4),
           f"max_err={np.abs(oc.data - og.data).max():.2e}")
    _check(f"trilinear align={align} dx",
           np.allclose(xc.grad, xg.grad, atol=1e-4),
           f"max_err={np.abs(xc.grad - xg.grad).max():.2e}")

# --------------------------------------------------------------------- #
# interpolate (nearest 4-D and 5-D)
# --------------------------------------------------------------------- #
print("\n=== interpolate nearest ===")
xc = Tensor(xn); xg = gpu_t(xn)
oc = F.interpolate(xc, (4, 6), mode="nearest")
og = F.interpolate(xg, (4, 6), mode="nearest")
_check("nearest 4-D match",
       np.allclose(oc.data, og.data, atol=0),
       f"max_err={np.abs(oc.data - og.data).max():.2e}")

xc5 = Tensor(xn5); xg5 = gpu_t(xn5)
oc = F.interpolate(xc5, (2, 3, 4), mode="nearest")
og = F.interpolate(xg5, (2, 3, 4), mode="nearest")
_check("nearest 5-D match",
       np.allclose(oc.data, og.data, atol=0),
       f"max_err={np.abs(oc.data - og.data).max():.2e}")

# --------------------------------------------------------------------- #
# one_hot
# --------------------------------------------------------------------- #
print("\n=== one_hot ===")
ix = np.array([[0, 1, 2], [3, 1, 0]], dtype=np.int64)
oc = F.one_hot(Tensor(ix), num_classes=4)
og = F.one_hot(gpu_t(ix), num_classes=4)
ref = np.eye(4, dtype=np.int8)[ix]
_check("one_hot CPU value", np.allclose(oc.data, ref, atol=0),
       f"shape={oc.shape}")
_check("one_hot GPU vs CPU", np.allclose(np.array(oc.data), np.array(og.data), atol=0),
       f"shape={og.shape}")

# float dtype
from lucid.types import Float32
oc_f = F.one_hot(Tensor(ix), num_classes=4, dtype=Float32)
_check("one_hot float dtype", oc_f.dtype is Float32, "")

# --------------------------------------------------------------------- #
# rotate
# --------------------------------------------------------------------- #
print("\n=== rotate ===")
img = np.random.randn(1, 1, 5, 5).astype(np.float32)
oc = F.rotate(Tensor(img), 0.0)
_check("rotate 0° identity",
       np.allclose(oc.data, img, atol=0),
       f"max_err={np.abs(oc.data - img).max():.2e}")

# 90° rotation should map (h, w) → (w, H-1-h)
oc = F.rotate(Tensor(img), 90.0)
og = F.rotate(gpu_t(img), 90.0)
_check("rotate 90° CPU/GPU match",
       np.allclose(oc.data, og.data, atol=0),
       f"max_err={np.abs(oc.data - og.data).max():.2e}")

# --------------------------------------------------------------------- #
# bilinear layer
# --------------------------------------------------------------------- #
print("\n=== bilinear layer ===")
np.random.seed(7)
B, D1, D2, Dout = 4, 3, 5, 6
x1 = np.random.randn(B, D1).astype(np.float32)
x2 = np.random.randn(B, D2).astype(np.float32)
W = np.random.randn(Dout, D1, D2).astype(np.float32)
b = np.random.randn(Dout).astype(np.float32)
gout_b = np.random.randn(B, Dout).astype(np.float32)

# Reference (numpy einsum)
ref = np.einsum("bi,kij,bj->bk", x1, W, x2) + b

# CPU
x1c = Tensor(x1, requires_grad=True); x2c = Tensor(x2, requires_grad=True)
Wc = Tensor(W, requires_grad=True); bc_ = Tensor(b, requires_grad=True)
oc = F.bilinear(x1c, x2c, Wc, bc_)
err_fwd = np.abs(oc.data - ref).max()
_check("bilinear fwd CPU vs ref", err_fwd < 1e-4, f"err={err_fwd:.2e}")
sum_op(multiply(oc, Tensor(gout_b))).backward()

# Reference grads
dref_W = np.einsum("bk,bi,bj->kij", gout_b, x1, x2)
dref_x1 = np.einsum("bk,kij,bj->bi", gout_b, W, x2)
dref_x2 = np.einsum("bk,kij,bi->bj", gout_b, W, x1)
dref_b = gout_b.sum(axis=0)
_check("bilinear dx1 CPU", np.allclose(x1c.grad, dref_x1, atol=1e-4),
       f"err={np.abs(x1c.grad - dref_x1).max():.2e}")
_check("bilinear dx2 CPU", np.allclose(x2c.grad, dref_x2, atol=1e-4),
       f"err={np.abs(x2c.grad - dref_x2).max():.2e}")
_check("bilinear dW CPU", np.allclose(Wc.grad, dref_W, atol=1e-4),
       f"err={np.abs(Wc.grad - dref_W).max():.2e}")
_check("bilinear db CPU", np.allclose(bc_.grad, dref_b, atol=1e-4),
       f"err={np.abs(bc_.grad - dref_b).max():.2e}")

# GPU
x1g = gpu_t(x1, True); x2g = gpu_t(x2, True)
Wg = gpu_t(W, True); bg = gpu_t(b, True)
og = F.bilinear(x1g, x2g, Wg, bg)
sum_op(multiply(og, gpu_t(gout_b))).backward()
_check("bilinear fwd GPU vs CPU",
       np.allclose(og.data, oc.data, atol=1e-4),
       f"err={np.abs(og.data - oc.data).max():.2e}")
_check("bilinear dx1 GPU", np.allclose(x1g.grad, dref_x1, atol=1e-4), "")
_check("bilinear dW GPU", np.allclose(Wg.grad, dref_W, atol=1e-4), "")
_check("bilinear db GPU", np.allclose(bg.grad, dref_b, atol=1e-4), "")

# Multi-dim batch (..., D1) / (..., D2)
x1nd = np.random.randn(2, 3, D1).astype(np.float32)
x2nd = np.random.randn(2, 3, D2).astype(np.float32)
gnd = np.random.randn(2, 3, Dout).astype(np.float32)
ref_nd = np.einsum("bsi,kij,bsj->bsk", x1nd, W, x2nd) + b
xc1 = Tensor(x1nd, requires_grad=True); xc2 = Tensor(x2nd, requires_grad=True)
Wc2 = Tensor(W, requires_grad=True); bc2 = Tensor(b, requires_grad=True)
oc_nd = F.bilinear(xc1, xc2, Wc2, bc2)
_check("bilinear N-D fwd", np.allclose(oc_nd.data, ref_nd, atol=1e-4),
       f"err={np.abs(oc_nd.data - ref_nd).max():.2e}")
sum_op(multiply(oc_nd, Tensor(gnd))).backward()
dref_nd_W = np.einsum("bsk,bsi,bsj->kij", gnd, x1nd, x2nd)
_check("bilinear N-D dW", np.allclose(Wc2.grad, dref_nd_W, atol=1e-4), "")

# --------------------------------------------------------------------- #
print(f"\n--- TOTAL: {_pass} passed, {_fail} failed ---")
import sys
sys.exit(0 if _fail == 0 else 1)
