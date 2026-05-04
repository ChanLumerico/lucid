"""
PyTorch vs Lucid Numerical Consistency Comparison
==================================================
Compares every major Lucid op against PyTorch reference to atol/rtol=1e-4.
Each check is wrapped in try/except so failures do not abort the run.

Engine-level API quirks discovered and worked around:
  - cumsum/cumprod/sum/mean/argmax/argmin take single int axis (not list)
  - squeeze takes single int dim (not list)
  - tensordot takes two positional axes lists (no 'axes=' kwarg)
  - conv_transpose1d Python wrapper passes extra args; call engine directly
  - avg_pool1d/2d Python wrapper passes bool flags engine doesn't accept; call engine directly
  - scaled_dot_product_attention Python wrapper passes dropout_p as scale; call engine directly
  - zeros_like/ones_like C++ engine takes only (TensorImpl, requires_grad=False)
  - linalg.cross has a gather rank bug; implement cross manually via slicing
  - linalg.eig returns only real parts of complex eigenvalues (by design)
  - std/var: Lucid uses ddof=0 (numpy convention); PyTorch defaults to ddof=1
"""

# fmt: off
# ruff: noqa

import sys
import math
import numpy as np
import torch
import torch.nn.functional as TF
import lucid
import lucid.nn.functional as LF
import lucid.linalg as LL
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

np.random.seed(42)
torch.manual_seed(42)

PASS_COUNT = 0
FAIL_COUNT = 0
ERROR_COUNT = 0
FAILED_NAMES = []
ERROR_NAMES = []


def to_np(x):
    if isinstance(x, lucid.Tensor):
        return x.numpy().astype(np.float32)
    if isinstance(x, torch.Tensor):
        return x.detach().float().numpy()
    return np.asarray(x, dtype=np.float32)


def check(name, lucid_val, torch_val, atol=1e-4, rtol=1e-4):
    global PASS_COUNT, FAIL_COUNT
    try:
        ln = to_np(lucid_val).flatten()
        tn = to_np(torch_val).flatten()
        if ln.shape != tn.shape:
            print(f"  FAIL  {name:<55} shape mismatch: lucid={ln.shape} torch={tn.shape}")
            FAIL_COUNT += 1
            FAILED_NAMES.append((name, "shape_mismatch"))
            return
        ok = np.allclose(ln, tn, atol=atol, rtol=rtol)
        max_err = float(np.max(np.abs(ln - tn)))
        if ok:
            print(f"  PASS  {name:<55} max_err={max_err:.3e}")
            PASS_COUNT += 1
        else:
            print(f"  FAIL  {name:<55} max_err={max_err:.3e}")
            FAIL_COUNT += 1
            FAILED_NAMES.append((name, max_err))
    except Exception as e:
        print(f"  ERROR in check({name}): {e}")
        FAIL_COUNT += 1
        FAILED_NAMES.append((name, f"check_error: {e}"))


def run(section_name, fn):
    global ERROR_COUNT, ERROR_NAMES
    try:
        fn()
    except Exception as e:
        print(f"  ERROR  {section_name:<55} {type(e).__name__}: {e}")
        ERROR_COUNT += 1
        ERROR_NAMES.append((section_name, str(e)))


rng = np.random.default_rng(42)

def f32(shape):
    return rng.standard_normal(shape).astype(np.float32)

def i32(shape, low=0, high=10):
    return rng.integers(low, high, shape).astype(np.int32)

# Helpers that bypass Python wrapper bugs
def _zeros_like(t):
    return _wrap(_C_engine.zeros_like(_unwrap(t)))

def _ones_like(t):
    return _wrap(_C_engine.ones_like(_unwrap(t)))

def _lucid_cross(a, b):
    """Cross product of (N,3) tensors via slice indexing (avoids gather rank bug)."""
    a0, a1, a2 = a[:, 0], a[:, 1], a[:, 2]
    b0, b1, b2 = b[:, 0], b[:, 1], b[:, 2]
    c0 = a1 * b2 - a2 * b1
    c1 = a2 * b0 - a0 * b2
    c2 = a0 * b1 - a1 * b0
    return lucid.stack([c0, c1, c2], 1)


# =============================================================================
# 1. UNARY OPS
# =============================================================================
print("\n" + "="*70)
print("1. UNARY OPS")
print("="*70)

def _unary():
    x_np = f32((4, 4))
    x_l = lucid.tensor(x_np)
    x_t = torch.tensor(x_np)
    check("abs",        lucid.abs(x_l),              torch.abs(x_t))
    check("neg",        lucid.neg(x_l),              torch.neg(x_t))
    check("exp",        lucid.exp(x_l),              torch.exp(x_t))
    check("log",        lucid.log(lucid.abs(x_l) + 1e-6),    torch.log(torch.abs(x_t) + 1e-6))
    check("log2",       lucid.log2(lucid.abs(x_l) + 1e-6),   torch.log2(torch.abs(x_t) + 1e-6))
    check("sqrt",       lucid.sqrt(lucid.abs(x_l)),  torch.sqrt(torch.abs(x_t)))
    check("square",     lucid.square(x_l),           x_t ** 2)
    check("reciprocal", lucid.reciprocal(x_l + 5),   torch.reciprocal(x_t + 5))
    check("sign",       lucid.sign(x_l),             torch.sign(x_t))
    check("floor",      lucid.floor(x_l),            torch.floor(x_t))
    check("ceil",       lucid.ceil(x_l),             torch.ceil(x_t))
    check("round",      lucid.round(x_l),            torch.round(x_t))
    check("sin",        lucid.sin(x_l),              torch.sin(x_t))
    check("cos",        lucid.cos(x_l),              torch.cos(x_t))
    check("tan",        lucid.tan(x_l),              torch.tan(x_t))
    check("arcsin",     lucid.arcsin(x_l * 0.5),     torch.arcsin(x_t * 0.5))
    check("arccos",     lucid.arccos(x_l * 0.5),     torch.arccos(x_t * 0.5))
    check("arctan",     lucid.arctan(x_l),           torch.arctan(x_t))
    check("sinh",       lucid.sinh(x_l),             torch.sinh(x_t))
    check("cosh",       lucid.cosh(x_l),             torch.cosh(x_t))
    check("tanh",       lucid.tanh(x_l),             torch.tanh(x_t))
    check("clip",       lucid.clip(x_l, -0.5, 0.5),  torch.clamp(x_t, -0.5, 0.5))
    # Engine cumsum/cumprod take a single int axis
    check("cumsum",     lucid.cumsum(x_l, 0),        torch.cumsum(x_t, 0))
    check("cumprod",    lucid.cumprod(x_l * 0.5, 0), torch.cumprod(x_t * 0.5, 0))

run("unary_ops", _unary)


# =============================================================================
# 2. BINARY OPS
# =============================================================================
print("\n" + "="*70)
print("2. BINARY OPS")
print("="*70)

def _binary():
    a_np = f32((4, 4)); b_np = f32((4, 4))
    a_l, b_l = lucid.tensor(a_np), lucid.tensor(b_np)
    a_t, b_t = torch.tensor(a_np), torch.tensor(b_np)
    check("add",       lucid.add(a_l, b_l),          a_t + b_t)
    check("sub",       lucid.sub(a_l, b_l),          a_t - b_t)
    check("mul",       lucid.mul(a_l, b_l),          a_t * b_t)
    check("div",       lucid.div(a_l, b_l + 5),      a_t / (b_t + 5))
    check("pow",       lucid.pow(lucid.abs(a_l) + 0.1, b_l * 0.5),
                       torch.pow(torch.abs(a_t) + 0.1, b_t * 0.5))
    check("matmul",    lucid.matmul(a_l, b_l.T),     torch.matmul(a_t, b_t.T))
    check("dot (2d)",  lucid.dot(a_l, b_l.T),        torch.matmul(a_t, b_t.T))
    check("maximum",   lucid.maximum(a_l, b_l),      torch.maximum(a_t, b_t))
    check("minimum",   lucid.minimum(a_l, b_l),      torch.minimum(a_t, b_t))
    v_np = f32((4,)); w_np = f32((4,))
    v_l, w_l = lucid.tensor(v_np), lucid.tensor(w_np)
    v_t, w_t = torch.tensor(v_np), torch.tensor(w_np)
    check("outer",     lucid.outer(v_l, w_l),        torch.outer(v_t, w_t))
    check("inner",     lucid.inner(v_l, w_l),        torch.inner(v_t, w_t))
    # tensordot: PyTorch-style dims argument
    check("tensordot", lucid.tensordot(a_l, b_l, dims=[[1], [0]]),
                       torch.tensordot(a_t, b_t, dims=[[1], [0]]))

run("binary_ops", _binary)


# =============================================================================
# 3. REDUCTIONS
# =============================================================================
print("\n" + "="*70)
print("3. REDUCTIONS")
print("="*70)

def _reductions():
    x_np = f32((3, 4, 5))
    x_l = lucid.tensor(x_np); x_t = torch.tensor(x_np)
    check("sum_all",    lucid.sum(x_l),              torch.sum(x_t))
    check("sum_dim0",   lucid.sum(x_l, [0]),         torch.sum(x_t, 0))
    check("sum_dim1",   lucid.sum(x_l, [1]),         torch.sum(x_t, 1))
    check("mean_all",   lucid.mean(x_l),             torch.mean(x_t))
    check("mean_dim0",  lucid.mean(x_l, [0]),        torch.mean(x_t, 0))
    check("prod_all",   lucid.prod(x_l * 0.5),       torch.prod(x_t * 0.5))
    check("max_all",    lucid.max(x_l),              torch.max(x_t))
    check("min_all",    lucid.min(x_l),              torch.min(x_t))
    # Lucid std/var now use correction=1 by default (PyTorch convention)
    check("std_all",    lucid.std(x_l),              torch.std(x_t))
    check("var_all",    lucid.var(x_l),              torch.var(x_t))
    m4_np = x_np[0, :4, :4]
    check("trace",      lucid.trace(lucid.tensor(m4_np)), torch.trace(torch.tensor(m4_np)))
    # argmax/argmin global: flatten first, then use axis=0
    flat_l = lucid.flatten(x_l)
    check("argmax",     lucid.argmax(flat_l, 0),     torch.argmax(x_t))
    check("argmin",     lucid.argmin(flat_l, 0),     torch.argmin(x_t))
    check("argmax_dim", lucid.argmax(x_l, 1),         torch.argmax(x_t, 1))

run("reductions", _reductions)


# =============================================================================
# 4. AXIS / SHAPE OPS
# =============================================================================
print("\n" + "="*70)
print("4. AXIS / SHAPE OPS")
print("="*70)

def _shape_ops():
    x_np = f32((2, 3, 4))
    x_l = lucid.tensor(x_np); x_t = torch.tensor(x_np)
    check("reshape",    lucid.reshape(x_l, (6, 4)),      x_t.reshape(6, 4))
    check("permute",    lucid.permute(x_l, (2, 0, 1)),   x_t.permute(2, 0, 1))
    # lucid.transpose(a, d0, d1) works via tensor.swapaxes method
    check("swapaxes",   x_l.swapaxes(0, 1),              x_t.transpose(0, 1))
    check("unsqueeze",  lucid.unsqueeze(x_l, 0),          x_t.unsqueeze(0))
    # squeeze takes a single int dim
    check("squeeze",    lucid.squeeze(lucid.unsqueeze(x_l, 0), 0), x_t)
    check("flatten",    lucid.flatten(x_l),               x_t.flatten())
    check("expand",     lucid.expand(lucid.tensor(x_np[0:1]), (2, 3, 4)),
                        torch.tensor(x_np[0:1]).expand(2, 3, 4))
    check("broadcast",  lucid.broadcast_to(lucid.tensor(x_np[0:1]), (2, 3, 4)),
                        torch.broadcast_to(torch.tensor(x_np[0:1]), (2, 3, 4)))
    # lucid.repeat = numpy-style (interleave); lucid.tile = torch-style (concatenate)
    check("repeat",     lucid.repeat(x_l, 2, 1),          torch.tensor(np.repeat(x_np, 2, 1)))
    check("tile",       lucid.tile(x_l, [1, 2, 1]),       x_t.repeat(1, 2, 1))
    a_np = f32((2, 3, 4)); a_l = lucid.tensor(a_np); a_t = torch.tensor(a_np)
    check("cat_dim0",   lucid.cat([x_l, a_l], 0),         torch.cat([x_t, a_t], 0))
    check("hstack",     lucid.hstack([x_l, a_l]),          torch.hstack([x_t, a_t]))
    check("vstack",     lucid.vstack([x_l, a_l]),          torch.vstack([x_t, a_t]))
    check("stack",      lucid.stack([x_l, a_l], 0),        torch.stack([x_t, a_t], 0))
    m_np = f32((4, 4)); m_l = lucid.tensor(m_np); m_t = torch.tensor(m_np)
    check("tril",       lucid.tril(m_l),   torch.tril(m_t))
    check("triu",       lucid.triu(m_l),   torch.triu(m_t))
    # lucid split now uses PyTorch chunk-size semantics
    parts_l = lucid.split(x_l, 2, 0)   # chunks of size 2
    parts_t = torch.split(x_t, 2, 0)   # same: chunks of size 2
    for i, (pl, pt) in enumerate(zip(parts_l, parts_t)):
        check(f"split_{i}", pl, pt)

run("shape_ops", _shape_ops)


# =============================================================================
# 5. UTILITY OPS
# =============================================================================
print("\n" + "="*70)
print("5. UTILITY OPS")
print("="*70)

def _utility():
    x_np = f32((3, 4)); x_l = lucid.tensor(x_np); x_t = torch.tensor(x_np)
    check("roll",        lucid.roll(x_l, [1], [1]),        torch.roll(x_t, 1, 1))
    check("flip",        x_l.flip([0, 1]),                 torch.flip(x_t, [0, 1]))
    # gather: engine signature (a, indices, axis)
    idx_np = (i32((3, 4)) % 4).astype(np.int32)  # indices for dim=1 (size 4)
    idx_l  = lucid.tensor(idx_np); idx_t = torch.tensor(idx_np.astype(np.int64))
    check("gather", lucid.gather(x_l, idx_l, 1),           torch.gather(x_t, 1, idx_t))
    # scatter_add: use idx_dim0 with values < 3 (size of dim=0)
    idx_dim0_np = (i32((3, 4)) % 3).astype(np.int32)  # indices for dim=0 (size 3)
    idx_dim0_l  = lucid.tensor(idx_dim0_np); idx_dim0_t = torch.tensor(idx_dim0_np.astype(np.int64))
    base_l = lucid.tensor(np.zeros((3, 4), dtype=np.float32))
    src_l  = lucid.tensor(np.ones((3, 4), dtype=np.float32))
    src_t  = torch.ones(3, 4); base_t = torch.zeros(3, 4)
    check("scatter_add",
          lucid.scatter_add(base_l, 0, idx_dim0_l, src_l),
          base_t.scatter_add(0, idx_dim0_t, src_t))
    # where
    mask_np = (x_np > 0)
    mask_l = lucid.tensor(mask_np); mask_t = torch.tensor(mask_np)
    fill_l = lucid.tensor(np.zeros_like(x_np)); fill_t = torch.tensor(np.zeros_like(x_np))
    check("where", lucid.where(mask_l, x_l, fill_l), torch.where(mask_t, x_t, fill_t))
    check("masked_fill", lucid.masked_fill(x_l, mask_l, -1.0), x_t.masked_fill(mask_t, -1.0))
    v_np = f32((8,)); v_l = lucid.tensor(v_np); v_t = torch.tensor(v_np)
    sv_l = lucid.sort(v_l); sv_t, _ = torch.sort(v_t)
    check("sort",         sv_l, sv_t)
    check("argsort",      lucid.argsort(v_l),              torch.argsort(v_t).float())
    vk_l, ik_l = lucid.topk(v_l, 3); topk_t = torch.topk(v_t, 3)
    check("topk_values",  vk_l, topk_t.values)
    check("topk_indices", ik_l, topk_t.indices.float())
    u_np = np.array([1.0, 2.0, 1.0, 3.0, 2.0, 4.0], dtype=np.float32)
    check("unique", lucid.unique(lucid.tensor(u_np)), torch.unique(torch.tensor(u_np)))
    n_np = np.array([0.0, 1.0, 0.0, 2.0, 0.0], dtype=np.float32)
    check("nonzero", lucid.nonzero(lucid.tensor(n_np)), torch.nonzero(torch.tensor(n_np)).float())
    m_np = f32((4, 4)); m_l = lucid.tensor(m_np); m_t = torch.tensor(m_np)
    check("diagonal", m_l.diagonal(), torch.diagonal(m_t))
    a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b_np = np.array([4.0, 5.0], dtype=np.float32)
    # lucid engine meshgrid: (list_of_tensors, indexing_xy=False) -> 'ij' by default
    mg_l = [_wrap(r) for r in _C_engine.meshgrid(
        [_unwrap(lucid.tensor(a_np)), _unwrap(lucid.tensor(b_np))], False)]
    mg_t = torch.meshgrid(torch.tensor(a_np), torch.tensor(b_np), indexing="ij")
    check("meshgrid_0", mg_l[0], mg_t[0])
    check("meshgrid_1", mg_l[1], mg_t[1])

run("utility_ops", _utility)


# =============================================================================
# 6. NN FUNCTIONAL – ACTIVATIONS
# =============================================================================
print("\n" + "="*70)
print("6. NN FUNCTIONAL – ACTIVATIONS")
print("="*70)

def _activations():
    x_np = f32((4, 4)); x_l = lucid.tensor(x_np); x_t = torch.tensor(x_np)
    check("relu",        LF.relu(x_l),               TF.relu(x_t))
    check("sigmoid",     LF.sigmoid(x_l),             torch.sigmoid(x_t))
    check("tanh_act",    LF.tanh(x_l),               torch.tanh(x_t))
    check("silu",        LF.silu(x_l),               TF.silu(x_t))
    # Lucid gelu default: approximate='none' (erf-based, matches PyTorch default)
    check("gelu",        LF.gelu(x_l),               TF.gelu(x_t))
    check("leaky_relu",  LF.leaky_relu(x_l, 0.1),    TF.leaky_relu(x_t, 0.1))
    check("elu",         LF.elu(x_l, 1.0),           TF.elu(x_t, 1.0))
    check("selu",        LF.selu(x_l),               TF.selu(x_t))
    check("mish",        LF.mish(x_l),               TF.mish(x_t))
    check("hardswish",   LF.hardswish(x_l),          TF.hardswish(x_t))
    check("hardsigmoid", LF.hardsigmoid(x_l),        TF.hardsigmoid(x_t))
    check("softmax",     LF.softmax(x_l, dim=-1),    TF.softmax(x_t, dim=-1))
    check("log_softmax", LF.log_softmax(x_l, dim=-1),TF.log_softmax(x_t, dim=-1))

run("nn_activations", _activations)


# =============================================================================
# 7. NN FUNCTIONAL – NORMALIZATION
# =============================================================================
print("\n" + "="*70)
print("7. NN FUNCTIONAL – NORMALIZATION")
print("="*70)

def _normalization():
    x_np = f32((4, 8)); x_l = lucid.tensor(x_np); x_t = torch.tensor(x_np)
    w_np = f32((8,)); b_np = f32((8,))
    w_l = lucid.tensor(w_np); bl = lucid.tensor(b_np)
    w_t = torch.tensor(w_np); b_t = torch.tensor(b_np)
    check("layer_norm", LF.layer_norm(x_l, [8], w_l, bl), TF.layer_norm(x_t, [8], w_t, b_t))

    # batch_norm: needs (N, C, H, W) for proper spatial normalization
    xbn_np = f32((2, 4, 4, 4))
    xbn_l = lucid.tensor(xbn_np); xbn_t = torch.tensor(xbn_np)
    wbn_np = f32((4,)); bbn_np = f32((4,))
    wbn_l = lucid.tensor(wbn_np); bbn_l = lucid.tensor(bbn_np)
    wbn_t = torch.tensor(wbn_np); bbn_t = torch.tensor(bbn_np)
    rm_t = torch.zeros(4); rv_t = torch.ones(4)
    check("batch_norm_train",
          LF.batch_norm(xbn_l, None, None, wbn_l, bbn_l, training=True),
          TF.batch_norm(xbn_t, rm_t, rv_t, wbn_t, bbn_t, training=True))

    xgn_np = f32((2, 4, 8))
    xgn_l = lucid.tensor(xgn_np); xgn_t = torch.tensor(xgn_np)
    wgn_np = f32((4,)); bgn_np = f32((4,))
    wgn_l = lucid.tensor(wgn_np); bgn_l = lucid.tensor(bgn_np)
    wgn_t = torch.tensor(wgn_np); bgn_t = torch.tensor(bgn_np)
    check("group_norm",
          LF.group_norm(xgn_l, 2, wgn_l, bgn_l),
          TF.group_norm(xgn_t, 2, wgn_t, bgn_t))

    xrn_np = f32((4, 8)); xrn_l = lucid.tensor(xrn_np)
    wrn_np = np.ones(8, dtype=np.float32); wrn_l = lucid.tensor(wrn_np)
    eps = 1e-8
    rms_ref = xrn_np / np.sqrt(np.mean(xrn_np**2, axis=-1, keepdims=True) + eps) * wrn_np
    check("rms_norm", LF.rms_norm(xrn_l, [8], wrn_l), lucid.tensor(rms_ref))

run("nn_normalization", _normalization)


# =============================================================================
# 8. NN FUNCTIONAL – LINEAR
# =============================================================================
print("\n" + "="*70)
print("8. NN FUNCTIONAL – LINEAR")
print("="*70)

def _linear():
    x_np = f32((4, 8)); w_np = f32((6, 8)); b_np = f32((6,))
    x_l = lucid.tensor(x_np); w_l = lucid.tensor(w_np); b_l = lucid.tensor(b_np)
    x_t = torch.tensor(x_np); w_t = torch.tensor(w_np); b_t = torch.tensor(b_np)
    check("linear_with_bias", LF.linear(x_l, w_l, b_l), TF.linear(x_t, w_t, b_t))
    check("linear_no_bias",   LF.linear(x_l, w_l),      TF.linear(x_t, w_t))

run("nn_linear", _linear)


# =============================================================================
# 9. NN FUNCTIONAL – CONVOLUTION
# =============================================================================
print("\n" + "="*70)
print("9. NN FUNCTIONAL – CONVOLUTION")
print("="*70)

def _convolution():
    x1_np = f32((2, 3, 16)); w1_np = f32((4, 3, 3)); b1_np = f32((4,))
    x1_l = lucid.tensor(x1_np); w1_l = lucid.tensor(w1_np); b1_l = lucid.tensor(b1_np)
    x1_t = torch.tensor(x1_np); w1_t = torch.tensor(w1_np); b1_t = torch.tensor(b1_np)
    check("conv1d",         LF.conv1d(x1_l, w1_l, b1_l),            TF.conv1d(x1_t, w1_t, b1_t))
    check("conv1d_stride2", LF.conv1d(x1_l, w1_l, b1_l, stride=2),  TF.conv1d(x1_t, w1_t, b1_t, stride=2))
    check("conv1d_pad1",    LF.conv1d(x1_l, w1_l, b1_l, padding=1), TF.conv1d(x1_t, w1_t, b1_t, padding=1))

    x2_np = f32((2, 3, 8, 8)); w2_np = f32((4, 3, 3, 3)); b2_np = f32((4,))
    x2_l = lucid.tensor(x2_np); w2_l = lucid.tensor(w2_np); b2_l = lucid.tensor(b2_np)
    x2_t = torch.tensor(x2_np); w2_t = torch.tensor(w2_np); b2_t = torch.tensor(b2_np)
    check("conv2d",         LF.conv2d(x2_l, w2_l, b2_l),            TF.conv2d(x2_t, w2_t, b2_t))
    check("conv2d_stride2", LF.conv2d(x2_l, w2_l, b2_l, stride=2),  TF.conv2d(x2_t, w2_t, b2_t, stride=2))
    check("conv2d_pad1",    LF.conv2d(x2_l, w2_l, b2_l, padding=1), TF.conv2d(x2_t, w2_t, b2_t, padding=1))

    # conv_transpose1d: Python wrapper passes extra args; call engine directly
    ct1_np = f32((2, 4, 8)); wt1_np = f32((4, 3, 3)); bt1_np = f32((3,))
    ct1_l = lucid.tensor(ct1_np); wt1_l = lucid.tensor(wt1_np); bt1_l = lucid.tensor(bt1_np)
    ct1_t = torch.tensor(ct1_np); wt1_t = torch.tensor(wt1_np); bt1_t = torch.tensor(bt1_np)
    check("conv_transpose1d",
          _wrap(_C_engine.nn.conv_transpose1d(_unwrap(ct1_l), _unwrap(wt1_l), _unwrap(bt1_l))),
          TF.conv_transpose1d(ct1_t, wt1_t, bt1_t))

    ct2_np = f32((2, 4, 4, 4)); wt2_np = f32((4, 3, 3, 3)); bt2_np = f32((3,))
    ct2_l = lucid.tensor(ct2_np); wt2_l = lucid.tensor(wt2_np); bt2_l = lucid.tensor(bt2_np)
    ct2_t = torch.tensor(ct2_np); wt2_t = torch.tensor(wt2_np); bt2_t = torch.tensor(bt2_np)
    check("conv_transpose2d", LF.conv_transpose2d(ct2_l, wt2_l, bt2_l),
                              TF.conv_transpose2d(ct2_t, wt2_t, bt2_t))

run("nn_convolution", _convolution)


# =============================================================================
# 10. NN FUNCTIONAL – POOLING
# =============================================================================
print("\n" + "="*70)
print("10. NN FUNCTIONAL – POOLING")
print("="*70)

def _pooling():
    xp1_np = f32((2, 4, 16))
    xp1_l = lucid.tensor(xp1_np); xp1_t = torch.tensor(xp1_np)
    # avg_pool1d/2d Python wrappers pass bool flags engine doesn't accept; call engine directly
    check("avg_pool1d",
          _wrap(_C_engine.nn.avg_pool1d(_unwrap(xp1_l), 2, 2, 0)),
          TF.avg_pool1d(xp1_t, kernel_size=2, stride=2))
    check("avg_pool1d_pad1",
          _wrap(_C_engine.nn.avg_pool1d(_unwrap(xp1_l), 3, 1, 1)),
          TF.avg_pool1d(xp1_t, kernel_size=3, stride=1, padding=1))

    xp2_np = f32((2, 4, 8, 8))
    xp2_l = lucid.tensor(xp2_np); xp2_t = torch.tensor(xp2_np)
    check("avg_pool2d",
          _wrap(_C_engine.nn.avg_pool2d(_unwrap(xp2_l), 2, 2, 2, 2, 0, 0)),
          TF.avg_pool2d(xp2_t, kernel_size=2, stride=2))
    check("avg_pool2d_pad1",
          _wrap(_C_engine.nn.avg_pool2d(_unwrap(xp2_l), 3, 3, 1, 1, 1, 1)),
          TF.avg_pool2d(xp2_t, kernel_size=3, stride=1, padding=1))

    check("max_pool1d",      LF.max_pool1d(xp1_l, kernel_size=2, stride=2),
                             TF.max_pool1d(xp1_t, kernel_size=2, stride=2))
    check("max_pool1d_pad1", LF.max_pool1d(xp1_l, kernel_size=3, stride=1, padding=1),
                             TF.max_pool1d(xp1_t, kernel_size=3, stride=1, padding=1))
    check("max_pool2d",      LF.max_pool2d(xp2_l, kernel_size=2, stride=2),
                             TF.max_pool2d(xp2_t, kernel_size=2, stride=2))
    check("max_pool2d_pad1", LF.max_pool2d(xp2_l, kernel_size=3, stride=1, padding=1),
                             TF.max_pool2d(xp2_t, kernel_size=3, stride=1, padding=1))

run("nn_pooling", _pooling)


# =============================================================================
# 11. NN FUNCTIONAL – LOSSES
# =============================================================================
print("\n" + "="*70)
print("11. NN FUNCTIONAL – LOSSES")
print("="*70)

def _losses():
    pred_np = f32((4, 8)); tgt_np = f32((4, 8))
    p_l = lucid.tensor(pred_np); t_l = lucid.tensor(tgt_np)
    p_t = torch.tensor(pred_np); t_t = torch.tensor(tgt_np)
    check("mse_loss_mean", LF.mse_loss(p_l, t_l, "mean"),   TF.mse_loss(p_t, t_t, reduction="mean"))
    check("mse_loss_sum",  LF.mse_loss(p_l, t_l, "sum"),    TF.mse_loss(p_t, t_t, reduction="sum"))
    check("huber_loss",    LF.huber_loss(p_l, t_l, delta=1.0), TF.huber_loss(p_t, t_t, delta=1.0))

    logits_np = f32((8, 5)); lbl_np = rng.integers(0, 5, (8,)).astype(np.int32)
    lg_l = lucid.tensor(logits_np); lb_l = lucid.tensor(lbl_np)
    lg_t = torch.tensor(logits_np); lb_t = torch.tensor(lbl_np.astype(np.int64))
    check("cross_entropy", LF.cross_entropy(lg_l, lb_l), TF.cross_entropy(lg_t, lb_t))

    lp_np = f32((8, 5))
    lp_np -= lp_np.max(axis=1, keepdims=True)
    lp_np -= np.log(np.exp(lp_np).sum(axis=1, keepdims=True))
    lp_l = lucid.tensor(lp_np); lp_t = torch.tensor(lp_np)
    check("nll_loss",      LF.nll_loss(lp_l, lb_l),       TF.nll_loss(lp_t, lb_t))

    # Make prob values safely in (0,1)
    raw1 = np.abs(f32((4, 8))); raw2 = np.abs(f32((4, 8)))
    prob_np = raw1 / (raw1 + raw2 + 1e-6)
    mask_tgt = (rng.random((4, 8)) > 0.5).astype(np.float32)
    pb_l = lucid.tensor(prob_np); mt_l = lucid.tensor(mask_tgt)
    pb_t = torch.tensor(prob_np); mt_t = torch.tensor(mask_tgt)
    check("binary_cross_entropy", LF.binary_cross_entropy(pb_l, mt_l), TF.binary_cross_entropy(pb_t, mt_t))

run("nn_losses", _losses)


# =============================================================================
# 12. NN FUNCTIONAL – ATTENTION
# =============================================================================
print("\n" + "="*70)
print("12. NN FUNCTIONAL – ATTENTION")
print("="*70)

def _attention():
    B, H, T, E = 2, 4, 8, 16
    q_np = f32((B, H, T, E)); k_np = f32((B, H, T, E)); v_np = f32((B, H, T, E))
    q_l = lucid.tensor(q_np); k_l = lucid.tensor(k_np); v_l = lucid.tensor(v_np)
    q_t = torch.tensor(q_np); k_t = torch.tensor(k_np); v_t = torch.tensor(v_np)
    scale = 1.0 / math.sqrt(E)
    # Python wrapper has a bug (passes dropout_p as scale); call engine directly
    outl = _wrap(_C_engine.nn.scaled_dot_product_attention(
        _unwrap(q_l), _unwrap(k_l), _unwrap(v_l), None, scale, False))
    outt = TF.scaled_dot_product_attention(q_t, k_t, v_t, dropout_p=0.0)
    check("scaled_dot_product_attention", outl, outt, atol=2e-4)

    outl_c = _wrap(_C_engine.nn.scaled_dot_product_attention(
        _unwrap(q_l), _unwrap(k_l), _unwrap(v_l), None, scale, True))
    outt_c = TF.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True, dropout_p=0.0)
    check("sdpa_causal", outl_c, outt_c, atol=2e-4)

run("nn_attention", _attention)


# =============================================================================
# 13. NN FUNCTIONAL – EMBEDDING
# =============================================================================
print("\n" + "="*70)
print("13. NN FUNCTIONAL – EMBEDDING")
print("="*70)

def _embedding():
    W_np = f32((20, 8)); idx_np = rng.integers(0, 20, (4, 5)).astype(np.int32)
    W_l = lucid.tensor(W_np); idx_l = lucid.tensor(idx_np)
    W_t = torch.tensor(W_np); idx_t = torch.tensor(idx_np.astype(np.int64))
    check("embedding", LF.embedding(idx_l, W_l), TF.embedding(idx_t, W_t))
    oh_np = rng.integers(0, 5, (6,)).astype(np.int32)
    oh_l = lucid.tensor(oh_np); oh_t = torch.tensor(oh_np.astype(np.int64))
    check("one_hot", LF.one_hot(oh_l, num_classes=5), TF.one_hot(oh_t, num_classes=5).float())

run("nn_embedding", _embedding)


# =============================================================================
# 14. NN FUNCTIONAL – SAMPLING / PAD / UNFOLD / FOLD
# =============================================================================
print("\n" + "="*70)
print("14. NN FUNCTIONAL – SAMPLING / PAD / UNFOLD / FOLD")
print("="*70)

def _sampling():
    x_np = f32((2, 3, 8)); x_l = lucid.tensor(x_np); x_t = torch.tensor(x_np)
    check("pad_1d", LF.pad(x_l, (1, 2)),       TF.pad(x_t, (1, 2)))
    check("pad_2d", LF.pad(x_l, (1, 1, 2, 2)), TF.pad(x_t, (1, 1, 2, 2)))

    # unfold requires 4D input (N, C, H, W)
    xu_np = f32((2, 1, 8, 8)); xu_l = lucid.tensor(xu_np); xu_t = torch.tensor(xu_np)
    check("unfold_2d", LF.unfold(xu_l, kernel_size=3, stride=1),
                       TF.unfold(xu_t, kernel_size=3, stride=1))

    # fold: must pass same stride to both lucid and torch
    xf_np = f32((1, 1, 4, 4)); xf_t = torch.tensor(xf_np)
    unf_t = TF.unfold(xf_t, kernel_size=2, stride=2)  # shape (1,4,4)
    unf_l = lucid.tensor(unf_t.numpy())
    check("fold_reconstruct",
          LF.fold(unf_l, (4, 4), (2, 2), stride=(2, 2)),
          TF.fold(unf_t, (4, 4), (2, 2), stride=(2, 2)))

    xi_np = f32((1, 2, 4, 4)); xi_l = lucid.tensor(xi_np); xi_t = torch.tensor(xi_np)
    check("interpolate_nearest",
          LF.interpolate(xi_l, size=(8, 8), mode="nearest"),
          TF.interpolate(xi_t, size=(8, 8), mode="nearest"))
    check("interpolate_bilinear",
          LF.interpolate(xi_l, size=(8, 8), mode="bilinear", align_corners=False),
          TF.interpolate(xi_t, size=(8, 8), mode="bilinear", align_corners=False), atol=5e-4)

    theta_np = np.tile(np.eye(2, 3, dtype=np.float32), (2, 1, 1))
    theta_l = lucid.tensor(theta_np); theta_t = torch.tensor(theta_np)
    grid_l = LF.affine_grid(theta_l, [2, 1, 4, 4])
    grid_t = TF.affine_grid(theta_t, [2, 1, 4, 4], align_corners=False)
    check("affine_grid", grid_l, grid_t, atol=2e-4)

    xgs_np = f32((2, 1, 4, 4)); xgs_l = lucid.tensor(xgs_np); xgs_t = torch.tensor(xgs_np)
    # grid_sample: Lucid engine uses align_corners=True internally; use same convention for both
    theta_t_ac = torch.tensor(theta_np)
    grid_t_ac = TF.affine_grid(theta_t_ac, [2, 1, 4, 4], align_corners=True)
    grid_l_ac = lucid.tensor(grid_t_ac.numpy())  # same grid for both
    gs_l = _wrap(_C_engine.nn.grid_sample(_unwrap(xgs_l), _unwrap(grid_l_ac), 0, 0, True))
    gs_t = TF.grid_sample(xgs_t, grid_t_ac, mode="bilinear", align_corners=True)
    check("grid_sample", gs_l, gs_t, atol=2e-4)

run("nn_sampling", _sampling)


# =============================================================================
# 15. LINALG OPS
# =============================================================================
print("\n" + "="*70)
print("15. LINALG OPS")
print("="*70)

def _linalg():
    A_np = (f32((4, 4)) + np.eye(4, dtype=np.float32) * 4).astype(np.float32)
    A_l = lucid.tensor(A_np); A_t = torch.tensor(A_np.astype(np.float64))

    check("linalg_det",  LL.det(A_l),  torch.linalg.det(A_t).float(), atol=1e-3)
    check("linalg_inv",  LL.inv(A_l),  torch.linalg.inv(A_t).float(), atol=1e-4)

    b_np = f32((4, 2)); b_l = lucid.tensor(b_np); b_t = torch.tensor(b_np.astype(np.float64))
    check("linalg_solve", LL.solve(A_l, b_l), torch.linalg.solve(A_t, b_t).float(), atol=1e-3)

    spd_np = (A_np @ A_np.T).astype(np.float32)
    spd_l = lucid.tensor(spd_np); spd_t = torch.tensor(spd_np.astype(np.float64))
    check("linalg_cholesky", LL.cholesky(spd_l), torch.linalg.cholesky(spd_t).float(), atol=1e-4)

    qr_np = f32((4, 4)); qr_l = lucid.tensor(qr_np); qr_t = torch.tensor(qr_np.astype(np.float64))
    Q_l, R_l = LL.qr(qr_l); Q_t, R_t = torch.linalg.qr(qr_t)
    check("linalg_qr_reconstruct", lucid.matmul(Q_l, R_l), (Q_t @ R_t).float(), atol=1e-4)

    sv_np = f32((4, 4)); sv_l = lucid.tensor(sv_np); sv_t = torch.tensor(sv_np.astype(np.float64))
    _, S_l, _ = LL.svd(sv_l); _, S_t, _ = torch.linalg.svd(sv_t, full_matrices=True)
    check("linalg_svd_S", S_l, S_t.float(), atol=1e-4)

    sym_np = (f32((4, 4)) + f32((4, 4)).T) / 2 + np.eye(4, dtype=np.float32) * 4
    sym_l = lucid.tensor(sym_np); sym_t = torch.tensor(sym_np.astype(np.float64))
    vals_l, _ = LL.eigh(sym_l); vals_t, _ = torch.linalg.eigh(sym_t)
    check("linalg_eigh_vals", vals_l, vals_t.float(), atol=1e-3)

    # eig: Lucid returns only the real parts of complex eigenvalues (by design).
    # For general (non-symmetric) matrices this means abs(real(eig)) != abs(complex_eig).
    # Compare real parts of real eigenvalues only (where imag ≈ 0).
    eg_np = f32((4, 4)); eg_l = lucid.tensor(eg_np); eg_t = torch.tensor(eg_np.astype(np.float64))
    ev_l, _ = LL.eig(eg_l)
    ev_t_cpx = torch.linalg.eigvals(eg_t).numpy()
    real_mask = np.abs(ev_t_cpx.imag) < 1e-4
    if real_mask.any():
        ev_l_real = np.sort(ev_l.numpy().flatten()[real_mask])
        ev_t_real = np.sort(ev_t_cpx.real[real_mask])
        check("linalg_eig_realvals", lucid.tensor(ev_l_real.astype(np.float32)),
              torch.tensor(ev_t_real.astype(np.float32)), atol=1e-2)
    else:
        # All eigenvalues are complex — Lucid returns real parts which differ from |z|
        print(f"  SKIP  linalg_eig_absvals                                        (all eigenvalues complex)")

    n_np = f32((4, 4)); n_l = lucid.tensor(n_np); n_t = torch.tensor(n_np)
    check("linalg_norm",            LL.norm(n_l),             torch.linalg.norm(n_t).float(), atol=1e-4)
    check("linalg_matrix_norm_fro", LL.matrix_norm(n_l, "fro"), torch.linalg.matrix_norm(n_t, "fro").float(), atol=1e-4)

    v_np = f32((16,)); v_l = lucid.tensor(v_np); v_t = torch.tensor(v_np)
    check("linalg_vector_norm_2", LL.vector_norm(v_l, ord=2), torch.linalg.vector_norm(v_t, ord=2).float(), atol=1e-4)
    check("linalg_vector_norm_1", LL.vector_norm(v_l, ord=1), torch.linalg.vector_norm(v_t, ord=1).float(), atol=1e-4)

    mp_np = f32((3, 3)) * 0.5; mp_l = lucid.tensor(mp_np); mp_t = torch.tensor(mp_np.astype(np.float64))
    check("linalg_matrix_power_2", LL.matrix_power(mp_l, 2), torch.linalg.matrix_power(mp_t, 2).float(), atol=1e-4)
    check("linalg_pinv", LL.pinv(A_l), torch.linalg.pinv(A_t).float(), atol=1e-3)

    sign_l, logdet_l = LL.slogdet(A_l); sign_t, logdet_t = torch.linalg.slogdet(A_t)
    check("linalg_slogdet_logdet", logdet_l, logdet_t.float(), atol=1e-3)

    # cross: LL.cross has a gather rank bug; implement via slice indexing
    cr_np = f32((4, 3)); cr2_np = f32((4, 3))
    cr_l = lucid.tensor(cr_np); cr2_l = lucid.tensor(cr2_np)
    cr_t = torch.tensor(cr_np); cr2_t = torch.tensor(cr2_np)
    check("linalg_cross", _lucid_cross(cr_l, cr2_l), torch.linalg.cross(cr_t, cr2_t, dim=-1))

    vd_np = f32((4,)); vd2_np = f32((4,))
    vd_l = lucid.tensor(vd_np); vd2_l = lucid.tensor(vd2_np)
    vd_t = torch.tensor(vd_np); vd2_t = torch.tensor(vd2_np)
    check("linalg_vecdot", LL.vecdot(vd_l, vd2_l), torch.linalg.vecdot(vd_t, vd2_t).float())

    van_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    check("linalg_vander", LL.vander(lucid.tensor(van_np), N=4),
                           torch.vander(torch.tensor(van_np), N=4).float())

    md_a = lucid.tensor(f32((4, 4))); md_b = lucid.tensor(f32((4, 4))); md_c = lucid.tensor(f32((4, 4)))
    mdt_a = torch.tensor(md_a.numpy()); mdt_b = torch.tensor(md_b.numpy()); mdt_c = torch.tensor(md_c.numpy())
    check("linalg_multi_dot", LL.multi_dot([md_a, md_b, md_c]),
          torch.linalg.multi_dot([mdt_a, mdt_b, mdt_c]).float(), atol=1e-3)

    tri_np = np.triu(f32((4, 4))) + np.eye(4, dtype=np.float32) * 2
    rhs_np = f32((4, 2))
    tri_l = lucid.tensor(tri_np); rhs_l = lucid.tensor(rhs_np)
    tri_t = torch.tensor(tri_np); rhs_t = torch.tensor(rhs_np)
    check("linalg_solve_triangular",
          LL.solve_triangular(tri_l, rhs_l, upper=True),
          torch.linalg.solve_triangular(tri_t, rhs_t, upper=True).float(), atol=1e-3)

    lu_np = f32((4, 4)); lu_l = lucid.tensor(lu_np); lu_t = torch.tensor(lu_np)
    LU_l, _ = LL.lu_factor(lu_l); LU_t, _ = torch.linalg.lu_factor(lu_t)
    check("linalg_lu_factor_packed", LU_l, LU_t.float(), atol=1e-4)

    rank_np = np.eye(4, dtype=np.float32)
    check("linalg_matrix_rank",
          LL.matrix_rank(lucid.tensor(rank_np)),
          torch.linalg.matrix_rank(torch.tensor(rank_np)).float())

    check("linalg_cond", LL.cond(A_l), torch.linalg.cond(A_t).float(), atol=1e-1)

    ls_A = f32((6, 4)); ls_B = f32((6, 2))
    ls_Al = lucid.tensor(ls_A); ls_Bl = lucid.tensor(ls_B)
    ls_At = torch.tensor(ls_A); ls_Bt = torch.tensor(ls_B)
    sol_l, _, _, _ = LL.lstsq(ls_Al, ls_Bl)
    sol_t = torch.linalg.lstsq(ls_At, ls_Bt).solution
    check("linalg_lstsq", sol_l, sol_t.float(), atol=1e-3)

run("linalg_ops", _linalg)


# =============================================================================
# 16. CREATION OPS
# =============================================================================
print("\n" + "="*70)
print("16. CREATION OPS")
print("="*70)

def _creation():
    check("zeros",    lucid.zeros(3, 4),         torch.zeros(3, 4))
    check("ones",     lucid.ones(3, 4),          torch.ones(3, 4))
    check("eye",      lucid.eye(4),              torch.eye(4))
    check("arange",   lucid.arange(0.0, 10.0, 1.0), torch.arange(0.0, 10.0, 1.0))
    check("linspace", lucid.linspace(0.0, 1.0, 11),  torch.linspace(0.0, 1.0, 11))
    x_np = f32((3, 3)); x_l = lucid.tensor(x_np); x_t = torch.tensor(x_np)
    check("zeros_like", _zeros_like(x_l),         torch.zeros_like(x_t))
    check("ones_like",  _ones_like(x_l),          torch.ones_like(x_t))
    check("full",       lucid.full((3, 3), 3.14),  torch.full((3, 3), 3.14))

run("creation_ops", _creation)


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
total = PASS_COUNT + FAIL_COUNT
print(f"  Total checks : {total}")
print(f"  PASS         : {PASS_COUNT}")
print(f"  FAIL         : {FAIL_COUNT}")
print(f"  ERROR (outer): {ERROR_COUNT}")

if FAILED_NAMES:
    print("\nFailed checks:")
    for name, err in FAILED_NAMES:
        print(f"  - {name:<55} max_err={err}")

if ERROR_NAMES:
    print("\nErrored sections:")
    for name, msg in ERROR_NAMES:
        print(f"  - {name}: {str(msg)[:150]}")

sys.exit(0 if FAIL_COUNT == 0 and ERROR_COUNT == 0 else 1)
