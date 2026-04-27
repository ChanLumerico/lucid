#!/usr/bin/env python3
"""Verify C++ utility-op autograd on CPU and GPU."""
import sys

import numpy as np

from lucid._C import engine as eng

PASSED = 0
FAILED = 0


def make_tensor(data, device, requires_grad=False):
    return eng.TensorImpl(np.asarray(data), device, requires_grad)


def grad_np(tensor):
    return np.asarray(tensor.grad_as_python(), dtype=np.float32)


def check(name, got, expected, tol=1e-5):
    global PASSED, FAILED
    got = np.asarray(got, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if got.shape != expected.shape:
        FAILED += 1
        print(f"  FAIL  {name}: shape {got.shape} != expected {expected.shape}")
        return
    err = float(np.abs(got - expected).max(initial=0.0))
    if err <= tol:
        PASSED += 1
        print(f"  PASS  {name}: max|err|={err:.2e}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}: max|err|={err:.2e}")
        print(f"        got={got}")
        print(f"        expected={expected}")


def backward_sum(out):
    eng.engine_backward(eng.sum(out, [], False), retain_graph=False)


def check_where(device):
    cond_np = np.array([[True, False], [False, True]], dtype=np.bool_)
    x_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y_np = np.array([[10, 20], [30, 40]], dtype=np.float32)
    cond = make_tensor(cond_np, device)
    x = make_tensor(x_np, device, True)
    y = make_tensor(y_np, device, True)

    backward_sum(eng.where(cond, x, y))
    check(f"where x ({device.name})", grad_np(x), cond_np.astype(np.float32))
    check(f"where y ({device.name})", grad_np(y), (~cond_np).astype(np.float32))


def check_masked_fill(device):
    mask_np = np.array([[False, True, False], [True, False, False]],
                       dtype=np.bool_)
    a = make_tensor(np.arange(6, dtype=np.float32).reshape(2, 3), device, True)
    mask = make_tensor(mask_np, device)

    backward_sum(eng.masked_fill(a, mask, -7.0))
    check(f"masked_fill ({device.name})", grad_np(a),
          (~mask_np).astype(np.float32))


def check_roll(device):
    base_np = np.arange(6, dtype=np.float32).reshape(2, 3)
    weight_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    a = make_tensor(base_np, device, True)
    weight = make_tensor(weight_np, device)

    out = eng.roll(a, [1, -1], [0, 1])
    backward_sum(eng.mul(out, weight))
    expected = np.roll(weight_np, shift=(-1, 1), axis=(0, 1))
    check(f"roll ({device.name})", grad_np(a), expected)


def check_gather(device):
    src_np = np.arange(8, dtype=np.float32).reshape(2, 4)
    idx_np = np.array([[0, 2, 2], [3, 1, 0]], dtype=np.int64)
    weight_np = np.array([[1, 10, 100], [2, 20, 200]], dtype=np.float32)
    a = make_tensor(src_np, device, True)
    idx = make_tensor(idx_np, device)
    weight = make_tensor(weight_np, device)

    backward_sum(eng.mul(eng.gather(a, idx, 1), weight))
    expected = np.zeros_like(src_np)
    for row in range(idx_np.shape[0]):
        for col in range(idx_np.shape[1]):
            expected[row, idx_np[row, col]] += weight_np[row, col]
    check(f"gather duplicate indices ({device.name})", grad_np(a), expected)


def check_diagonal(device):
    src_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    weight_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    a = make_tensor(src_np, device, True)
    weight = make_tensor(weight_np, device)

    backward_sum(eng.mul(eng.diagonal(a, 1, 1, 2), weight))
    expected = np.zeros_like(src_np)
    for batch in range(2):
        for i in range(3):
            expected[batch, i, i + 1] = weight_np[batch, i]
    check(f"diagonal offset scatter ({device.name})", grad_np(a), expected)


def check_tri(device):
    base_np = np.arange(6, dtype=np.float32).reshape(2, 3)
    weight_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    a = make_tensor(base_np, device, True)
    weight = make_tensor(weight_np, device)
    backward_sum(eng.mul(eng.tril(a, 0), weight))
    check(f"tril ({device.name})", grad_np(a), np.tril(weight_np, 0))

    a = make_tensor(base_np, device, True)
    weight = make_tensor(weight_np, device)
    backward_sum(eng.mul(eng.triu(a, 1), weight))
    check(f"triu ({device.name})", grad_np(a), np.triu(weight_np, 1))


def check_sort_topk(device):
    src_np = np.array([[3, 1, 4, 2], [8, 5, 7, 6]], dtype=np.float32)
    weight_np = np.array([[10, 20, 30, 40], [1, 2, 3, 4]], dtype=np.float32)
    a = make_tensor(src_np, device, True)
    weight = make_tensor(weight_np, device)
    backward_sum(eng.mul(eng.sort(a, 1), weight))
    expected = np.zeros_like(src_np)
    order = np.argsort(src_np, axis=1)
    for row in range(src_np.shape[0]):
        for out_col, in_col in enumerate(order[row]):
            expected[row, in_col] += weight_np[row, out_col]
    check(f"sort ({device.name})", grad_np(a), expected)

    weight_np = np.array([[10, 20], [1, 2]], dtype=np.float32)
    a = make_tensor(src_np, device, True)
    weight = make_tensor(weight_np, device)
    backward_sum(eng.mul(eng.topk(a, 2, 1), weight))
    expected = np.zeros_like(src_np)
    order = np.argsort(src_np, axis=1)[:, ::-1][:, :2]
    for row in range(src_np.shape[0]):
        for out_col, in_col in enumerate(order[row]):
            expected[row, in_col] += weight_np[row, out_col]
    check(f"topk ({device.name})", grad_np(a), expected)


def check_meshgrid(device):
    x_np = np.array([1, 2, 3], dtype=np.float32)
    y_np = np.array([10, 20], dtype=np.float32)
    wx_np = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    wy_np = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)

    x = make_tensor(x_np, device, True)
    y = make_tensor(y_np, device, True)
    gx, gy = eng.meshgrid([x, y], False)
    wx = make_tensor(wx_np, device)
    wy = make_tensor(wy_np, device)
    loss = eng.add(eng.sum(eng.mul(gx, wx), [], False),
                   eng.sum(eng.mul(gy, wy), [], False))
    eng.engine_backward(loss, retain_graph=False)
    check(f"meshgrid x ij ({device.name})", grad_np(x), wx_np.sum(axis=1))
    check(f"meshgrid y ij ({device.name})", grad_np(y), wy_np.sum(axis=0))

    wx_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    wy_np = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)
    x = make_tensor(x_np, device, True)
    y = make_tensor(y_np, device, True)
    gx, gy = eng.meshgrid([x, y], True)
    wx = make_tensor(wx_np, device)
    wy = make_tensor(wy_np, device)
    loss = eng.add(eng.sum(eng.mul(gx, wx), [], False),
                   eng.sum(eng.mul(gy, wy), [], False))
    eng.engine_backward(loss, retain_graph=False)
    check(f"meshgrid x xy ({device.name})", grad_np(x), wx_np.sum(axis=0))
    check(f"meshgrid y xy ({device.name})", grad_np(y), wy_np.sum(axis=1))


def section(title):
    print(f"\n=== {title} ===")


def check_device(device):
    section(f"utils backward {device.name}")
    check_where(device)
    check_masked_fill(device)
    check_roll(device)
    check_gather(device)
    check_diagonal(device)
    check_tri(device)
    check_sort_topk(device)
    check_meshgrid(device)


def main():
    check_device(eng.Device.CPU)
    check_device(eng.Device.GPU)
    print(f"\n--- TOTAL: {PASSED} passed, {FAILED} failed ---")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
