#!/usr/bin/env python3
"""Verify GPU layout ops and unfold autograd."""
import itertools
import sys

import numpy as np

from lucid._C import engine as _C_engine

PASSED = 0
FAILED = 0


def make_tensor(data, device, requires_grad=False):
    return _C_engine.TensorImpl(np.asarray(data), device, requires_grad)


def data_np(tensor):
    return np.asarray(tensor.data_as_python(), dtype=np.float32)


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


def backward_sum(out):
    _C_engine.engine_backward(_C_engine.sum(out, [], False), retain_graph=False)


def unfold_np(x, kernel, stride, pad, dilation):
    batch, channels = x.shape[:2]
    spatial = x.shape[2:]
    rank = len(kernel)
    out_spatial = [
        (s + 2 * p - (d * (k - 1) + 1)) // st + 1
        for s, k, st, p, d in zip(spatial, kernel, stride, pad, dilation)
    ]
    k_total = int(np.prod(kernel))
    o_total = int(np.prod(out_spatial))
    out = np.zeros((batch, channels * k_total, o_total), dtype=x.dtype)

    kernel_ranges = [range(k) for k in kernel]
    output_ranges = [range(o) for o in out_spatial]
    for b in range(batch):
        for c in range(channels):
            for k_flat, k_coord in enumerate(itertools.product(*kernel_ranges)):
                row = c * k_total + k_flat
                for o_flat, o_coord in enumerate(itertools.product(*output_ranges)):
                    src = tuple(
                        o_coord[d] * stride[d] - pad[d] +
                        k_coord[d] * dilation[d]
                        for d in range(rank)
                    )
                    if all(0 <= src[d] < spatial[d] for d in range(rank)):
                        out[(b, row, o_flat)] = x[(b, c, *src)]
    return out


def col2im_np(cols, input_shape, kernel, stride, pad, dilation):
    batch, channels = input_shape[:2]
    spatial = input_shape[2:]
    rank = len(kernel)
    out_spatial = [
        (s + 2 * p - (d * (k - 1) + 1)) // st + 1
        for s, k, st, p, d in zip(spatial, kernel, stride, pad, dilation)
    ]
    k_total = int(np.prod(kernel))
    dx = np.zeros(input_shape, dtype=cols.dtype)

    kernel_ranges = [range(k) for k in kernel]
    output_ranges = [range(o) for o in out_spatial]
    for b in range(batch):
        for c in range(channels):
            for k_flat, k_coord in enumerate(itertools.product(*kernel_ranges)):
                row = c * k_total + k_flat
                for o_flat, o_coord in enumerate(itertools.product(*output_ranges)):
                    src = tuple(
                        o_coord[d] * stride[d] - pad[d] +
                        k_coord[d] * dilation[d]
                        for d in range(rank)
                    )
                    if all(0 <= src[d] < spatial[d] for d in range(rank)):
                        dx[(b, c, *src)] += cols[(b, row, o_flat)]
    return dx


def check_device(device):
    print(f"\n=== layout/unfold {device.name} ===")

    base = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    perm = [2, 0, 1]
    weight = np.arange(24, dtype=np.float32).reshape(4, 2, 3) / 7.0
    x = make_tensor(base, device, True)
    y = _C_engine.permute(x, perm)
    check(f"permute forward ({device.name})", data_np(y),
          np.transpose(base, perm))
    backward_sum(_C_engine.mul(y, make_tensor(weight, device)))
    check(f"permute backward ({device.name})", grad_np(x),
          np.transpose(weight, np.argsort(perm)))

    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    weight = np.arange(12, dtype=np.float32).reshape(4, 3) / 5.0
    x = make_tensor(base, device, True)
    y = _C_engine.contiguous(_C_engine.transpose(x))
    check(f"contiguous transpose forward ({device.name})", data_np(y), base.T)
    backward_sum(_C_engine.mul(y, make_tensor(weight, device)))
    check(f"contiguous transpose backward ({device.name})", grad_np(x),
          weight.T)

    x_np = np.arange(40, dtype=np.float32).reshape(1, 2, 4, 5) / 10.0
    kernel = [2, 3]
    stride = [1, 2]
    pad = [1, 1]
    dilation = [1, 1]
    expected = unfold_np(x_np, kernel, stride, pad, dilation)
    weight = np.arange(expected.size, dtype=np.float32).reshape(expected.shape) / 13.0
    x = make_tensor(x_np, device, True)
    y = _C_engine.nn.unfold(x, kernel, stride, pad, dilation)
    check(f"unfold2d forward ({device.name})", data_np(y), expected)
    backward_sum(_C_engine.mul(y, make_tensor(weight, device)))
    check(f"unfold2d backward ({device.name})", grad_np(x),
          col2im_np(weight, x_np.shape, kernel, stride, pad, dilation))


def main():
    check_device(_C_engine.Device.CPU)
    check_device(_C_engine.Device.GPU)
    print(f"\n--- TOTAL: {PASSED} passed, {FAILED} failed ---")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
