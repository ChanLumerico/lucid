#!/usr/bin/env python3
"""Verify grid_sample GPU forward and backward against the CPU implementation."""
import sys

import numpy as np

from lucid._C import engine as eng

PASSED = 0
FAILED = 0


def make_tensor(data, device, requires_grad=False):
    return eng.TensorImpl(np.asarray(data), device, requires_grad)


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
        print(f"        got={got}")
        print(f"        expected={expected}")


def run_case(name, x_np, grid_np, weight_np, mode, padding_mode,
             align_corners):
    x_cpu = make_tensor(x_np, eng.Device.CPU, True)
    grid_cpu = make_tensor(grid_np, eng.Device.CPU, True)
    weight_cpu = make_tensor(weight_np, eng.Device.CPU)
    out_cpu = eng.nn.grid_sample(
        x_cpu, grid_cpu, mode, padding_mode, align_corners)
    loss_cpu = eng.sum(eng.mul(out_cpu, weight_cpu), [], False)
    eng.engine_backward(loss_cpu, retain_graph=False)

    x_gpu = make_tensor(x_np, eng.Device.GPU, True)
    grid_gpu = make_tensor(grid_np, eng.Device.GPU, True)
    weight_gpu = make_tensor(weight_np, eng.Device.GPU)
    out_gpu = eng.nn.grid_sample(
        x_gpu, grid_gpu, mode, padding_mode, align_corners)
    loss_gpu = eng.sum(eng.mul(out_gpu, weight_gpu), [], False)
    eng.engine_backward(loss_gpu, retain_graph=False)

    check(f"{name} forward", data_np(out_gpu), data_np(out_cpu))
    check(f"{name} input grad", grad_np(x_gpu), grad_np(x_cpu))
    check(f"{name} grid grad", grad_np(grid_gpu), grad_np(grid_cpu))


def main():
    x = (np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4) / 7.0) - 1.0
    grid = np.array(
        [[[
            [-1.0, -1.0],
            [-0.2, 0.3],
            [1.2, -0.5],
        ], [
            [0.5, 0.9],
            [0.0, 0.0],
            [-1.4, 1.3],
        ]]],
        dtype=np.float32)
    weight = (np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3) + 1.0) / 5.0
    run_case("bilinear zeros align_corners=True", x, grid, weight,
             mode=0, padding_mode=0, align_corners=True)

    x = np.array(
        [[[[0.1, 0.5, 1.0],
           [1.5, 2.0, 2.5],
           [3.0, 3.5, 4.0]]]],
        dtype=np.float32)
    grid = np.array(
        [[[
            [-1.2, -0.8],
            [0.2, 0.4],
        ], [
            [1.4, 1.1],
            [-0.4, 0.9],
        ]]],
        dtype=np.float32)
    weight = np.array([[[[1.0, -2.0], [0.5, 3.0]]]], dtype=np.float32)
    run_case("bilinear border align_corners=False", x, grid, weight,
             mode=0, padding_mode=1, align_corners=False)

    x = (np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4) / 3.0)
    grid = np.array(
        [[[
            [-1.0, -1.0],
            [-0.1, 0.1],
            [0.8, -0.6],
        ], [
            [1.4, 0.0],
            [0.2, 1.2],
            [-1.3, 1.3],
        ]]],
        dtype=np.float32)
    weight = np.array(
        [[[[1.0, 2.0, 3.0], [4.0, -1.0, -2.0]]]],
        dtype=np.float32)
    run_case("nearest zeros align_corners=True", x, grid, weight,
             mode=1, padding_mode=0, align_corners=True)

    print(f"\n--- TOTAL: {PASSED} passed, {FAILED} failed ---")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
