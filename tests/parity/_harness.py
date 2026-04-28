"""
Parity harness — runs an :class:`OpSpec` through 6 verification axes.

The DUT is the C++ engine (`lucid._C.engine`); the reference is PyTorch.
Every spec is exercised on:

    1. CPU forward      vs torch
    2. GPU forward      vs torch         (skipped if skip_gpu)
    3. CPU vs GPU       forward          (skipped if skip_gpu)
    4. CPU backward     vs torch         (skipped if skip_grad)
    5. GPU backward     vs torch         (skipped if skip_grad or skip_gpu)
    6. CPU vs GPU       backward         (skipped if skip_grad or skip_gpu)

Backward axes drive the engine via ``engine_backward(scalar)`` after a
sum-to-scalar (or ``spec.post_fn``). Torch's gradient path is the reference.

Failures raise ``AssertionError`` with the spec name, axis, and the
shape/dtype/device combination that diverged.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from lucid._C import engine as E

from ._specs import OpSpec


# --------------------------------------------------------------------------- #
# Input generation
# --------------------------------------------------------------------------- #

def _make_numpy_inputs(spec: OpSpec) -> list[np.ndarray]:
    """Materialize per-spec numpy inputs (deterministic via spec.seed)."""
    rng = np.random.default_rng(spec.seed)
    if spec.input_gen is not None:
        arrs = spec.input_gen(rng)
    else:
        arrs = []
        for shape in spec.input_shapes:
            if np.dtype(spec.dtype).kind in "iu":
                # Integers in a small safe range — avoid overflow in matmul/sum.
                a = rng.integers(-8, 8, size=tuple(shape)).astype(spec.dtype)
            else:
                a = rng.standard_normal(size=tuple(shape)).astype(spec.dtype)
            arrs.append(a)
    return [np.ascontiguousarray(a) for a in arrs]


def _to_engine(arrs: Sequence[np.ndarray], device: "E.Device",
               requires_grad: bool):
    out = []
    for a in arrs:
        # Match _to_torch — only float inputs request grad.
        rg = bool(requires_grad) and np.issubdtype(a.dtype, np.floating)
        out.append(E.TensorImpl(a, device, rg))
    return out


def _to_torch(arrs: Sequence[np.ndarray], requires_grad: bool):
    out = []
    for a in arrs:
        # requires_grad only valid on float/complex; int / bool inputs (indices,
        # masks) silently fall back to non-grad.
        rg = bool(requires_grad) and np.issubdtype(a.dtype,
                                                    np.floating)
        out.append(torch.tensor(a, requires_grad=rg))
    return out


# --------------------------------------------------------------------------- #
# Conversions / scalar reduction
# --------------------------------------------------------------------------- #

def _engine_to_numpy(t) -> np.ndarray:
    arr = t.data_as_python()
    a = np.asarray(arr)
    # Engine returns flattened bytes; reshape to declared shape.
    if list(a.shape) != list(t.shape):
        a = a.reshape(tuple(t.shape) if list(t.shape) else ())
    return a


def _engine_sum_to_scalar(t):
    """Sum every element into a 0-d scalar tensor for backward seeding."""
    axes = list(range(len(t.shape)))
    if not axes:
        return t  # already scalar
    return E.sum(t, axes, False)


def _torch_sum_to_scalar(t: torch.Tensor) -> torch.Tensor:
    return t.sum() if t.dim() > 0 else t


# --------------------------------------------------------------------------- #
# Axis runners
# --------------------------------------------------------------------------- #

def _allclose(actual: np.ndarray, expected: np.ndarray,
              atol: float, rtol: float, label: str) -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{label}: shape mismatch — got {actual.shape}, "
            f"expected {expected.shape}"
        )
    if not np.allclose(actual, expected, atol=atol, rtol=rtol,
                       equal_nan=True):
        diff = np.abs(actual - expected)
        i = int(np.argmax(diff))
        raise AssertionError(
            f"{label}: max abs diff {float(diff.max()):.3e} "
            f"(atol={atol}, rtol={rtol}); first off-element flat[{i}]="
            f"{actual.flatten()[i]} vs {expected.flatten()[i]}"
        )


def _run_engine_forward(spec: OpSpec, device, requires_grad: bool):
    arrs = _make_numpy_inputs(spec)
    inputs = _to_engine(arrs, device, requires_grad)
    out = spec.engine_fn(inputs, **spec.kwargs)
    return arrs, inputs, out


def _run_torch_forward(spec: OpSpec, requires_grad: bool):
    arrs = _make_numpy_inputs(spec)
    tinputs = _to_torch(arrs, requires_grad)
    tout = spec.torch_fn(tinputs, **spec.kwargs)
    return tinputs, tout


def check_forward(spec: OpSpec, device) -> None:
    _, _, out = _run_engine_forward(spec, device, requires_grad=False)
    _, tout = _run_torch_forward(spec, requires_grad=False)
    expected = tout.detach().cpu().numpy()
    label = f"[{spec.name} | fwd | {device.name if hasattr(device,'name') else device}]"
    _allclose(_engine_to_numpy(out), expected, spec.atol, spec.rtol, label)


def check_cross_device(spec: OpSpec) -> None:
    _, _, out_cpu = _run_engine_forward(spec, E.Device.CPU, requires_grad=False)
    _, _, out_gpu = _run_engine_forward(spec, E.Device.GPU, requires_grad=False)
    label = f"[{spec.name} | fwd | CPU vs GPU]"
    _allclose(_engine_to_numpy(out_cpu), _engine_to_numpy(out_gpu),
              spec.atol, spec.rtol, label)


def _engine_backward(spec, out, inputs):
    if spec.post_fn is not None:
        scalar = spec.post_fn(out, engine=E)
    else:
        scalar = _engine_sum_to_scalar(out)
    E.engine_backward(scalar, False)
    grads = []
    for t in inputs:
        if not t.requires_grad:
            grads.append(None)
            continue
        g = t.grad_as_python()
        if g is None:
            grads.append(None)
        else:
            arr = np.asarray(g)
            if list(arr.shape) != list(t.shape):
                arr = arr.reshape(tuple(t.shape) if list(t.shape) else ())
            grads.append(arr)
    return grads


def _torch_backward(spec, tout, tinputs):
    if spec.post_fn is not None:
        scalar = spec.post_fn(tout, engine=torch)
    else:
        scalar = _torch_sum_to_scalar(tout)
    scalar.backward()
    return [
        (t.grad.detach().cpu().numpy() if t.grad is not None else None)
        for t in tinputs
    ]


def check_backward(spec: OpSpec, device) -> None:
    _, inputs, out = _run_engine_forward(spec, device, requires_grad=True)
    e_grads = _engine_backward(spec, out, inputs)
    tinputs, tout = _run_torch_forward(spec, requires_grad=True)
    t_grads = _torch_backward(spec, tout, tinputs)

    devname = device.name if hasattr(device, "name") else str(device)
    for i, (eg, tg) in enumerate(zip(e_grads, t_grads)):
        label = f"[{spec.name} | bwd | {devname} | input {i}]"
        if tg is None and eg is None:
            continue
        if tg is None or eg is None:
            raise AssertionError(
                f"{label}: grad presence mismatch (engine={eg is not None}, "
                f"torch={tg is not None})"
            )
        _allclose(eg, tg, spec.atol, spec.rtol, label)


def check_cross_device_backward(spec: OpSpec) -> None:
    _, inputs_c, out_c = _run_engine_forward(spec, E.Device.CPU, requires_grad=True)
    g_cpu = _engine_backward(spec, out_c, inputs_c)
    _, inputs_g, out_g = _run_engine_forward(spec, E.Device.GPU, requires_grad=True)
    g_gpu = _engine_backward(spec, out_g, inputs_g)
    for i, (gc, gg) in enumerate(zip(g_cpu, g_gpu)):
        label = f"[{spec.name} | bwd | CPU vs GPU | input {i}]"
        if gc is None and gg is None:
            continue
        if gc is None or gg is None:
            raise AssertionError(
                f"{label}: grad presence mismatch (cpu={gc is not None}, "
                f"gpu={gg is not None})"
            )
        _allclose(gc, gg, spec.atol, spec.rtol, label)
