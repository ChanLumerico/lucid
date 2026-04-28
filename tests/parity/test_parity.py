"""
Parity entrypoint — fans out every OpSpec across 6 verification axes.

Run all:           pytest tests/parity/
Filter by op:      pytest tests/parity/ -k matmul
One axis only:     pytest tests/parity/ -k forward
"""

from __future__ import annotations

import pytest

from lucid._C import engine as E

from ._harness import (
    check_backward,
    check_cross_device,
    check_cross_device_backward,
    check_forward,
)


def test_forward_CPU(spec):
    check_forward(spec, E.Device.CPU)


def test_forward_GPU(spec):
    if spec.skip_gpu:
        pytest.skip("spec opted out of GPU")
    check_forward(spec, E.Device.GPU)


def test_cross_device_forward(spec):
    if spec.skip_gpu:
        pytest.skip("spec opted out of GPU")
    check_cross_device(spec)


def test_backward_CPU(spec):
    if spec.skip_grad:
        pytest.skip("spec is non-differentiable")
    check_backward(spec, E.Device.CPU)


def test_backward_GPU(spec):
    if spec.skip_grad or spec.skip_gpu:
        pytest.skip("spec opted out of GPU or non-differentiable")
    check_backward(spec, E.Device.GPU)


def test_cross_device_backward(spec):
    if spec.skip_grad or spec.skip_gpu:
        pytest.skip("spec opted out of GPU or non-differentiable")
    check_cross_device_backward(spec)
