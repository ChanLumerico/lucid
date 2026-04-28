"""
Pytest config for parity harness.

Exposes ``spec`` fixture parameterized over every registered OpSpec.
Skips GPU axes when MLX/Metal isn't usable on the host.
"""

from __future__ import annotations

import pytest

from lucid._C import engine as E

from . import (
    specs_bfunc,
    specs_edge_cases,
    specs_einops,
    specs_gfunc,
    specs_inplace,
    specs_linalg,
    specs_nn,
    specs_random,
    specs_ufunc,
    specs_utils,
)
from ._specs import collect_specs

ALL_SPECS = collect_specs([
    specs_bfunc,
    specs_ufunc,
    specs_utils,
    specs_einops,
    specs_linalg,
    specs_nn,
    specs_gfunc,
    specs_inplace,
    specs_random,
    specs_edge_cases,
])


def _gpu_available() -> bool:
    try:
        import numpy as np
        t = E.TensorImpl(np.ones((2, 2), dtype="float32"), E.Device.GPU, False)
        _ = t.shape
        return True
    except Exception:
        return False


GPU_AVAILABLE = _gpu_available()


def pytest_collection_modifyitems(config, items):
    if GPU_AVAILABLE:
        return
    skip_gpu = pytest.mark.skip(reason="GPU not available on this host")
    for item in items:
        if "GPU" in item.name or "cross_device" in item.name:
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session", params=ALL_SPECS, ids=lambda s: s.name)
def spec(request):
    return request.param
