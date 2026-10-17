"""Per-device AMP autocast state and float32 reduction dtype preservation (LCD-70, LCD-120).

Verifies that:
- ``autocast(device_type="cpu")`` only affects CPU ops and leaves Metal ops in their
  input dtype,
- ``autocast(device_type="metal")`` only affects Metal ops and leaves CPU ops in their
  input dtype,
- nested CPU and Metal autocast scopes maintain independent target dtypes and
  ``enabled=False`` overrides on each device without disturbing the other device,
- ``var``, ``std``, and ``logsumexp`` of ``float32`` inputs stay ``float32`` under
  autocast on both CPU and Metal (LCD-120).
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.amp import autocast
from lucid.test._fixtures.devices import metal_available

OFF = (False, None)


def _state(device_type: str | None = None) -> tuple[bool, lucid.dtype | None]:
    return (
        autocast.is_autocast_enabled(device_type),
        autocast.get_autocast_dtype(device_type),
    )


@pytest.fixture(autouse=True)
def _amp_off_around_each_test() -> object:
    assert _state() == OFF
    assert _state("cpu") == OFF
    assert _state("metal") == OFF
    yield
    assert _state() == OFF
    assert _state("cpu") == OFF
    assert _state("metal") == OFF


def _run_promote_ops(device: str) -> dict[str, lucid.dtype]:
    x = lucid.randn(2, 4, device=device, dtype=lucid.float32)
    w = lucid.randn(4, 4, device=device, dtype=lucid.float32)
    b = lucid.randn(4, device=device, dtype=lucid.float32)

    img = lucid.randn(2, 3, 6, 6, device=device, dtype=lucid.float32)
    ker = lucid.randn(4, 3, 3, 3, device=device, dtype=lucid.float32)
    cb = lucid.randn(4, device=device, dtype=lucid.float32)

    return {
        "matmul": (x @ w).dtype,
        "linear": F.linear(x, w, b).dtype,
        "conv2d": F.conv2d(img, ker, cb).dtype,
        "add": (x + w[:2]).dtype,
    }


def _run_batch_norm_bf16(device: str) -> tuple[lucid.dtype, lucid.dtype, lucid.dtype]:
    """Run training ``batch_norm`` with bfloat16 activations and float32 running stats.

    Under autocast on ``device``, ``AmpPolicy::ForceFP32`` promotes the output
    to ``float32`` while keeping ``running_mean`` / ``running_var`` in ``float32``.
    When autocast is inactive on ``device``, the output stays ``bfloat16``.
    """
    img = lucid.randn(2, 3, 6, 6, device=device, dtype=lucid.bfloat16)
    bn_mean = lucid.zeros(3, device=device, dtype=lucid.float32)
    bn_var = lucid.ones(3, device=device, dtype=lucid.float32)
    bn_w = lucid.ones(3, device=device, dtype=lucid.bfloat16)
    bn_b = lucid.zeros(3, device=device, dtype=lucid.bfloat16)
    out = F.batch_norm(img, bn_mean, bn_var, bn_w, bn_b, training=True)
    return out.dtype, bn_mean.dtype, bn_var.dtype


@pytest.mark.skipif(not metal_available(), reason="Metal not available")
def test_cpu_autocast_does_not_cast_metal_ops() -> None:
    with autocast(device_type="cpu", dtype=lucid.bfloat16):
        assert _state("cpu") == (True, lucid.bfloat16)
        assert _state("metal") == OFF
        metal_dtypes = _run_promote_ops("metal")
        cpu_dtypes = _run_promote_ops("cpu")
        metal_bn, metal_rm, metal_rv = _run_batch_norm_bf16("metal")
        cpu_bn, cpu_rm, cpu_rv = _run_batch_norm_bf16("cpu")

    for op_name, dt in metal_dtypes.items():
        assert dt == lucid.float32, f"Metal {op_name} was cast under CPU autocast: {dt}"
    for op_name, dt in cpu_dtypes.items():
        assert (
            dt == lucid.bfloat16
        ), f"CPU {op_name} was not cast under CPU autocast: {dt}"
    # Metal batch_norm was not affected by CPU autocast (stays bfloat16);
    # CPU batch_norm was ForceFP32-promoted to float32.
    assert metal_bn == lucid.bfloat16
    assert cpu_bn == lucid.float32
    assert (metal_rm, metal_rv, cpu_rm, cpu_rv) == (
        lucid.float32,
        lucid.float32,
        lucid.float32,
        lucid.float32,
    )


def test_metal_autocast_does_not_cast_cpu_ops() -> None:
    with autocast(device_type="metal", dtype=lucid.bfloat16):
        assert _state("metal") == (True, lucid.bfloat16)
        assert _state("cpu") == OFF
        cpu_dtypes = _run_promote_ops("cpu")
        cpu_bn, cpu_rm, cpu_rv = _run_batch_norm_bf16("cpu")
        if metal_available():
            metal_dtypes = _run_promote_ops("metal")
            metal_bn, metal_rm, metal_rv = _run_batch_norm_bf16("metal")
        else:
            metal_dtypes = {}
            metal_bn = metal_rm = metal_rv = lucid.float32

    for op_name, dt in cpu_dtypes.items():
        assert dt == lucid.float32, f"CPU {op_name} was cast under Metal autocast: {dt}"
    assert cpu_bn == lucid.bfloat16
    assert (cpu_rm, cpu_rv) == (lucid.float32, lucid.float32)
    for op_name, dt in metal_dtypes.items():
        assert (
            dt == lucid.bfloat16
        ), f"Metal {op_name} was not cast under Metal autocast: {dt}"
    if metal_available():
        assert metal_bn == lucid.float32
        assert (metal_rm, metal_rv) == (lucid.float32, lucid.float32)


@pytest.mark.skipif(not metal_available(), reason="Metal not available")
def test_nested_cpu_and_metal_autocast_independent_dtypes() -> None:
    with (
        autocast("cpu", dtype=lucid.bfloat16),
        autocast("metal", dtype=lucid.float16),
    ):
        assert _state("cpu") == (True, lucid.bfloat16)
        assert _state("metal") == (True, lucid.float16)
        cpu_dtypes = _run_promote_ops("cpu")
        metal_dtypes = _run_promote_ops("metal")
        cpu_bn, cpu_rm, cpu_rv = _run_batch_norm_bf16("cpu")
        metal_bn, metal_rm, metal_rv = _run_batch_norm_bf16("metal")

    for op_name, dt in cpu_dtypes.items():
        assert dt == lucid.bfloat16, f"CPU {op_name} expected bfloat16, got {dt}"
    for op_name, dt in metal_dtypes.items():
        assert dt == lucid.float16, f"Metal {op_name} expected float16, got {dt}"
    assert cpu_bn == lucid.float32 and metal_bn == lucid.float32
    assert (cpu_rm, cpu_rv, metal_rm, metal_rv) == (
        lucid.float32,
        lucid.float32,
        lucid.float32,
        lucid.float32,
    )


@pytest.mark.skipif(not metal_available(), reason="Metal not available")
def test_inner_disabled_scope_only_disables_target_device() -> None:
    x_cpu = lucid.randn(4, 4, device="cpu", dtype=lucid.float32)
    x_gpu = lucid.randn(4, 4, device="metal", dtype=lucid.float32)

    with (
        autocast("cpu", dtype=lucid.bfloat16),
        autocast("metal", dtype=lucid.float16),
    ):
        assert (x_cpu @ x_cpu).dtype == lucid.bfloat16
        assert (x_gpu @ x_gpu).dtype == lucid.float16

        # Disable CPU autocast inside while keeping Metal autocast active.
        with autocast("cpu", enabled=False):
            assert _state("cpu") == OFF
            assert _state("metal") == (True, lucid.float16)
            assert (x_cpu @ x_cpu).dtype == lucid.float32
            assert (x_gpu @ x_gpu).dtype == lucid.float16

        # CPU autocast restored after inner scope exits.
        assert _state("cpu") == (True, lucid.bfloat16)
        assert (x_cpu @ x_cpu).dtype == lucid.bfloat16

        # Disable Metal autocast inside while keeping CPU autocast active.
        with autocast("metal", enabled=False):
            assert _state("metal") == OFF
            assert _state("cpu") == (True, lucid.bfloat16)
            assert (x_gpu @ x_gpu).dtype == lucid.float32
            assert (x_cpu @ x_cpu).dtype == lucid.bfloat16

        # Metal autocast restored after inner scope exits.
        assert _state("metal") == (True, lucid.float16)
        assert (x_gpu @ x_gpu).dtype == lucid.float16


@pytest.mark.parametrize(
    ("device", "amp_dtype"),
    [
        ("cpu", lucid.bfloat16),
        pytest.param(
            "metal",
            lucid.float16,
            marks=pytest.mark.skipif(
                not metal_available(), reason="Metal not available"
            ),
        ),
        pytest.param(
            "metal",
            lucid.bfloat16,
            marks=pytest.mark.skipif(
                not metal_available(), reason="Metal not available"
            ),
        ),
    ],
)
@pytest.mark.parametrize("requires_grad", [False, True])
def test_var_std_logsumexp_float32_stay_float32_under_autocast(
    ref: object,
    device: str,
    amp_dtype: lucid.dtype,
    requires_grad: bool,
) -> None:
    """LCD-120: ``var``, ``std``, and ``logsumexp`` of float32 inputs stay float32
    under autocast even though their composite steps use elementwise helpers."""
    vals = [[1.0, 2.5, -0.5, 4.0], [0.25, -1.5, 3.25, 2.0]]
    x = lucid.tensor(
        vals, dtype=lucid.float32, device=device, requires_grad=requires_grad
    )
    rx = ref.tensor(vals, dtype=ref.float32, requires_grad=requires_grad)  # type: ignore[attr-defined]

    with autocast(device_type=device, dtype=amp_dtype):
        v_unbiased = lucid.var(x, dim=-1, unbiased=True)
        v_biased = lucid.var(x, dim=-1, unbiased=False)
        s_unbiased = lucid.std(x, dim=-1, unbiased=True)
        s_biased = lucid.std(x, dim=-1, unbiased=False)
        lse = lucid.logsumexp(x, dim=-1)

    rv_unbiased = ref.var(rx, dim=-1, unbiased=True)  # type: ignore[attr-defined]
    rv_biased = ref.var(rx, dim=-1, unbiased=False)  # type: ignore[attr-defined]
    rs_unbiased = ref.std(rx, dim=-1, unbiased=True)  # type: ignore[attr-defined]
    rs_biased = ref.std(rx, dim=-1, unbiased=False)  # type: ignore[attr-defined]
    rlse = ref.logsumexp(rx, dim=-1)  # type: ignore[attr-defined]

    for name, got, want in [
        ("var(unbiased=True)", v_unbiased, rv_unbiased),
        ("var(unbiased=False)", v_biased, rv_biased),
        ("std(unbiased=True)", s_unbiased, rs_unbiased),
        ("std(unbiased=False)", s_biased, rs_biased),
        ("logsumexp", lse, rlse),
    ]:
        assert (
            got.dtype == lucid.float32
        ), f"{name} returned {got.dtype}, expected float32"
        np.testing.assert_allclose(
            got.numpy(), want.detach().numpy(), rtol=1e-5, atol=1e-5
        )

    if requires_grad:
        total = (v_unbiased + s_unbiased + lse).sum()
        total.backward()
        assert x.grad is not None and x.grad.dtype == lucid.float32
        rtotal = (rv_unbiased + rs_unbiased + rlse).sum()
        rtotal.backward()
        np.testing.assert_allclose(
            x.grad.numpy(), rx.grad.numpy(), rtol=1e-5, atol=1e-5
        )
