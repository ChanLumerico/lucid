"""``lucid.backends`` — surface + dispatch behaviour smoke."""

import pytest

import lucid


class TestBackendsSurface:
    def test_module_present(self) -> None:
        assert hasattr(lucid, "backends")


class TestMpsAccessor:
    def test_is_available_callable(self) -> None:
        # ``lucid.backends.mps.is_available()`` (or equivalent) should
        # exist and return a bool — engine surface varies, so we tolerate.
        if not hasattr(lucid.backends, "mps"):
            pytest.skip("backends.mps not exposed")
        if hasattr(lucid.backends.mps, "is_available"):
            v = lucid.backends.mps.is_available()
            assert isinstance(v, bool)


class TestQuantizedEngine:
    """``"auto"`` follows availability alone, not where the tensors live.

    The docstring used to say the MLX kernel ran only for tensors on Metal,
    while ``use_mlx`` ignored the device.  The code was the intended half:
    ``QuantizedLinearMLX`` moves a CPU activation to Metal and back, so a
    CPU model is meant to get the fast path too.
    """

    def test_auto_follows_availability(self) -> None:
        quantized = lucid.backends.quantized
        prev = quantized.engine
        try:
            quantized.engine = "auto"
            assert quantized.use_mlx() == quantized.available
        finally:
            quantized.engine = prev

    @pytest.mark.skipif(
        not lucid.backends.quantized.available,
        reason="engine lacks the MLX quantized ops",
    )
    def test_auto_routes_a_cpu_model_to_the_mlx_layer(self) -> None:
        import lucid.nn as nn
        import lucid.nn.quantized as nnq
        import lucid.quantization as Q

        quantized = lucid.backends.quantized
        prev = quantized.engine
        try:
            quantized.engine = "auto"
            model = nn.Sequential(nn.Linear(64, 32))
            model.eval()
            qmodel = Q.quantize_dynamic(model)
            assert isinstance(qmodel[0], nnq.QuantizedLinearMLX)
            # Device-transparent: a CPU input still answers on the CPU.
            assert not qmodel(lucid.randn(2, 64)).is_metal
        finally:
            quantized.engine = prev
