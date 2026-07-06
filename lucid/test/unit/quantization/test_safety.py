"""``lucid.quantization`` Phase-2 — calibration safety + entry-point validation."""

import warnings

import pytest

import lucid
import lucid.backends as backends
import lucid.nn as nn
import lucid.quantization as Q


class TestCalibrationSafety:
    def test_uncalibrated_convert_warns(self) -> None:
        # Dequant path (reference engine) with no calibration → loud warning,
        # not a silent near-zero collapse.
        prev = backends.quantized.engine
        backends.quantized.engine = "reference"
        try:
            m = nn.Sequential(nn.Linear(8, 8))
            m.eval()
            prepared = Q.prepare(m, Q.get_default_qconfig_mapping())  # no calibration
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                Q.convert(prepared)
            assert any("never saw" in str(w.message) for w in rec)
        finally:
            backends.quantized.engine = prev

    def test_calibrated_convert_is_silent(self) -> None:
        prev = backends.quantized.engine
        backends.quantized.engine = "reference"
        try:
            lucid.manual_seed(0)
            m = nn.Sequential(nn.Linear(8, 8))
            m.eval()
            prepared = Q.prepare(m, Q.get_default_qconfig_mapping())
            for _ in range(5):
                prepared(lucid.randn(4, 8))
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                Q.convert(prepared)
            assert not any("never saw" in str(w.message) for w in rec)
        finally:
            backends.quantized.engine = prev


class TestEntryValidation:
    def test_prepare_zero_match_warns(self) -> None:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            Q.prepare(nn.Sequential(nn.ReLU()), Q.get_default_qconfig_mapping())
        assert any("matched no quantizable" in str(w.message) for w in rec)

    def test_quantize_dynamic_zero_match_warns(self) -> None:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            Q.quantize_dynamic(nn.Sequential(nn.ReLU()))
        assert any("no module of a targeted type" in str(w.message) for w in rec)


class TestQConfigGuards:
    def test_qat_linear_requires_qconfig(self) -> None:
        import lucid.nn.qat as nnqat

        with pytest.raises(ValueError, match="qconfig"):
            nnqat.Linear(8, 8)

    def test_qat_conv_requires_qconfig(self) -> None:
        import lucid.nn.qat as nnqat

        with pytest.raises(ValueError, match="qconfig"):
            nnqat.Conv2d(3, 8, 3)
