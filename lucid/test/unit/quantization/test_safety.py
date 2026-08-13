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


class TestQuantizedConvKeepsThePaddingSpec:
    """``from_float`` used to read the wrong attribute and drop the spec.

    ``Conv*d.__init__`` moves a string padding into ``_padding_str`` and
    leaves ``padding`` at ``0``, so the guard that tested
    ``isinstance(f.padding, str)`` never fired — and ``padding_mode`` was
    not looked at at all.  Both produced a quantized module that does not
    compute what the float module it came from computes.
    """

    @staticmethod
    def _convert(layer: nn.Module, x: lucid.Tensor):
        model = nn.Sequential(layer)
        float_out = model(x)
        model.qconfig = Q.get_default_qconfig()
        Q.prepare(model, inplace=True)
        model(x)
        Q.convert(model, inplace=True)
        return float_out, model(x)

    def test_same_padding_survives_quantization(self) -> None:
        """This one at least changed the shape — 8x8 became 6x6."""
        x = lucid.randn(1, 2, 8, 8)
        f, q = self._convert(nn.Conv2d(2, 4, 3, padding="same"), x)
        assert tuple(f.shape) == tuple(q.shape) == (1, 4, 8, 8)

    def test_valid_padding_survives_quantization(self) -> None:
        x = lucid.randn(1, 2, 8, 8)
        f, q = self._convert(nn.Conv2d(2, 4, 3, padding="valid"), x)
        assert tuple(f.shape) == tuple(q.shape) == (1, 4, 6, 6)

    def test_padding_mode_survives_quantization(self) -> None:
        """The quieter one: no shape change, just a different border.

        Comparing against the *float it was made from* is what catches it;
        a shape check never would.
        """
        x = lucid.randn(1, 2, 8, 8)
        f, q = self._convert(nn.Conv2d(2, 4, 3, padding=1, padding_mode="reflect"), x)
        err = float((f - q).abs().max().item())
        # Quantization error alone is ~0.01; dropping the mode gave ~1.1.
        assert err < 0.2, err

    def test_plain_integer_padding_is_unaffected(self) -> None:
        x = lucid.randn(1, 2, 8, 8)
        f, q = self._convert(nn.Conv2d(2, 4, 3, padding=1), x)
        assert tuple(f.shape) == tuple(q.shape)
        assert float((f - q).abs().max().item()) < 0.2

    @pytest.mark.parametrize("rank", [1, 3])
    def test_other_ranks_keep_same_padding_too(self, rank: int) -> None:
        cls = {1: nn.Conv1d, 3: nn.Conv3d}[rank]
        x = lucid.randn(1, 2, *([8] * rank))
        f, q = self._convert(cls(2, 4, 3, padding="same"), x)
        assert tuple(f.shape) == tuple(q.shape)
        assert tuple(f.shape)[2:] == tuple([8] * rank)
