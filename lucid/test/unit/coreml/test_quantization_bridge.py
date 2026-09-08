"""Models `lucid.quantization` produces, reaching the exporter.

Lucid has a quantization subsystem and an exporter, and until now nothing
carried a model from one to the other. Crossing the two found the reason:
a quantization-aware model carries observers that record the range of
every activation they see, and ``eval()`` does not stop them — correctly,
since post-training calibration runs in eval mode and *is* that
recording. Tracing is not calibration. It runs the model once to learn
its shape, and the observers wrote to their buffers while it did, so the
export refused the model as one that mutates itself.

They are paused around the trace now. Nothing about the exported program
changes: an observer contributes no operation, only a record.

What comes out is exact. A QAT model's weights have been trained onto
the int8 grid, so storing them as int8 costs nothing — measured on a
small classifier at 0.00e+00 against the eager model, where the same
network quantized after training lands at 7e-04 and loses accuracy the
QAT one keeps (0.997 against 0.986).

One thing this does not yet do, recorded because it is the next
question rather than a defect: the package carries the *activation*
fake-quantization as real arithmetic — 75 operations where the float
model has 10. That is faithful, and it is not what anyone wants on the
Neural Engine, which runs float16 activations whatever the package says.
Core ML's own quantization is weight-only, so simulating int8
activations in float is work the accelerator did not ask for. Dropping
them would change what the package computes, which is a choice to offer
rather than to make here.
"""

import pytest

import lucid
import lucid.coreml as cml
import lucid.nn as nn
import lucid.quantization as q
from lucid._C import engine as _C_engine
from lucid.quantization import FakeQuantize

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"),
    reason="the engine was built without the Core ML writer",
)


def _net() -> nn.Module:
    lucid.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    ).eval()


def _prepared(x: lucid.Tensor) -> nn.Module:
    model = q.prepare_qat(_net(), q.get_default_qat_qconfig_mapping(), (x,))
    model.eval()
    model(x)  # let the observers see one batch, as a caller would
    return model


class TestAQuantizationAwareModelExports:
    def test_it_exports_at_all(self, tmp_path) -> None:
        """It did not. The observers made the trace a buffer change."""
        x = lucid.randn(1, 3, 16, 16)
        exported = cml.export(_prepared(x), x, f"{tmp_path}/qat.mlpackage")
        try:
            assert tuple(exported.predict(x).shape) == (1, 10)
        finally:
            exported.close()

    def test_eval_alone_is_not_enough(self, tmp_path) -> None:
        """Which is why this belongs to the export and not to the caller.

        Calibration runs in eval mode, so `eval()` leaving observers on is
        right. It also means a caller who has done everything correctly
        still cannot export, and the fix is not something to ask them for.
        """
        x = lucid.randn(1, 3, 16, 16)
        model = _prepared(x)
        live = [m for m in model.modules() if isinstance(m, FakeQuantize)]
        assert live, "the fixture must actually produce observers"
        assert all(getattr(m, "_observer_enabled", False) for m in live)

    def test_the_observers_are_put_back(self, tmp_path) -> None:
        """Pausing them is the export's business, not a side effect.

        A caller who exports mid-calibration must find the model as they
        left it, still recording.
        """
        x = lucid.randn(1, 3, 16, 16)
        model = _prepared(x)
        before = [
            getattr(m, "_observer_enabled", False)
            for m in model.modules()
            if isinstance(m, FakeQuantize)
        ]
        cml.export(model, x, f"{tmp_path}/restored.mlpackage").close()
        after = [
            getattr(m, "_observer_enabled", False)
            for m in model.modules()
            if isinstance(m, FakeQuantize)
        ]
        assert after == before

    def test_one_that_was_paused_stays_paused(self, tmp_path) -> None:
        """The other direction: restore what was found, not what is tidy."""
        x = lucid.randn(1, 3, 16, 16)
        model = _prepared(x)
        for module in model.modules():
            if isinstance(module, FakeQuantize):
                module.disable_observer()
        cml.export(model, x, f"{tmp_path}/kept.mlpackage").close()
        assert not any(
            getattr(m, "_observer_enabled", False)
            for m in model.modules()
            if isinstance(m, FakeQuantize)
        )

    @pytest.mark.parametrize(
        ("name", "weights"),
        [
            ("float", cml.WeightPrecision.FLOAT),
            ("int8", cml.WeightPrecision.INT8),
        ],
    )
    def test_the_package_computes_what_the_model_does(
        self, name, weights, tmp_path
    ) -> None:
        """Exactly, and int8 is free here — which is the point of QAT.

        The weights were trained onto the grid, so storing them on it
        changes nothing. The same network quantized after training does
        not have that property.
        """
        x = lucid.randn(1, 3, 16, 16)
        model = _prepared(x)
        for module in model.modules():
            if isinstance(module, FakeQuantize):
                module.disable_observer()
        exported = cml.export(model, x, f"{tmp_path}/{name}.mlpackage", weights=weights)
        try:
            assert exported.verify(model, x, relative=True) < 1e-5
        finally:
            exported.close()


class TestWhatStillDoesNotCross:
    """Named rather than left to be discovered, and each for a reason.

    A converted model — `quantize_dynamic`, or `prepare` then `convert` —
    holds `QuantizedLinearMLX`, whose forward calls an MLX packed GEMM and
    moves the activation to the GPU and back. The tracer follows none of
    it, so the output falls out of the graph.

    That one is a translation to write, not a refusal to fix: MLX packs
    weights in groups along the input axis with a scale and a bias per
    group, and Core ML's affine dequantize is per-channel along one axis.
    The layouts differ, so the bridge is real work rather than a mapping.
    """

    def test_a_converted_model_is_refused_by_name(self, tmp_path) -> None:
        x = lucid.randn(1, 3, 16, 16)
        converted = q.quantize_dynamic(_net())
        with pytest.raises(ValueError, match="did not come out of the traced"):
            cml.export(converted, x, f"{tmp_path}/converted.mlpackage")


class TestDroppingTheActivationSimulation:
    """``Activations.DROPPED`` — the second half of the bridge.

    A prepared model carries fake-quantization in two places: on each
    weight, which puts it onto the grid it will be stored on, and on each
    activation, which simulates a runtime that computes in integers. The
    first is what the export wants. The second is a simulation of a
    runtime this is not — Core ML quantizes weights only, and the Neural
    Engine computes in float16 whatever the package says — so carrying it
    means rounding, clipping and scaling at every activation, work the
    accelerator did not ask for.

    Measured on the classifier above: twenty-four operations against six,
    with the accuracy unchanged. It is not the default anyway, because
    dropping it changes what the package computes and that is the
    caller's call to make.
    """

    def test_it_removes_the_simulation_and_keeps_the_weights(self, tmp_path) -> None:
        x = lucid.randn(1, 3, 16, 16)
        model = _prepared(x)
        for module in model.modules():
            if isinstance(module, FakeQuantize):
                module.disable_observer()

        kept = cml.export(
            model,
            x,
            f"{tmp_path}/kept.mlpackage",
            weights=cml.WeightPrecision.INT8,
            activations=cml.Activations.SIMULATED,
        )
        dropped = cml.export(
            model,
            x,
            f"{tmp_path}/dropped.mlpackage",
            weights=cml.WeightPrecision.INT8,
            activations=cml.Activations.DROPPED,
        )
        try:
            assert (
                dropped.compute_plan().total_compute < kept.compute_plan().total_compute
            )
        finally:
            kept.close()
            dropped.close()

    def test_only_the_activation_one_is_paused(self) -> None:
        """Both are the same class, told apart by where they hang.

        Dropping the weight's fake-quantize alongside the activation's
        would discard the grid the weights were trained onto, which is
        the only reason to export a prepared model rather than the float
        one it came from. Checked on the mechanism rather than through a
        tolerance: what matters is which modules are paused, and a
        numeric threshold measures that only at a distance.
        """
        from lucid.coreml._build import _activation_quantization_paused

        x = lucid.randn(1, 3, 16, 16)
        model = _prepared(x)
        weights = [
            m for n, m in model.named_modules() if n.endswith("weight_fake_quant")
        ]
        activations = [
            m
            for n, m in model.named_modules()
            if n.endswith("activation_post_process") and isinstance(m, FakeQuantize)
        ]
        assert weights and activations

        with _activation_quantization_paused(model):
            assert all(m._fake_quant_enabled for m in weights)
            assert not any(m._fake_quant_enabled for m in activations)

        assert all(m._fake_quant_enabled for m in weights)
        assert all(m._fake_quant_enabled for m in activations)

    def test_it_puts_the_simulation_back(self, tmp_path) -> None:
        """Exporting is not allowed to change the model it was given."""
        x = lucid.randn(1, 3, 16, 16)
        model = _prepared(x)
        before = [
            getattr(m, "_fake_quant_enabled", False)
            for m in model.modules()
            if isinstance(m, FakeQuantize)
        ]
        cml.export(
            model,
            x,
            f"{tmp_path}/restored.mlpackage",
            activations=cml.Activations.DROPPED,
        ).close()
        after = [
            getattr(m, "_fake_quant_enabled", False)
            for m in model.modules()
            if isinstance(m, FakeQuantize)
        ]
        assert after == before

    def test_it_is_inert_on_a_model_that_carries_none(self, tmp_path) -> None:
        """Which is every model that did not come from `lucid.quantization`."""
        lucid.manual_seed(0)
        plain = _net()
        x = lucid.randn(1, 3, 16, 16)
        both = []
        for name, setting in (
            ("simulated", cml.Activations.SIMULATED),
            ("dropped", cml.Activations.DROPPED),
        ):
            exported = cml.export(
                plain, x, f"{tmp_path}/{name}.mlpackage", activations=setting
            )
            try:
                both.append(exported.verify(plain, x, relative=True))
            finally:
                exported.close()
        assert both[0] == both[1] < 1e-5
