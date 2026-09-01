"""``lucid.coreml`` — writing Core ML packages, running them, proving it.

Three properties, each with its own failure mode:

* **the bytes are right.**  Lucid writes the ``.mlpackage`` itself — MIL
  protobuf, weight blob, bundle — with no protobuf library and no
  coremltools.  A wrong field number or blob offset yields a file that
  still parses and still loads; only comparing against the reference
  reader, and against the eager model's numbers, catches it.
* **the numbers are right.**  A package missing a layer has the correct
  output shape and returns plausible values.  Only a value comparison
  finds that, which is why every model test verifies rather than asserting
  a shape.
* **the accelerator is actually used.**  Asking for the Neural Engine and
  not getting it is silent: a float32 program requested with
  ``CPU_AND_NE`` reports zero ANE operations, runs at CPU speed, and warns
  about nothing.  ``compute_plan`` turns that into an assertion.
"""

import pytest

import lucid
import lucid.coreml as cml
import lucid.models as M
import lucid.nn as nn
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"),
    reason="the engine was built without the Core ML writer",
)


class _Tiny(nn.Module):
    """One of each op class the classification tier covers."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 4 * 4, num_classes)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        h = nn.functional.relu(self.bn(self.conv(x)))
        return self.fc(self.pool(h).reshape(x.shape[0], -1))


def _tiny() -> tuple[nn.Module, lucid.Tensor]:
    return _Tiny().eval(), lucid.randn(1, 3, 8, 8)


class TestPackageFormat:
    """The engine writes the bundle; these check its shape on disk."""

    def test_a_package_has_the_three_parts(self, tmp_path: object) -> None:
        import os

        model, x = _tiny()
        cm = cml.export(model, x, str(tmp_path / "t.mlpackage"))

        root = cm.path
        assert os.path.exists(os.path.join(root, "Manifest.json"))
        assert os.path.exists(os.path.join(root, "Data/com.apple.CoreML/model.mlmodel"))
        assert os.path.exists(
            os.path.join(root, "Data/com.apple.CoreML/weights/weight.bin")
        )
        cm.close()

    def test_a_hand_built_program_loads_and_runs(self, tmp_path: object) -> None:
        # The smallest possible exercise of writer + runtime, with no
        # tracer involved: if this fails the format is wrong, not the
        # translation.
        engine = _C_engine.coreml
        f32 = engine.DTYPE_FLOAT32
        program = engine.MilProgram("x", (f32, [1, 4]))
        program.add_op("relu", [("x", ["x"])], "y", (f32, [1, 4]))
        program.set_output("y", (f32, [1, 4]))

        paths = engine.prepare_package(str(tmp_path / "relu.mlpackage"))
        blob = engine.BlobWriter(paths.weight_bin)
        blob.finalize()
        engine.finish_package(paths, program.serialize())

        handle = engine.load_model(paths.root, engine.ComputeUnits.CPU_ONLY)
        out = lucid.Tensor(
            handle.predict("x", lucid.tensor([[-1.0, 2.0, -3.0, 4.0]])._impl, "y")
        )

        assert out.numpy().ravel().tolist() == [0.0, 2.0, 0.0, 4.0]
        handle.close()


class TestAgreementWithEager:
    def test_a_tiny_model_matches(self, tmp_path: object) -> None:
        model, x = _tiny()

        cm = cml.export(model, x, str(tmp_path / "tiny.mlpackage"))

        assert cm.verify(model, x) < 1e-5
        cm.close()

    @pytest.mark.parametrize(
        "factory",
        ["resnet_18_cls", "mobilenet_v2_cls", "densenet_121_cls", "convnext_tiny_cls"],
    )
    def test_zoo_classifiers_match(self, factory: str, tmp_path: object) -> None:
        # Unmodified factories, output dataclass and all.  Compared
        # relative to the output's own scale: a 64x64 input through a
        # randomly initialised head can produce logits near 1e-12, where
        # an absolute bound says nothing.
        model = M.create_model(factory, num_classes=10).eval()
        x = lucid.randn(1, 3, 64, 64)
        scale = max(float(model(x).logits.abs().max().item()), 1e-6)

        cm = cml.export(model, x, str(tmp_path / f"{factory}.mlpackage"))

        assert cm.verify(model, x) / scale < 1e-4
        cm.close()

    def test_float16_costs_precision_and_says_so(self, tmp_path: object) -> None:
        model, x = _tiny()

        half = cml.export(
            model, x, str(tmp_path / "half.mlpackage"), precision=cml.Precision.FLOAT16
        )

        assert half.precision == "FLOAT16"
        assert half.verify(model, x) < 1e-2
        half.close()


class TestTheAcceleratorIsActuallyUsed:
    """The claim this package exists for, asserted rather than timed."""

    def test_float32_reaches_no_neural_engine(self, tmp_path: object) -> None:
        # The silent failure: asking for the ANE with a float32 program
        # succeeds, returns correct numbers, and uses none of it.
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        x = lucid.randn(1, 3, 64, 64)

        cm = cml.export(
            model,
            x,
            str(tmp_path / "f32.mlpackage"),
            precision=cml.Precision.FLOAT32,
            compute_units=cml.ComputeUnits.CPU_AND_NE,
        )

        plan = cm.compute_plan()
        if plan.total_compute:  # macOS < 14.4 reports nothing
            assert plan.ane_fraction == 0.0
        cm.close()

    def test_float16_runs_the_computation_on_the_neural_engine(
        self, tmp_path: object
    ) -> None:
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        x = lucid.randn(1, 3, 64, 64)

        cm = cml.export(
            model,
            x,
            str(tmp_path / "f16.mlpackage"),
            precision=cml.Precision.FLOAT16,
            compute_units=cml.ComputeUnits.CPU_AND_NE,
        )

        plan = cm.compute_plan()
        if plan.total_compute:
            # The only operations left on the CPU are the two casts that
            # keep the interface float32.
            assert plan.ane_fraction > 0.9
            assert plan.compute.get("CPU", 0) <= 2
        cm.close()

    def test_constants_are_not_counted_as_computation(self, tmp_path: object) -> None:
        # ResNet-18 emits ~250 const operations against ~70 real ones;
        # counting them would report ~20% ANE for a fully accelerated model.
        model, x = _tiny()
        cm = cml.export(model, x, str(tmp_path / "counts.mlpackage"))

        plan = cm.compute_plan()
        if plan.total_compute:
            assert plan.constants > plan.total_compute
        cm.close()


class TestRoundTrip:
    def test_a_written_package_can_be_loaded_back(self, tmp_path: object) -> None:
        model, x = _tiny()
        path = str(tmp_path / "again.mlpackage")
        first = cml.export(model, x, path)
        want = first.predict(x)
        first.close()

        again = cml.load(path)

        assert float((again.predict(x) - want).abs().max().item()) == 0.0
        again.close()


class TestRefusals:
    """Each of these would otherwise be a quietly wrong model."""

    def test_an_unmapped_op_names_itself(self, tmp_path: object) -> None:
        class UsesAnUnmappedOp(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return lucid.sin(x)

        with pytest.raises(cml.UnsupportedOp) as excinfo:
            cml.export(
                UsesAnUnmappedOp().eval(),
                lucid.randn(1, 4),
                str(tmp_path / "no.mlpackage"),
            )

        assert excinfo.value.op_name == "sin"

    def test_a_training_mode_model_is_refused(self, tmp_path: object) -> None:
        with pytest.raises(ValueError, match="training mode"):
            cml.export(
                _Tiny().train(), lucid.randn(1, 3, 8, 8), str(tmp_path / "t.mlpackage")
            )

    def test_float64_has_no_core_ml_equivalent(self, tmp_path: object) -> None:
        class Passthrough(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return x * 2.0

        with pytest.raises(TypeError, match="float64"):
            cml.export(
                Passthrough().eval(),
                lucid.ones(2, dtype=lucid.float64),
                str(tmp_path / "d.mlpackage"),
            )

    def test_a_metal_input_is_not_silently_downloaded(self, tmp_path: object) -> None:
        if not lucid.metal.is_available():
            pytest.skip("needs Metal")
        model, x = _tiny()
        cm = cml.export(model, x, str(tmp_path / "dev.mlpackage"))

        with pytest.raises(ValueError, match="CPU tensor"):
            cm.predict(x.to("metal"))
        cm.close()
