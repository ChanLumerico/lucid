"""One package for several input shapes.

A fixed-shape package needs one file per batch size and one per
resolution. Making it flexible means the program stops fixing the axes
that vary — and finding out *which* axes those are is done by tracing the
model at each shape and comparing, not by propagating symbols. The tracer
already knows every value's shape; asking it twice cannot disagree with
itself about how an operation behaves.

The interesting case is the one that cannot be made flexible, and it is
common: an adaptive pool records a kernel derived from the input size, so
the same model traced at two resolutions is two different models. That is
refused by name rather than fixed to one shape and wrong at the others.
"""

import pytest

import lucid
import lucid.coreml as cml
import lucid.models as M
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"),
    reason="the engine was built without the Core ML writer",
)


class TestFlexibleBatch:
    def test_every_enumerated_batch_size_runs_and_agrees(
        self, tmp_path: object
    ) -> None:
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        sizes = (1, 2, 4)
        exported = cml.export(
            model,
            lucid.randn(1, 3, 224, 224),
            str(tmp_path / "batch.mlpackage"),
            shapes=[(n, 3, 224, 224) for n in sizes],
        )
        try:
            for n in sizes:
                x = lucid.randn(n, 3, 224, 224)
                reference = model(x).logits
                got = exported.predict(x)
                assert got.shape == reference.shape
                scale = float(reference.abs().max().item())
                assert float((got - reference).abs().max().item()) / scale < 1e-5
        finally:
            exported.close()

    def test_a_shape_that_was_not_enumerated_is_refused(
        self, tmp_path: object
    ) -> None:
        """Enumerated means enumerated; Core ML does not interpolate."""
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        exported = cml.export(
            model,
            lucid.randn(1, 3, 224, 224),
            str(tmp_path / "gaps.mlpackage"),
            shapes=[(1, 3, 224, 224), (4, 3, 224, 224)],
        )
        try:
            with pytest.raises(RuntimeError, match="not in"):
                exported.predict(lucid.randn(3, 3, 224, 224))
        finally:
            exported.close()

    def test_flexibility_does_not_cost_the_neural_engine(
        self, tmp_path: object
    ) -> None:
        """Worth asserting: a flexible package that fell back to the CPU
        would still be correct, and would quietly lose the reason this
        subsystem exists."""
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        x = lucid.randn(1, 3, 224, 224)
        fixed = cml.export(
            model,
            x,
            str(tmp_path / "fixed.mlpackage"),
            precision=cml.Precision.FLOAT16,
            compute_units=cml.ComputeUnits.CPU_AND_NE,
        )
        flexible = cml.export(
            model,
            x,
            str(tmp_path / "flex.mlpackage"),
            precision=cml.Precision.FLOAT16,
            shapes=[(1, 3, 224, 224), (2, 3, 224, 224)],
            compute_units=cml.ComputeUnits.CPU_AND_NE,
        )
        try:
            one, many = fixed.compute_plan(), flexible.compute_plan()
            if one.total_compute and many.total_compute:
                assert many.ane_fraction >= one.ane_fraction - 0.05
        finally:
            fixed.close()
            flexible.close()


class TestFlexibleResolution:
    def test_a_fully_convolutional_model_takes_both(self, tmp_path: object) -> None:
        model = M.create_model("unet").eval()
        exported = cml.export(
            model,
            lucid.randn(1, 1, 64, 64),
            str(tmp_path / "unet.mlpackage"),
            shapes=[(1, 1, 64, 64), (1, 1, 128, 128)],
        )
        try:
            for side in (64, 128):
                x = lucid.randn(1, 1, side, side)
                reference = model(x).logits
                got = exported.predict(x)
                assert got.shape == reference.shape
                scale = float(reference.abs().max().item())
                assert float((got - reference).abs().max().item()) / scale < 1e-5
        finally:
            exported.close()

    def test_an_adaptive_pool_names_itself(self, tmp_path: object) -> None:
        """ResNet's head pools to 1x1 with a kernel taken from the input.

        Traced at 224 the kernel is 7; at 256 it is 8. Fixing one and
        accepting both shapes would give a package that is wrong at the
        other, with nothing about it looking wrong.
        """
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        with pytest.raises(cml.ShapeNotFlexible) as excinfo:
            cml.export(
                model,
                lucid.randn(1, 3, 224, 224),
                str(tmp_path / "adaptive.mlpackage"),
                shapes=[(1, 3, 224, 224), (1, 3, 256, 256)],
            )
        assert excinfo.value.op_name == "avg_pool2d"
        assert "kernel_size" in str(excinfo.value)


class TestFlexibleRefusals:
    def test_a_single_shape_is_not_flexible(self, tmp_path: object) -> None:
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        with pytest.raises(ValueError, match="at least one shape besides"):
            cml.export(
                model,
                lucid.randn(1, 3, 224, 224),
                str(tmp_path / "one.mlpackage"),
                shapes=[(1, 3, 224, 224)],
            )

    def test_more_than_one_input_is_refused(self, tmp_path: object) -> None:
        import lucid.nn as nn

        class TwoInputs(nn.Module):
            def forward(self, a: lucid.Tensor, b: lucid.Tensor) -> lucid.Tensor:
                return a + b

        with pytest.raises(ValueError, match="single-input"):
            cml.export(
                TwoInputs().eval(),
                (lucid.randn(1, 4), lucid.randn(1, 4)),
                str(tmp_path / "two.mlpackage"),
                shapes=[(1, 4), (2, 4)],
            )
