"""Core ML export: does the package Lucid writes compute what Lucid does?

This is the only route Lucid has to the Neural Engine — neither Accelerate
nor MLX targets it — and the only way a model trained here reaches an
Apple app. The exporter reuses the compile tracer's graph, so it is a
second backend for an IR that already exists rather than a new front end.

What is asserted is agreement with the eager model, not merely that a
file appears. An exporter that drops a layer still writes a valid
``.mlpackage`` and still returns plausible numbers; only a comparison
against the source model catches it. Tolerances are tight because the
default export is FLOAT32 — the looser FLOAT16 path (what the ANE wants)
is a separate assertion with its own bound.
"""

import numpy as np
import pytest

import lucid
import lucid.models as models
import lucid.nn as nn

ct = pytest.importorskip(
    "coremltools", reason="Core ML export is dev tooling: pip install lucid-dl[coreml]"
)
export_coreml = pytest.importorskip(
    "tools.export_coreml", reason="run pytest from the repository root"
)

export = export_coreml.export
UnsupportedOp = export_coreml.UnsupportedOp


class _Tiny(nn.Module):
    """One of each op class the mapped set covers."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 4 * 4, num_classes)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        h = nn.functional.relu(self.bn(self.conv(x)))
        h = self.pool(h)
        return self.fc(h.reshape(h.shape[0], -1))


def _predict(mlmodel: object, x: lucid.Tensor) -> np.ndarray:
    key = list(mlmodel.input_description)[0]
    out = mlmodel.predict({key: x.numpy()})
    return np.asarray(list(out.values())[0])


class TestAgreementWithEager:
    def test_a_tiny_model_matches(self, tmp_path: object) -> None:
        model = _Tiny().eval()
        x = lucid.randn(1, 3, 8, 8)
        want = model(x).numpy()

        mlmodel = export(model, x, str(tmp_path / "tiny.mlpackage"))

        got = _predict(mlmodel, x).reshape(want.shape)
        assert np.abs(want - got).max() < 1e-5

    def test_a_zoo_classifier_matches(self, tmp_path: object) -> None:
        # The real target: an unmodified factory from the model zoo,
        # output dataclass and all.
        model = models.create_model("resnet_18_cls", num_classes=10).eval()
        x = lucid.randn(1, 3, 224, 224)
        want = model(x).logits.numpy()

        mlmodel = export(model, x, str(tmp_path / "resnet18.mlpackage"))

        got = _predict(mlmodel, x).reshape(want.shape)
        assert np.abs(want - got).max() < 1e-4

    def test_float16_targets_the_neural_engine(self, tmp_path: object) -> None:
        # fp16 is what the ANE runs; the looser bound is the cost.
        model = _Tiny().eval()
        x = lucid.randn(1, 3, 8, 8)
        want = model(x).numpy()

        mlmodel = export(
            model,
            x,
            str(tmp_path / "half.mlpackage"),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            compute_precision=ct.precision.FLOAT16,
        )

        got = _predict(mlmodel, x).reshape(want.shape)
        assert np.abs(want - got).max() < 1e-2


class TestOutputSelection:
    def test_logits_are_found_on_an_output_dataclass(self, tmp_path: object) -> None:
        model = models.create_model("alexnet_cls", num_classes=4).eval()
        x = lucid.randn(1, 3, 224, 224)

        mlmodel = export(model, x, str(tmp_path / "alex.mlpackage"))

        assert _predict(mlmodel, x).reshape(1, 4).shape == (1, 4)

    def test_a_named_field_wins(self, tmp_path: object) -> None:
        model = models.create_model("alexnet_cls", num_classes=4).eval()
        x = lucid.randn(1, 3, 224, 224)

        mlmodel = export(
            model, x, str(tmp_path / "alex2.mlpackage"), output_field="logits"
        )

        assert _predict(mlmodel, x).reshape(1, 4).shape == (1, 4)


class TestRefusals:
    """A refusal names the gap; the alternative is a quietly wrong model."""

    def test_an_unmapped_op_names_itself(self, tmp_path: object) -> None:
        class UsesAnUnmappedOp(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return lucid.sin(x)

        with pytest.raises(UnsupportedOp) as excinfo:
            export(
                UsesAnUnmappedOp().eval(),
                lucid.randn(1, 4),
                str(tmp_path / "nope.mlpackage"),
            )

        assert excinfo.value.op_name == "sin"
        assert "sin" in str(excinfo.value)

    def test_a_training_mode_model_is_refused(self, tmp_path: object) -> None:
        model = _Tiny().train()

        with pytest.raises(ValueError, match="training mode"):
            export(model, lucid.randn(1, 3, 8, 8), str(tmp_path / "train.mlpackage"))
