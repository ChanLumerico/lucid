"""Agreement measured on weights that mean something.

Every other numeric test here runs on a freshly initialised model, which
answers a narrower question than it looks like it does. An untrained
classifier's logits are nearly uniform — several zoo heads start at
exactly zero — so a reordering under float16 says nothing about whether
float16 is safe, and an unchanged ordering says nothing either.

With trained weights the question has an answer: does the exported
package predict what the model predicts? These tests ask it of the
prediction, not of the tensor, because that is what a deployment cares
about.

Skipped when the weights are not already on the machine. They are a
download, and a test that reaches the network is a test that fails for
reasons that have nothing to do with the code.
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

FACTORY = "alexnet_cls"


def _trained() -> object:
    try:
        return M.create_model(FACTORY, pretrained=True).eval()
    except Exception as why:  # noqa: BLE001 — any failure means "not available"
        pytest.skip(f"pretrained {FACTORY} is not on this machine: {why}")


def _ranking(scores: lucid.Tensor, k: int = 5) -> list[int]:
    values = scores.reshape(-1).tolist()
    return sorted(range(len(values)), key=lambda i: -values[i])[:k]


class TestTrainedAgreement:
    def test_float16_predicts_what_the_model_predicts(
        self, tmp_path: object
    ) -> None:
        """The claim the Neural Engine path rests on.

        float16 is the only precision the accelerator runs, so if it
        moved predictions the whole subsystem would be a size trade
        rather than a speed one. Measured over several inputs: it does
        not — the top result and the top five are the model's own.
        """
        model = _trained()
        exported = cml.export(
            model,
            lucid.randn(1, 3, 224, 224),
            f"{tmp_path}/half.mlpackage",
            precision=cml.Precision.FLOAT16,
            compute_units=cml.ComputeUnits.CPU_AND_NE,
        )
        try:
            for _ in range(4):
                x = lucid.randn(1, 3, 224, 224)
                wanted = _ranking(model(x).logits)
                got = _ranking(exported.predict(x))
                assert got[0] == wanted[0]
                assert set(got) == set(wanted)
        finally:
            exported.close()

    def test_int8_holds_the_top_result_and_costs_the_rest(
        self, tmp_path: object
    ) -> None:
        """Quantization is a trade, and this is which half is paid.

        Measured across three trained families: the winning class
        survives, and the ordering below it does not always — AlexNet
        lost one top-1 in eight and DenseNet four top-5 places in thirty.
        Asserting only the top result keeps the test honest about what
        int8 actually preserves.
        """
        model = _trained()
        exported = cml.export(
            model,
            lucid.randn(1, 3, 224, 224),
            f"{tmp_path}/int8.mlpackage",
            precision=cml.Precision.FLOAT16,
            weights=cml.WeightPrecision.INT8,
            compute_units=cml.ComputeUnits.CPU_AND_NE,
        )
        try:
            agreed = 0
            for _ in range(6):
                x = lucid.randn(1, 3, 224, 224)
                wanted = _ranking(model(x).logits)
                got = _ranking(exported.predict(x))
                agreed += got[0] == wanted[0]
            assert agreed >= 5
        finally:
            exported.close()

    def test_a_trained_head_has_something_to_compare(
        self, tmp_path: object
    ) -> None:
        """Why this file exists at all.

        The untrained factory's logits are flat enough that any
        perturbation reorders them, which is what made a float16
        comparison on one look like a failure earlier. A trained one
        separates its classes, so a preserved ordering means something.
        """
        model = _trained()
        scores = sorted(
            model(lucid.randn(1, 3, 224, 224)).logits.reshape(-1).tolist(),
            reverse=True,
        )
        scale = max(abs(scores[0]), abs(scores[-1]))
        assert (scores[0] - scores[4]) / scale > 0.05
