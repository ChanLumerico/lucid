"""Every option, crossed with every other option.

Each feature of this subsystem has tests. Their combinations did not,
and that is where the last three defects of this kind came from: a
float16 body with an integer output, constant folding beside the rank-6
rewrite, constant folding beside carried state. Each was a pair that
worked separately.

Writing this file found a fourth. Palettizing a float16 export failed
outright — the weight is rounded to half before the palette is fitted,
and the edge table's sentinel is single precision, so Lloyd's algorithm
tried to concatenate the two. Nothing in the single-feature tests could
see it, because neither feature is wrong on its own.

The matrix is small on purpose. Thirty-two exports of a six-layer model
run in about a minute; a matrix that took ten would be one nobody runs.
What matters is that the axes are the ones that interact — how the body
computes, how the weights are stored, and what shape of interface the
package presents.
"""

import itertools

import pytest

import lucid
import lucid.nn as nn
import lucid.coreml as cml
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"),
    reason="the engine was built without the Core ML writer",
)

#: How the body computes.
PRECISIONS = [
    ("fp32", cml.Precision.FLOAT32),
    ("fp16", cml.Precision.FLOAT16),
]

#: How the weights are stored. Each takes a different path through the
#: writer: plain blob, int8 codes with per-channel scales, sub-byte keys
#: into grouped tables, survivors plus a bit mask.
WEIGHTS = [
    ("float", cml.WeightPrecision.FLOAT),
    ("int8", cml.WeightPrecision.INT8),
    ("palette", cml.Palettize(bits=4)),
    ("sparse", cml.Sparsify(ratio=0.5)),
]

#: What the package presents. A flexible input, a ranged one and a
#: classifier each rewrite the interface after the body is built.
INTERFACES = [
    ("fixed", {}),
    ("enumerated", {"shapes": [(1, 3, 32, 32), (2, 3, 32, 32)]}),
    ("ranged", {"shape_range": {0: (1, 4)}}),
    ("classifier", {"classifier": cml.Classifier(labels=[f"c{i}" for i in range(5)])}),
]

CASES = list(itertools.product(PRECISIONS, WEIGHTS, INTERFACES))


class _Net(nn.Module):
    """Deep enough to compress, small enough to export thirty-two times.

    A convolution stack and a linear head: between them they reach the
    weight paths that differ — a four-dimensional weight with an output
    channel axis, and a two-dimensional one.
    """

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(32, 5)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.head(self.body(x).mean(dim=(2, 3)))


@pytest.mark.parametrize(
    ("precision", "weights", "interface"),
    CASES,
    ids=[f"{p[0]}-{w[0]}-{i[0]}" for p, w, i in CASES],
)
def test_a_combination_exports_and_runs(precision, weights, interface, tmp_path):
    """Built, loaded, and asked for an answer.

    The check is deliberately that it *runs*, not that it agrees closely:
    four-bit palettization and half sparsity are lossy by design, and a
    tolerance loose enough to admit them would admit anything. What the
    combinations break is the writing and the loading, which either
    happen or do not.
    """
    _precision_name, precision_value = precision
    _weights_name, weights_value = weights
    interface_name, interface_kwargs = interface

    lucid.manual_seed(0)
    model = _Net().eval()
    x = lucid.randn(1, 3, 32, 32)

    exported = cml.export(
        model,
        x,
        str(tmp_path / "combination.mlpackage"),
        precision=precision_value,
        weights=weights_value,
        **interface_kwargs,
    )
    try:
        if interface_name == "classifier":
            label, probabilities = exported.classify(x)
            assert label in {f"c{i}" for i in range(5)}
            assert len(probabilities) == 5
        else:
            got = exported.predict(x)
            assert tuple(got.shape) == (1, 5)
    finally:
        exported.close()


def test_a_lossless_combination_still_agrees(tmp_path: object) -> None:
    """One case where closeness is meaningful, to keep the rest honest.

    The matrix above only asks whether each combination runs, which
    would pass for a package that runs and returns nonsense. Float32
    with plain weights has no reason to differ from the model at all, so
    it is checked against it — if the writing were broken in a way the
    matrix cannot see, this is where it shows.
    """
    lucid.manual_seed(0)
    model = _Net().eval()
    x = lucid.randn(1, 3, 32, 32)

    exported = cml.export(model, x, str(tmp_path / "exact.mlpackage"))
    try:
        assert exported.verify(model, x, relative=True) < 1e-5
    finally:
        exported.close()


def test_palettizing_a_half_precision_export(tmp_path: object) -> None:
    """The defect this file was written to catch, kept by name.

    The weight is rounded to half before its palette is fitted, and the
    edge table's sentinel is single precision — so the fit tried to
    concatenate the two and the export failed outright. Palettes are
    fitted in single precision now, which the weight has already been
    rounded past, so nothing is lost by it.
    """
    lucid.manual_seed(0)
    model = _Net().eval()
    x = lucid.randn(1, 3, 32, 32)

    errors = {}
    for bits in (2, 4, 8):
        exported = cml.export(
            model,
            x,
            str(tmp_path / f"half_{bits}.mlpackage"),
            precision=cml.Precision.FLOAT16,
            weights=cml.Palettize(bits=bits),
        )
        try:
            errors[bits] = exported.verify(model, x, relative=True)
        finally:
            exported.close()
    # More bits, less error — a fit that had gone wrong would not be
    # ordered.
    assert errors[8] < errors[4] < errors[2]
