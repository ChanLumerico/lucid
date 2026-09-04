"""One model per architecture family, exported and compared.

The rest of this directory tests operations one at a time and models
three at a time. That is what let a `gather` emitted for the wrong
opset, a rank cap that refused every windowed transformer, and an
operation listed as "never traced" that LP pooling traces, all sit in
the tree at once: each was reachable only from an architecture nothing
exported.

So this walks families rather than operations. It is deliberately not
the whole zoo — 440 factories at inference size is not a unit test — but
one representative per family, at a small input, is enough to keep the
class of defect from returning silently. Everything here passed when it
was written; a new entry that fails is a real gap, not a flaky test.

Families whose input is not a single image tensor (detection heads with
multiple inputs, generative samplers, world models) are named in
``NOT_SINGLE_IMAGE`` rather than left out, so the list says what it does
not cover.
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

#: One factory per family, with the input it takes. Small on purpose:
#: this is checking that the translation holds, not how fast it runs.
FAMILIES = [
    ("alexnet", (1, 3, 224, 224)),
    ("convnext_tiny", (1, 3, 224, 224)),
    ("crossvit_9", (1, 3, 224, 224)),
    ("cspdarknet_53", (1, 3, 224, 224)),
    ("coatnet_0", (1, 3, 224, 224)),
    ("cspresnet_50", (1, 3, 224, 224)),
    ("cspresnext_50", (1, 3, 224, 224)),
    ("cvt_13", (1, 3, 224, 224)),
    ("densenet_121", (1, 3, 224, 224)),
    ("efficientformer_l1", (1, 3, 224, 224)),
    ("efficientnet_b0", (1, 3, 224, 224)),
    ("googlenet", (1, 3, 224, 224)),
    ("inception_v3", (1, 3, 224, 224)),
    ("lenet_5", (1, 1, 32, 32)),
    ("maxvit_tiny", (1, 3, 224, 224)),
    ("mobilenet", (1, 3, 224, 224)),
    ("pvt_v2_b0", (1, 3, 224, 224)),
    ("resnest_14", (1, 3, 224, 224)),
    ("resnet_18", (1, 3, 224, 224)),
    ("resnext_50_32x4d", (1, 3, 224, 224)),
    ("se_resnet_18", (1, 3, 224, 224)),
    ("sk_resnet_18", (1, 3, 224, 224)),
    ("swin_tiny", (1, 3, 224, 224)),
    ("vgg_11", (1, 3, 224, 224)),
    ("vit_base_16", (1, 3, 224, 224)),
    ("wide_resnet_50", (1, 3, 224, 224)),
    ("xception", (1, 3, 224, 224)),
    ("zfnet", (1, 3, 224, 224)),
]

#: Families this file does not reach, and why. Listed rather than
#: omitted: an absence with no reason attached reads as an oversight
#: later, and some of these are boundaries rather than gaps.
NOT_SINGLE_IMAGE = {
    # Keys, not factory names: the family of ``mask2former_swin_base`` is
    # ``mask``, the same as ``mask_rcnn``'s. Crude on purpose — the point
    # is that nothing drops out of sight, not that the taxonomy is exact.
    #
    # Two or more inputs, or a head whose post-processing is data
    # dependent (non-maximum suppression, proposals) and cannot be a
    # static graph at all.
    "detr",
    "fast",
    "faster",
    "mask",
    "maskformer",
    "efficientdet",
    "yolo",
    "rcnn",
    # Token ids, not pixels.
    "bert",
    "gpt",
    "roformer",
    "transformer",
    "clip",
    # Samplers and world models: a step function over latents and time,
    # driven from Python.
    "ddpm",
    "diamond",
    "dit",
    "dreamer",
    "flow",
    "hvae",
    "mean",
    "ncsn",
    "neural",
    "nice",
    "planet",
    "rectified",
    "realnvp",
    "score",
    "stable",
    "vae",
    "vqvae",
    # Segmentation shapes that are not 224-square.
    "attention",
    "fcn",
    "res",
    "unet",
}


def _tensor_of(result: object) -> lucid.Tensor:
    """Zoo factories return output dataclasses as often as tensors."""
    if isinstance(result, lucid.Tensor):
        return result
    for field in ("logits", "out", "output", "last_hidden_state"):
        value = getattr(result, field, None)
        if isinstance(value, lucid.Tensor):
            return value
    for value in vars(result).values():
        if isinstance(value, lucid.Tensor):
            return value
    raise AssertionError(f"no tensor in {type(result).__name__}")


@pytest.mark.parametrize(
    ("factory", "shape"), FAMILIES, ids=[f for f, _s in FAMILIES]
)
def test_a_family_representative_exports_and_matches(factory, shape, tmp_path):
    lucid.manual_seed(0)
    model = M.create_model(factory).eval()
    x = lucid.randn(*shape)
    reference = _tensor_of(model(x))
    scale = max(float(reference.abs().max().item()), 1e-6)

    exported = cml.export(model, x, str(tmp_path / f"{factory}.mlpackage"))
    try:
        got = exported.predict(x)
        assert tuple(got.shape) == tuple(reference.shape)
        assert float((got - reference).abs().max().item()) / scale < 1e-4
    finally:
        exported.close()


def test_every_family_is_either_covered_or_named():
    """No family drops out of sight.

    A representative here, or an entry in ``NOT_SINGLE_IMAGE`` saying
    what shape it wants instead. Adding a family to the zoo without
    doing either fails this.
    """
    import re

    covered = {re.split(r"[_\d]", f)[0] for f, _s in FAMILIES}
    families = {re.split(r"[_\d]", n)[0] for n in M.list_models()}
    missing = families - covered - NOT_SINGLE_IMAGE
    assert not missing, (
        f"architecture families with no export coverage and no reason given: "
        f"{sorted(missing)}"
    )
