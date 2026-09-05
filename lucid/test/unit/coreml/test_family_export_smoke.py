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

#: One factory per family, with the input it takes: ``img`` for pixels
#: and ``ids`` for token indices. Small on purpose — this is checking
#: that the translation holds, not how fast it runs.
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
    # Segmentation: one channel in, a map out.
    ("unet", (1, 1, 64, 64)),
    ("res_unet_2d", (1, 1, 64, 64)),
    ("attention_unet", (1, 1, 64, 64)),
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

#: Families this file used to name in ``NOT_SINGLE_IMAGE`` on grounds
#: that turned out to be false. Each takes a single tensor and exports;
#: they are separated only because they return several outputs — boxes
#: beside scores, a reconstruction beside a latent — so the comparison
#: goes through ``verify`` rather than through one tensor.
MULTI_OUTPUT = [
    ("fcn_resnet101", (1, 3, 224, 224)),
    ("yolo", (1, 3, 448, 448)),
    ("detr_resnet101", (1, 3, 224, 224)),
    ("vqvae", (1, 3, 64, 64)),
    ("nice_cifar", (1, 3072)),
]

#: Families whose input is token indices rather than pixels.
TOKEN_FAMILIES = [
    ("gpt", (1, 16)),
    # Two heads, so ``predict`` answers with a dict keyed by field name
    # rather than one tensor — the comparison goes through ``verify``.
    ("bert_tiny", (1, 16)),
]

#: Families whose forward takes more than one tensor: a diffusion network
#: is a function of a sample *and* a time, and a sequence-to-sequence
#: model of two token streams. The sampler loop around them is Python and
#: does not export — the network it calls is what a deployment runs, and
#: that is what these check.
#:
#: Several of these zero-initialise their output projection, which is
#: ordinary practice and makes an untrained model answer with exact
#: zeros; ``verify`` refuses that comparison, correctly, so the fixture
#: perturbs the zeroed parameters first.
MULTI_INPUT = [
    ("ddpm_cifar", lambda: (lucid.randn(1, 3, 32, 32), lucid.zeros(1).to(lucid.int64))),
    ("flow_matching_cifar", lambda: (lucid.randn(1, 3, 32, 32), lucid.zeros(1))),
    (
        "ncsn_celeba",
        lambda: (lucid.randn(1, 3, 32, 32), lucid.zeros(1).to(lucid.int64)),
    ),
    ("dit_base_2", lambda: (lucid.randn(1, 4, 32, 32), lucid.zeros(1).to(lucid.int64))),
    (
        "mean_flow_base_2",
        lambda: (lucid.randn(1, 4, 32, 32), lucid.zeros(1), lucid.zeros(1)),
    ),
    (
        "transformer_base",
        lambda: (
            lucid.zeros(1, 16).to(lucid.int64),
            lucid.zeros(1, 16).to(lucid.int64),
        ),
    ),
]

#: Families this file does not reach, and why. Listed rather than
#: omitted: an absence with no reason attached reads as an oversight
#: later, and some of these are boundaries rather than gaps.
#:
#: The list used to be twice this size, and most of what left it was
#: never a boundary — ``fcn`` was excluded for a shape it does not want,
#: ``yolo`` and ``detr`` for post-processing that is not in their
#: forward at all, and the diffusion families for a sampler loop that
#: wraps the network rather than being inside it. Each entry below now
#: names something that was observed, not assumed.
NOT_SINGLE_IMAGE = {
    # Keys, not factory names: the family of ``mask2former_swin_base`` is
    # ``mask``, the same as ``mask_rcnn``'s. Crude on purpose — the point
    # is that nothing drops out of sight, not that the taxonomy is exact.
    #
    # Two-stage detectors: the proposal stage is data dependent, and the
    # models are large enough that a representative here would dominate
    # the suite's runtime. ``detr`` and ``yolo`` are covered above — one
    # stage, and their non-maximum suppression is post-processing the
    # caller runs, not part of ``forward``.
    "fast",
    "faster",
    "mask",
    "maskformer",
    "efficientdet",
    "rcnn",
    # Draws random numbers inside ``forward``: a variational encoder
    # samples its latent, so the traced graph is one draw and there is
    # nothing stable to compare against. Not a translation gap — a model
    # whose output is random has no reproducible answer to export.
    "hvae",
    "score",
    "vae",
    # Two inputs of different kinds (pixels beside token ids), and a
    # mixed-dtype multiply Core ML will not promote. Reachable, but not
    # yet.
    "clip",
    # Reshapes to rank 6, past Core ML's rank-5 program limit, in its
    # patch rearrangement. The staging that fixed window partition
    # applies here too and has not been done.
    "rectified",
    # A step function over latents and actions whose input specification
    # is a rollout, not a tensor: covering these needs an agreed shape
    # for observations and actions first.
    "diamond",
    "dreamer",
    "planet",
    "neural",
    "stable",
    # Covered above by a representative whose family key differs:
    # ``nice_cifar`` stands for ``nice`` and ``realnvp``, ``vqvae`` for
    # itself, ``ddpm``/``flow``/``ncsn``/``dit``/``mean`` and
    # ``transformer``/``bert`` are in the lists above.
    "realnvp",
    "bert",
    "roformer",
    # ``roformer`` exports and is a well-formed package, but Core ML
    # will not build an execution plan for it on the GPU or the Neural
    # Engine — it loads with CPU_ONLY.  That is the accelerator planner
    # refusing the graph, not a translation gap, and the runtime now
    # says so by name rather than reporting "Error in building plan".
    "roformer",
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


@pytest.mark.parametrize(("factory", "shape"), FAMILIES, ids=[f for f, _s in FAMILIES])
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


@pytest.mark.parametrize(
    ("factory", "shape"), TOKEN_FAMILIES, ids=[f for f, _s in TOKEN_FAMILIES]
)
def test_a_token_model_exports_and_matches(factory, shape, tmp_path):
    """Indices in, not pixels — a different feed dtype end to end."""
    lucid.manual_seed(0)
    model = M.create_model(factory).eval()
    x = lucid.zeros(*shape).to(lucid.int64)
    reference = _tensor_of(model(x))
    scale = max(float(reference.abs().max().item()), 1e-6)

    exported = cml.export(model, x, str(tmp_path / f"{factory}.mlpackage"))
    try:
        # ``verify`` rather than ``predict``: a model with two heads
        # answers with a dict, and the comparison should not depend on
        # how many the factory happens to have.
        assert exported.verify(model, x) / scale < 1e-4
    finally:
        exported.close()


def _without_zero_initialised_parameters(factory: str) -> object:
    """The same model with its zeroed parameters given small values.

    Diffusion networks zero-initialise their output projection so that
    the residual branch starts as the identity. An untrained one
    therefore answers with exact zeros, and ``verify`` refuses that
    comparison — rightly, since an export that dropped every layer would
    score the same. Perturbing first is what the refusal asks for.
    """
    model = M.create_model(factory).eval()
    replaced = {}
    for name, value in model.state_dict().items():
        zeroed = (
            hasattr(value, "shape")
            and int(value.numel()) > 0
            and float(value.abs().max().item()) == 0.0
        )
        replaced[name] = lucid.randn(*value.shape) * 0.05 if zeroed else value
    perturbed = M.create_model(factory).eval()
    perturbed.load_state_dict(replaced)
    return perturbed


@pytest.mark.parametrize(
    ("factory", "shape"), MULTI_OUTPUT, ids=[f for f, _s in MULTI_OUTPUT]
)
def test_a_multi_output_family_exports_and_matches(factory, shape, tmp_path):
    """One tensor in, several out — compared through ``verify``.

    ``relative=True`` because the outputs differ in magnitude: a flow
    returns a latent of order one beside a log-probability of order
    ten thousand, and the absolute worst across them is set by the
    second whatever the first did.
    """
    lucid.manual_seed(0)
    model = M.create_model(factory).eval()
    x = lucid.randn(*shape)

    exported = cml.export(model, x, str(tmp_path / f"{factory}.mlpackage"))
    try:
        assert exported.verify(model, x, relative=True) < 1e-4
    finally:
        exported.close()


@pytest.mark.parametrize(
    ("factory", "make_inputs"), MULTI_INPUT, ids=[f for f, _m in MULTI_INPUT]
)
def test_a_multi_input_family_exports_and_matches(factory, make_inputs, tmp_path):
    """A network of a sample and a time, or of two token streams.

    ``relative=True`` because these return outputs of different
    magnitudes — a latent beside a log-probability — and the absolute
    worst across them says nothing about either.
    """
    lucid.manual_seed(0)
    model = _without_zero_initialised_parameters(factory)
    inputs = make_inputs()

    exported = cml.export(model, inputs, str(tmp_path / f"{factory}.mlpackage"))
    try:
        assert exported.verify(model, inputs, relative=True) < 1e-4
    finally:
        exported.close()


def test_every_family_is_either_covered_or_named():
    """No family drops out of sight.

    A representative here, or an entry in ``NOT_SINGLE_IMAGE`` saying
    what shape it wants instead. Adding a family to the zoo without
    doing either fails this.
    """
    import re

    covered = {
        re.split(r"[_\d]", f)[0]
        for f, _s in (*FAMILIES, *TOKEN_FAMILIES, *MULTI_OUTPUT, *MULTI_INPUT)
    }
    families = {re.split(r"[_\d]", n)[0] for n in M.list_models()}
    missing = families - covered - NOT_SINGLE_IMAGE
    assert not missing, (
        f"architecture families with no export coverage and no reason given: "
        f"{sorted(missing)}"
    )
