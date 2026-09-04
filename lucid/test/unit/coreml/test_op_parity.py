"""Per-op agreement between an exported package and the eager model.

An end-to-end model test proves the ops that model happens to use. It
says nothing about the rest of the table, and a wrong emitter in an
unused corner waits there until someone exports a model that reaches it.
These run one op at a time.

Every case compares values, never shapes: an emitter bound to the wrong
MIL op — ``negative_slope`` read where Lucid writes ``slope``, ``axis``
where it writes ``dim`` — produces the right shape and the wrong numbers.
"""

import pytest

import lucid
import lucid.coreml as cml
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"),
    reason="the engine was built without the Core ML writer",
)


class _Apply(nn.Module):
    """One traced callable, so an op can be exported on its own."""

    def __init__(self, fn: object) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fn(x)  # type: ignore[operator]


def _check(fn: object, x: lucid.Tensor, tmp_path: object, tol: float = 1e-5) -> None:
    model = _Apply(fn).eval()
    reference = model(x)
    exported = cml.export(model, x, f"{tmp_path}/op.mlpackage")
    try:
        got = exported.predict(x)
        assert got.shape == reference.shape
        scale = float(reference.abs().max().item()) or 1.0
        assert float((got - reference).abs().max().item()) / scale < tol
    finally:
        exported.close()


# ── the tables ───────────────────────────────────────────────────────────────
#
# ``positive`` marks the ops whose domain excludes the negatives that
# ``randn`` would otherwise hand them.

_UNARY = [
    ("abs", lucid.abs, False),
    ("arccos", lambda x: lucid.arccos(lucid.tanh(x)), False),
    ("arcsin", lambda x: lucid.arcsin(lucid.tanh(x)), False),
    ("arctan", lucid.arctan, False),
    ("ceil", lucid.ceil, False),
    ("cos", lucid.cos, False),
    ("cosh", lucid.cosh, False),
    ("erf", lucid.erf, False),
    ("exp", lucid.exp, False),
    ("floor", lucid.floor, False),
    ("log", lucid.log, True),
    ("log2", lucid.log2, True),
    ("neg", lambda x: -x, False),
    ("reciprocal", lucid.reciprocal, True),
    ("round", lucid.round, False),
    ("rsqrt", lucid.rsqrt, True),
    ("sign", lucid.sign, False),
    ("sin", lucid.sin, False),
    ("sinh", lucid.sinh, False),
    ("sqrt", lucid.sqrt, True),
    ("square", lucid.square, False),
    ("tan", lucid.tan, False),
]

_ACTIVATION = [
    ("elu", F.elu),
    ("mish", F.mish),
    ("selu", F.selu),
    ("softplus", F.softplus),
    ("log_softmax", lambda x: F.log_softmax(x, dim=1)),
]

_ACTIVATION += [
    ("hard_sigmoid", F.hardsigmoid),
    ("hard_swish", F.hardswish),
]

_SPATIAL = [
    ("conv1d", nn.Conv1d(4, 2, 3).eval(), (1, 4, 10)),
    ("conv2d", nn.Conv2d(4, 2, 3).eval(), (1, 4, 6, 6)),
    ("conv3d", nn.Conv3d(4, 2, 3).eval(), (1, 4, 6, 6, 6)),
    # A sliding window along one axis.  MIL has the operation and puts
    # the window axis directly after the one it slid along, where Lucid
    # appends it last — so the emitter is that op plus a transpose, and
    # a wrong reconciliation shows up here as values rather than shapes.
    ("unfold_dim", lambda x: x.unfold(2, 3, 2), (1, 2, 8, 4)),
    # ``x ** c`` and ``c ** x`` with the constant on the trace.  Reached
    # through LP pooling and the p-norms.
    ("pow_scalar", lambda x: F.lp_pool2d(x.abs() + 0.5, 2.0, 2), (1, 2, 8, 8)),
    # MIL's ``conv_transpose`` takes the weight as (C_in, C_out / groups,
    # *K) — the layout Lucid already stores — so grouping needs no
    # relayout on this path.  The cases are here because nothing else
    # would notice if that stopped being true.
    ("conv_transpose1d", nn.ConvTranspose1d(4, 2, 3, stride=2).eval(), (1, 4, 10)),
    ("conv_transpose2d", nn.ConvTranspose2d(4, 2, 3, stride=2).eval(), (1, 4, 6, 6)),
    (
        "conv_transpose3d",
        nn.ConvTranspose3d(4, 2, 3, stride=2).eval(),
        (1, 4, 6, 6, 6),
    ),
    (
        "conv_transpose1d_grouped",
        nn.ConvTranspose1d(4, 4, 3, stride=2, groups=2).eval(),
        (1, 4, 10),
    ),
    (
        "conv_transpose2d_grouped",
        nn.ConvTranspose2d(4, 4, 3, stride=2, groups=2).eval(),
        (1, 4, 6, 6),
    ),
    (
        "conv_transpose2d_depthwise",
        nn.ConvTranspose2d(4, 4, 3, stride=2, groups=4).eval(),
        (1, 4, 6, 6),
    ),
    (
        "conv_transpose3d_grouped",
        nn.ConvTranspose3d(4, 4, 3, stride=2, groups=2).eval(),
        (1, 4, 6, 6, 6),
    ),
    (
        "conv_transpose1d_dilated",
        nn.ConvTranspose1d(4, 2, 3, stride=2, dilation=2).eval(),
        (1, 4, 10),
    ),
    (
        "conv_transpose2d_dilated",
        nn.ConvTranspose2d(4, 2, 3, stride=2, dilation=2).eval(),
        (1, 4, 6, 6),
    ),
    ("max_pool1d", lambda x: F.max_pool1d(x, 2), (1, 4, 10)),
    ("max_pool2d", lambda x: F.max_pool2d(x, 2), (1, 4, 6, 6)),
    ("max_pool3d", lambda x: F.max_pool3d(x, 2), (1, 4, 6, 6, 6)),
    ("avg_pool1d", lambda x: F.avg_pool1d(x, 2), (1, 4, 10)),
    ("avg_pool2d", lambda x: F.avg_pool2d(x, 2), (1, 4, 6, 6)),
    ("avg_pool3d", lambda x: F.avg_pool3d(x, 2), (1, 4, 6, 6, 6)),
    ("group_norm", lambda x: F.group_norm(x, 2), (1, 4, 6, 6)),
    ("rms_norm", lambda x: F.rms_norm(x, (6,)), (1, 4, 6, 6)),
]

_OTHER = [
    ("clip", lambda x: lucid.clip(x, -0.5, 0.5), False),
    ("diagonal", lucid.diagonal, False),
    ("norm", lambda x: lucid.linalg.norm(x), False),
    ("repeat", lambda x: lucid.repeat(x, 2, dim=1), False),
    ("split", lambda x: lucid.split(x, 2, dim=1)[0], False),
    ("flip", lambda x: lucid.flip(x, dims=1), False),
    ("masked_fill", lambda x: lucid.masked_fill(x, x > 0, 0.0), False),
    ("maximum", lambda x: lucid.maximum(x, x * 2), False),
    ("minimum", lambda x: lucid.minimum(x, x * 2), False),
    ("pad", lambda x: lucid.pad(x, (1, 1, 1, 1)), False),
    ("pow", lambda x: lucid.pow(x, 2.0), True),
    ("prod", lambda x: lucid.prod(x, dim=1), True),
    ("sum", lambda x: lucid.sum(x, dim=1), False),
    ("tile", lambda x: lucid.tile(x, (1, 1, 2, 2)), False),
]


@pytest.mark.parametrize(("name", "fn", "positive"), _UNARY, ids=[c[0] for c in _UNARY])
def test_a_unary_op_matches(
    name: str, fn: object, positive: bool, tmp_path: object
) -> None:
    x = lucid.rand(1, 4, 6, 6) + 0.5 if positive else lucid.randn(1, 4, 6, 6)
    _check(fn, x, tmp_path)


@pytest.mark.parametrize(("name", "fn"), _ACTIVATION, ids=[c[0] for c in _ACTIVATION])
def test_an_activation_matches(name: str, fn: object, tmp_path: object) -> None:
    _check(fn, lucid.randn(1, 4, 6, 6), tmp_path)


@pytest.mark.parametrize(("name", "fn", "positive"), _OTHER, ids=[c[0] for c in _OTHER])
def test_a_structural_op_matches(
    name: str, fn: object, positive: bool, tmp_path: object
) -> None:
    x = lucid.rand(1, 4, 6, 6) + 0.5 if positive else lucid.randn(1, 4, 6, 6)
    _check(fn, x, tmp_path)


class TestTheOutputBufferIsReadCorrectly:
    """Core ML does not always hand back a packed array.

    On the paths that allow the Neural Engine, an output whose innermost
    dimension is not a multiple of the alignment comes back padded. Copied
    as if packed, the values interleave with the padding: right shape,
    wrong numbers, no error. ``tile`` on the last axis is the smallest
    case that produces one.
    """

    @pytest.mark.parametrize(
        "units",
        [cml.ComputeUnits.ALL, cml.ComputeUnits.CPU_ONLY, cml.ComputeUnits.CPU_AND_NE],
        ids=["all", "cpu", "cpu_and_ne"],
    )
    def test_a_padded_output_is_not_read_as_packed(
        self, units: object, tmp_path: object
    ) -> None:
        x = lucid.arange(0, 12).reshape(1, 1, 3, 4) * 1.0
        model = _Apply(lambda t: lucid.tile(t, (1, 1, 1, 2))).eval()
        reference = model(x)
        exported = cml.export(
            model, x, f"{tmp_path}/strided.mlpackage", compute_units=units
        )
        try:
            got = exported.predict(x)
            assert float((got - reference).abs().max().item()) == 0.0
        finally:
            exported.close()


@pytest.mark.parametrize(
    ("name", "fn", "shape"), _SPATIAL, ids=[c[0] for c in _SPATIAL]
)
def test_a_spatial_op_matches(
    name: str, fn: object, shape: tuple[int, ...], tmp_path: object
) -> None:
    _check(fn, lucid.randn(*shape), tmp_path)


class TestTheGapAgainstCompileIsAccountedFor:
    """Every op `lucid.compile` runs either exports, or is listed here.

    The two backends read the same trace, so an op the MPSGraph builder
    accepts is one an export can meet. Leaving that difference unwritten
    is how a model becomes unexportable without anyone noticing which op
    did it; naming each absence makes adding a compile emitter without an
    export emitter a failing test rather than a surprise months later.
    """

    #: Core ML's program dialect has no equivalent and no decomposition.
    IMPOSSIBLE = {
        # No complex dtype.
        "complex",
        "conj",
        "imag",
        "real",
        # No linear-algebra solver.
        "det",
        "inv",
        "solve",
    }

    #: Only reachable from a model in training mode, which export refuses.
    TRAINING_ONLY = {
        "alpha_dropout",
        "bce_loss",
        "bce_with_logits",
        "cross_entropy_loss",
        "drop_block",
        "drop_path",
        "dropoutnd",
        "huber_loss",
        "mse_loss",
        "nll_loss",
    }

    #: In the registry, but decomposed into mapped ops before tracing.
    NEVER_TRACED = {
        "batch_norm",
        "batch_norm1d",
        "batch_norm3d",
        "cube",
        "cube_root",
        # ``rpow_scalar`` belongs to the bucket below rather than this
        # one — nothing calls it — but it is kept here beside its
        # sibling because ``pow_scalar`` was listed here and was wrong:
        # LP pooling traces it, which is how ``zfnet`` found it.  Three
        # models cannot tell a list like this from the truth.
        "rpow_scalar",
        "norm",
        # No Python caller anywhere in ``lucid/``: an engine op with no
        # way to reach it from a model.
        "global_response_norm",
        # ``F.apply_rotary_emb`` decomposes into mapped ops; the engine
        # op it is named after never appears in a trace.
        "rotary_pos_embedding",
    }

    #: The trace does not carry enough to translate them.
    TRACER_GAP = {
        # Operands the tracer does not all wire, re-measured 2026-09-04.
        # Translating from what is recorded would put the wrong values in
        # the result, so each emitter refuses by name rather than
        # guessing at the rest.
        #
        #   meshgrid       one node, N outputs. ``OpScopeFull`` mints a
        #                  single output with an empty shape and each of
        #                  the N ``wire_autograd`` calls overwrites the
        #                  same node, so only the last input survives.
        #                  Carrying it needs a multi-output IR, not a
        #                  wiring fix.
        #   embedding_bag  the weight is recorded, the indices and
        #                  offsets are not. ``embedding`` right above it
        #                  patches this up by calling ``on_op_io``
        #                  explicitly; this one does not, and its
        #                  segments are data-dependent besides.
        #
        # ``grid_sample`` was here until its operands reached the trace.
        # It was never a Core ML limit — MIL has had ``resample`` all
        # along; the op returned before ``wire_autograd`` whenever no
        # gradient was wanted, which is every inference pass.
        "meshgrid",
        "embedding_bag",
    }

    #: Expressible and not written. Empty: every operation
    #: ``lucid.compile`` runs and a trace can carry now exports.
    NOT_YET: set[str] = set()

    def test_no_unaccounted_gap(self) -> None:
        from lucid.coreml._emit import EMITTERS

        registry = {s.name for s in _C_engine.op_registry_all() if not s.internal}
        compiled = {
            name
            for name in registry | set(EMITTERS)
            if _C_engine.compile.emitter_registered(name)
        }
        accounted = (
            self.IMPOSSIBLE
            | self.TRAINING_ONLY
            | self.NEVER_TRACED
            | self.TRACER_GAP
            | self.NOT_YET
        )
        unaccounted = compiled - set(EMITTERS) - accounted
        assert not unaccounted, (
            f"lucid.compile runs {sorted(unaccounted)} and lucid.coreml does not. "
            "Add an emitter, or add the op to one of the sets above with the "
            "reason it cannot have one."
        )

    def test_nothing_listed_is_already_supported(self) -> None:
        from lucid.coreml._emit import EMITTERS

        stale = (self.IMPOSSIBLE | self.NOT_YET) & set(EMITTERS)
        assert not stale, f"{sorted(stale)} are mapped now — take them off the list"


class TestTheOperationsMilDoesNotHave:
    """Each of these is several MIL operations standing in for one.

    A decomposition is where a translation goes quietly wrong — the shape
    survives and the values do not — so every one is compared against the
    eager result rather than merely exported.
    """

    def test_unfold_matches_exactly(self, tmp_path: object) -> None:
        for kernel, stride, padding in ((2, 1, 0), (3, 2, 1)):
            _check(
                lambda x, k=kernel, s=stride, p=padding: F.unfold(
                    x, k, stride=s, padding=p
                ),
                lucid.randn(1, 3, 8, 8),
                tmp_path,
            )

    def test_fold_matches_exactly(self, tmp_path: object) -> None:
        """``fold`` adds where blocks overlap, which is the hard half."""
        for kernel, stride, padding, side in ((2, 1, 0, 6), (2, 2, 0, 6), (3, 1, 1, 8)):
            _check(
                lambda x, k=kernel, s=stride, p=padding, n=side: F.fold(
                    F.unfold(x, k, stride=s, padding=p), (n, n), k, stride=s, padding=p
                ),
                lucid.randn(1, 2, side, side),
                tmp_path,
            )

    def test_affine_grid_matches(self, tmp_path: object) -> None:
        for corners in (False, True):
            _check(
                lambda x, c=corners: F.affine_grid(
                    x.reshape(1, 2, 3), (1, 4, 6, 6), align_corners=c
                ),
                lucid.randn(1, 6),
                tmp_path,
            )

    def test_bilinear_and_lp_normalize_match(self, tmp_path: object) -> None:
        weight = lucid.randn(3, 4, 4)
        _check(lambda x: F.bilinear(x, x, weight), lucid.randn(1, 4), tmp_path)
        for order in (1, 2):
            _check(
                lambda x, p=order: F.normalize(x, p=p, dim=1),
                lucid.randn(1, 4, 6, 6),
                tmp_path,
            )

    def test_three_dimensional_resampling_matches(self, tmp_path: object) -> None:
        for mode in ("nearest", "trilinear"):
            _check(
                lambda x, m=mode: F.interpolate(x, scale_factor=2, mode=m),
                lucid.randn(1, 2, 4, 4, 4),
                tmp_path,
            )

    def test_erfinv_is_an_approximation_and_says_how_close(
        self, tmp_path: object
    ) -> None:
        """MIL has no inverse error function; this one is a polynomial.

        Looser than the other ops on purpose, and measured rather than
        hoped: the central region agrees to ~3e-7 and the tails to ~5e-6,
        which is the order of a float32 export's other error.
        """
        _check(
            lambda x: lucid.erfinv(lucid.tanh(x) * 0.9),
            lucid.randn(1, 4, 16, 16),
            tmp_path,
            tol=1e-5,
        )


class TestOperationsTheTraceCannotCarry:
    """Refused by name, because the trace does not hold their operands.

    Not a Core ML limit: `lucid.compile` reads the same trace. Emitting
    from what is recorded would produce a package that runs and answers
    with the wrong values.
    """

    def test_meshgrid_names_itself(self, tmp_path: object) -> None:
        import lucid.nn as nn

        class UsesMeshgrid(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return lucid.meshgrid(x.reshape(-1)[:3], x.reshape(-1)[:4])[0]

        with pytest.raises(cml.UnsupportedOp) as excinfo:
            cml.export(
                UsesMeshgrid().eval(),
                lucid.randn(1, 12),
                f"{tmp_path}/meshgrid.mlpackage",
            )
        assert excinfo.value.op_name == "meshgrid"


class TestTheSamplingGridIsTheOneTheModelAsked:
    """``align_corners`` decides which source coordinate each sample reads.

    It reaches the exported package only if the trace carries it. It did
    not for the three-dimensional resamplers, so a model built with
    ``align_corners=True`` exported as its ``False`` counterpart: a
    package that loads, runs, and returns a plausible volume whose values
    are 23% off. Both settings are checked here, in both ranks, because
    agreeing on the default proves nothing about the flag.
    """

    @pytest.mark.parametrize("align_corners", [False, True], ids=["centres", "corners"])
    @pytest.mark.parametrize(
        ("size", "shape", "mode"),
        [
            ((8, 8), (1, 2, 4, 4), "bilinear"),
            ((6, 8, 8), (1, 2, 3, 4, 4), "trilinear"),
            ((5, 7, 6), (1, 2, 3, 4, 4), "trilinear"),
            ((3, 8, 8), (1, 2, 3, 4, 4), "trilinear"),
        ],
        ids=[
            "bilinear",
            "trilinear-grown",
            "trilinear-fractional",
            "trilinear-same-depth",
        ],
    )
    def test_both_settings_export_to_what_they_mean(
        self,
        size: tuple[int, ...],
        shape: tuple[int, ...],
        mode: str,
        align_corners: bool,
        tmp_path: object,
    ) -> None:
        _check(
            lambda x: F.interpolate(
                x, size=size, mode=mode, align_corners=align_corners
            ),
            lucid.randn(*shape),
            tmp_path,
        )


class TestGridSampleReachesTheExport:
    """It was listed as untranslatable, and the reason was in the tracer.

    ``grid_sample`` returned before ``wire_autograd`` whenever no
    gradient was wanted. Every inference pass takes that path, and an
    export traces one, so the node arrived with no operands at all —
    which reads as "Core ML cannot do this" and was never about Core ML.
    MIL has ``resample``: same operands, same ``(x, y)`` order, same
    normalised range.

    ``mode``, ``padding_mode`` and ``align_corners`` each change the
    answer rather than the layout, so all of them are checked.
    """

    @pytest.mark.parametrize("align_corners", [False, True], ids=["edges", "corners"])
    @pytest.mark.parametrize("padding_mode", ["zeros", "border"])
    @pytest.mark.parametrize("mode", ["bilinear", "nearest"])
    def test_every_setting_exports_to_what_it_means(
        self, mode: str, padding_mode: str, align_corners: bool, tmp_path: object
    ) -> None:
        lucid.manual_seed(1)
        # Coordinates spread past +-1 so the padding mode is actually
        # exercised rather than only the interior.
        grid = lucid.rand(1, 4, 4, 2) * 2.6 - 1.3

        _check(
            lambda x: F.grid_sample(
                x,
                grid,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
            ),
            lucid.randn(1, 3, 5, 5),
            tmp_path,
        )

    def test_reflection_follows_the_engine_and_not_core_ml(
        self, tmp_path: object
    ) -> None:
        """The engine has no true reflection; it clamps, so the export must.

        Emitting Core ML's real reflection would make the package
        disagree with the model it was built from — right by Core ML's
        definition and wrong as a translation.
        """
        lucid.manual_seed(2)
        grid = lucid.rand(1, 4, 4, 2) * 3.0 - 1.5
        _check(
            lambda x: F.grid_sample(x, grid, padding_mode="reflection"),
            lucid.randn(1, 3, 5, 5),
            tmp_path,
        )


class TestTheOpsetTheWriterDeclares:
    """A payload shaped for an older opset than the one declared.

    ``gather_along_axis`` gained ``validate_indices`` in iOS17, which is
    the opset this writer says it is emitting. Leaving the parameter out
    produced a package that failed to *parse* — "Required param
    'validate_indices' is missing" — so nothing in it ran, and every
    model with a gather went down together. ``coatnet`` was one.

    Rank-1 indices along an axis is the shape that reaches
    ``gather_along_axis`` rather than the row-lookup ``gather``.
    """

    def test_gather_along_an_axis_parses_and_matches(self, tmp_path: object) -> None:
        lucid.manual_seed(0)
        index = lucid.tensor([[2, 0, 1, 3], [1, 1, 0, 2]]).to(lucid.int64)
        _check(lambda x: lucid.gather(x, index, dim=1), lucid.randn(2, 4), tmp_path)

    def test_a_row_lookup_still_parses(self, tmp_path: object) -> None:
        """``embedding`` takes the other branch and already had it."""
        lucid.manual_seed(0)
        table = nn.Embedding(4, 8).eval()
        _check(
            lambda ids: table(ids),
            lucid.tensor([[0, 2, 1]]).to(lucid.int64),
            tmp_path,
        )


class TestTheWindowPartitionGetsPastTheRankCap:
    """Core ML caps tensors at rank 5; window partition wants six.

    ``(B, H, W, C)`` becomes ``(B, H/w, w, W/w, w, C)``, is permuted, and
    collapses back — the tall shape exists for exactly two operations.
    Every windowed-attention transformer is built from it, so refusing
    the rank took Swin and MaxViT (22 of the zoo's factories) with it.

    The triple is now staged instead: a reshape is a reinterpretation of
    a contiguous buffer, so a run of adjacent axes can be grouped into
    one, and each stage views the value as ``(left, a, b, right)`` — rank
    four whatever the logical rank is — and swaps the middle pair. The
    tall shape stays bookkeeping and is never asked for.
    """

    def test_a_window_partition_exports(self, tmp_path: object) -> None:
        def partition(x: lucid.Tensor) -> lucid.Tensor:
            y = x.reshape(1, 4, 2, 4, 2, 3)
            y = lucid.permute(y, (0, 1, 3, 2, 4, 5))
            return y.reshape(16, 2, 2, 3)

        lucid.manual_seed(0)
        _check(partition, lucid.randn(1, 8, 8, 3), tmp_path)

    def test_the_merge_back_exports(self, tmp_path: object) -> None:
        def merge(x: lucid.Tensor) -> lucid.Tensor:
            y = x.reshape(1, 4, 4, 2, 2, 3)
            y = lucid.permute(y, (0, 1, 3, 2, 4, 5))
            return y.reshape(1, 8, 8, 3)

        lucid.manual_seed(0)
        _check(merge, lucid.randn(16, 2, 2, 3), tmp_path)

    def test_a_permutation_that_is_not_one_adjacent_swap(
        self, tmp_path: object
    ) -> None:
        """MaxViT's grid partition moves three axes, not two.

        Staging by adjacent swaps covers it without special-casing —
        one stage per inversion — which is why the rewrite matches the
        pattern rather than the model.
        """

        def grid(x: lucid.Tensor) -> lucid.Tensor:
            y = x.reshape(1, 2, 4, 2, 4, 3)
            y = lucid.permute(y, (0, 2, 4, 1, 3, 5))
            return y.reshape(16, 2, 2, 3)

        lucid.manual_seed(0)
        _check(grid, lucid.randn(1, 8, 8, 3), tmp_path)

    def test_a_tall_tensor_that_does_not_collapse_is_still_refused(
        self, tmp_path: object
    ) -> None:
        """The cap is real; only the transient case can be staged.

        A rank-6 value that something other than the triple reads has to
        exist, and Core ML has no way to hold it. Saying so by name beats
        the compiler's parse failure several steps downstream.
        """

        class Tall(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return x.reshape(1, 4, 2, 4, 2, 3).sum(dim=5)

        with pytest.raises(cml.UnsupportedRank):
            cml.export(
                Tall().eval(), lucid.randn(1, 8, 8, 3), f"{tmp_path}/tall.mlpackage"
            )
