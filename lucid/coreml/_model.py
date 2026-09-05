"""The exported model: run it, check it, and see where it actually runs.

``CoreMLModel`` wraps a written ``.mlpackage`` plus the compiled handle
Core ML loaded from it.  Two of its methods exist because of how this
subsystem fails rather than what it does:

* :meth:`verify` compares against the eager model, because an exporter
  that drops a layer still writes a valid package and still returns
  plausible numbers.
* :meth:`compute_plan` asks Core ML which device each operation landed
  on, because asking for the Neural Engine and not getting it is silent.
  A float32 program requested with ``CPU_AND_NE`` reports **zero** ANE
  operations, runs at CPU speed, and warns about nothing.
"""

from typing import TYPE_CHECKING, override

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid.coreml import _spec
from lucid.coreml._build import (
    _apply_image_normalisation,
    _named_examples,
    _select_outputs,
)
from lucid.coreml._spec import Classifier, ComputeUnits, ImageInput

if TYPE_CHECKING:
    from lucid._C.engine import TensorImpl
    from lucid._tensor.tensor import Tensor
    from lucid.nn.module import Module

__all__ = ["CoreMLModel", "PlacementSummary"]

_UNITS = {
    ComputeUnits.ALL: _C_engine.coreml.ComputeUnits.ALL,
    ComputeUnits.CPU_ONLY: _C_engine.coreml.ComputeUnits.CPU_ONLY,
    ComputeUnits.CPU_AND_GPU: _C_engine.coreml.ComputeUnits.CPU_AND_GPU,
    ComputeUnits.CPU_AND_NE: _C_engine.coreml.ComputeUnits.CPU_AND_NE,
}


def _reads_a_palette(path: str) -> bool:
    """Whether the package expands weights through a lookup table.

    Read off the serialized program rather than remembered from the
    build, so a package opened by ``load`` is judged the same way as one
    that has just been written. The operation's type is a plain string in
    the protobuf, so finding it needs no schema — and a false positive
    would only cost a compute unit, not correctness.
    """
    try:
        with open(f"{path}/Data/com.apple.CoreML/model.mlmodel", "rb") as handle:
            return b"constexpr_lut_to_dense" in handle.read()
    except OSError:
        return False


def _palettized_units(units: ComputeUnits, palettized: bool) -> ComputeUnits:
    """The units a palettized model may actually be run on.

    Core ML's GPU path expands ``constexpr_lut_to_dense`` incorrectly for
    the palette sizes 4, 16 and 256 — measured against the package's own
    tables, on macOS 26: a stack of eight 128-channel convolutions comes
    back with 15% error at four bits, while the same package on the CPU
    or the Neural Engine is exact to the last bit. Palette sizes 2, 8 and
    64 are unaffected.

    The failure is silent, and ``ALL`` is the default, so a palettized
    model would otherwise return plausible wrong numbers on the compute
    unit nobody chose. ``ALL`` therefore becomes ``CPU_AND_NE`` — the
    fast path on this hardware anyway — and an explicit request for the
    GPU is refused rather than quietly redirected, because a caller who
    named the GPU is owed an answer about the GPU.
    """
    if not palettized:
        return units
    if units is ComputeUnits.CPU_AND_GPU:
        raise ValueError(
            "lucid.coreml: a palettized model cannot be run on Core ML's GPU "
            "path — it expands the lookup table incorrectly for palette sizes "
            "4, 16 and 256, and does so without reporting an error. Use "
            "ComputeUnits.CPU_AND_NE (the default for these models) or "
            "CPU_ONLY, or export with weights=WeightPrecision.INT8, which the "
            "GPU handles correctly."
        )
    return ComputeUnits.CPU_AND_NE if units is ComputeUnits.ALL else units


class PlacementSummary:
    """Where a model's operations are scheduled.

    ``const`` operations carry no device assignment — they are data, not
    computation — so they are counted separately and kept out of the
    fraction. Reporting 21% ANE for a model whose every computation runs
    on the ANE would be true of the raw operation list and useless.
    """

    def __init__(
        self,
        placements: list[tuple[str, str]],
        *,
        precision: str = "",
        units: ComputeUnits | None = None,
    ) -> None:
        self.placements = placements
        # What the model was built and opened as, so a plan of all-CPU
        # can say whether that was the request or the consequence of one.
        self.precision = precision
        self.units = units
        self.compute: dict[str, int] = {}
        self.constants = 0
        for op, device in placements:
            if device == "unknown":
                self.constants += 1
                continue
            self.compute[device] = self.compute.get(device, 0) + 1

    @property
    def total_compute(self) -> int:
        return sum(self.compute.values())

    @property
    def ane_fraction(self) -> float:
        """Share of computation the Neural Engine takes, 0.0 to 1.0.

        ``0.0`` on a model asked to use the ANE is the signal that the
        request did not take — most often because the program is float32,
        which the Neural Engine does not run.
        """
        total = self.total_compute
        return 0.0 if total == 0 else self.compute.get("ANE", 0) / total

    @property
    def note(self) -> str:
        """Why the Neural Engine took none of the work, when it took none.

        A float32 program cannot run on the Neural Engine at all — it is
        a float16 device — so an export that keeps the default precision
        lands entirely on the CPU however the compute units were asked
        for. Measured on a ResNet-18: 66 of 66 operations on the CPU at
        float32, 66 of 68 on the Neural Engine at float16.

        Nothing about that is an error, and Core ML reports no problem,
        so the only place it can surface is here — where somebody is
        already asking where the work went.
        """
        wanted = self.units in (ComputeUnits.ALL, ComputeUnits.CPU_AND_NE)
        if not wanted or self.total_compute == 0 or self.ane_fraction > 0.0:
            return ""
        if self.precision.upper() != "FLOAT32":
            return ""
        return (
            "no operation reached the Neural Engine because the program is "
            "float32, which that device does not run — export with "
            "precision=Precision.FLOAT16 to reach it"
        )

    @override
    def __repr__(self) -> str:
        parts = ", ".join(f"{d}={n}" for d, n in sorted(self.compute.items()))
        summary = (
            f"PlacementSummary({parts}, constants={self.constants}, "
            f"ane={self.ane_fraction:.0%})"
        )
        return f"{summary} — {self.note}" if self.note else summary


class CoreMLModel:
    """A Core ML package written by Lucid, loaded and ready to run.

    Holds the ``.mlpackage`` on disk plus the compiled model Core ML
    produced from it. Compilation happens once, when the handle is
    created, because it is the expensive step — hundreds of milliseconds
    for a real network — and every prediction afterwards reuses it.
    """

    def __init__(
        self,
        path: str,
        input_names: list[str],
        output_names: list[str],
        *,
        compute_units: ComputeUnits = ComputeUnits.ALL,
        precision: str = "FLOAT32",
        output_shapes: dict[str, tuple[int, ...]] | None = None,
        image_input: ImageInput | None = None,
        classifier: Classifier | None = None,
        function_name: str = "",
    ) -> None:
        self.path = path
        self.input_names = input_names
        self.output_names = output_names
        # A palettized package computes the wrong answer on Core ML's GPU
        # path — see ``_palettized_units`` — so the units it is opened
        # with are not always the units that were asked for.
        self.palettized = _reads_a_palette(path)
        self.compute_units = _palettized_units(compute_units, self.palettized)
        compute_units = self.compute_units
        self.precision = precision
        # Core ML's multi-array has no rank-0 form, so a model whose output
        # is a scalar comes back shaped (1,).  Keeping the traced shapes
        # lets ``predict`` hand back what the eager model would.
        self.output_shapes = output_shapes or {}
        # An image input changes two things: Core ML refuses a multi-array
        # for it, so the tensor has to reach the runtime as a pixel buffer;
        # and the normalisation now lives inside the package, so a
        # comparison against the eager model has to apply it on that side.
        self.image_input = image_input
        # A classifier returns a string and a dictionary, not arrays, so
        # it is read through ``classify`` rather than ``predict``.
        self.classifier = classifier
        # Empty takes whichever entry point the package names as default.
        self.function_name = function_name
        self._handle = _C_engine.coreml.load_model(
            path, _UNITS[compute_units], function_name
        )

    def _feed(self, x: object) -> list[tuple[str, TensorImpl]]:
        """Pair each input feature with its tensor.

        Accepts the same three shapes ``export`` did — a lone tensor, a
        tuple in the model's argument order, or a mapping — so a caller
        drives the package the way they built it.
        """
        if isinstance(x, lucid.Tensor):
            given: list[tuple[str, Tensor]] = [(self.input_names[0], x)]
        elif isinstance(x, dict):
            given = list(x.items())
        elif isinstance(x, (tuple, list)):
            given = list(zip(self.input_names, x))
        else:
            raise TypeError(
                f"lucid.coreml: expected a Tensor, a tuple, or a mapping — got "
                f"{type(x).__name__}"
            )
        if len(given) != len(self.input_names):
            raise ValueError(
                f"lucid.coreml: this package takes {len(self.input_names)} input(s) "
                f"{self.input_names}, and {len(given)} were given"
            )

        fed: list[tuple[str, TensorImpl]] = []
        for name, tensor in given:
            if name not in self.input_names:
                raise KeyError(
                    f"lucid.coreml: {name!r} is not an input of this package "
                    f"{self.input_names}"
                )
            if tensor.dtype in (lucid.int64, lucid.int32):
                # Core ML's multi-array has int32 and no int64, so an
                # integer input is narrowed here rather than at every call
                # site. Token ids and masks are nowhere near the range
                # where that loses anything.
                tensor = (
                    tensor if tensor.dtype == lucid.int32 else tensor.to(lucid.int32)
                )
            fed.append((name, tensor._impl))
        return fed

    @property
    def carries_state(self) -> bool:
        """Whether the package keeps values between predictions."""
        return bool(self._handle.carries_state)

    def reset_state(self) -> None:
        """Forget everything the package has accumulated.

        A state persists across predictions by design, so starting a fresh
        sequence has to be asked for; there is no other way back to the
        value it began at.

        Raises
        ------
        ValueError
            The package carries no state.
        """
        self._handle.reset_state()

    def _images(self) -> list[tuple[str, int]]:
        if self.image_input is None:
            return []
        return [(self.input_names[0], _spec.color_space(self.image_input.color))]

    def classify(self, x: object) -> tuple[str, dict[str, float]]:
        """Run a classifier package and read back what it names.

        Parameters
        ----------
        x : Tensor or tuple of Tensor or dict of str to Tensor
            Input, in the same shapes :meth:`predict` accepts.

        Returns
        -------
        tuple[str, dict[str, float]]
            The winning label, and every label with its probability.

        Raises
        ------
        TypeError
            The package was not exported with a classifier.
        """
        if self.classifier is None:
            raise TypeError(
                "lucid.coreml: this package returns scores, not labels — export it "
                "with classifier=Classifier(labels=...) to get labels"
            )
        label, scores = self._handle.classify(
            self._feed(x),
            self._images(),
            self.classifier.label_name,
            self.classifier.probabilities_name,
        )
        return label, {name: float(value) for name, value in scores}

    def predict(self, x: object) -> Tensor | dict[str, Tensor]:
        """Run the model.

        Inputs must be host tensors: Core ML reads host memory, and moving
        a Metal tensor here would hide a copy the caller did not ask for.
        Move it explicitly with ``.to("cpu")``.

        Parameters
        ----------
        x : Tensor or tuple of Tensor or dict of str to Tensor
            One tensor for a single-input package; otherwise a tuple in
            the package's input order, or a mapping by feature name.

        Returns
        -------
        Tensor or dict[str, Tensor]
            The output for a single-output package; otherwise every
            output, keyed by the field the model declared it as.
        """
        if self.classifier is not None:
            raise TypeError(
                "lucid.coreml: this package returns a label and a probability map, "
                "not arrays — use classify()"
            )
        raw = self._handle.predict(self._feed(x), self.output_names, self._images())
        produced: dict[str, Tensor] = {}
        for name, impl in zip(self.output_names, raw):
            out = _wrap(impl)
            declared = self.output_shapes.get(name)
            if declared is not None and out.shape != declared:
                out = out.reshape(*declared)
            produced[name] = out
        if len(self.output_names) == 1:
            return produced[self.output_names[0]]
        return produced

    def verify(self, model: Module, x: object, *, relative: bool = False) -> float:
        """Largest difference against the eager model.

        Shapes agreeing is not evidence: a package missing a layer has
        the right shape and returns plausible numbers. This runs both and
        compares values — every output, not just the first, since a
        detector that exported its class scores and dropped its boxes
        would otherwise pass.

        The default is an **absolute** difference, which is only
        interpretable against outputs of a known size. A model whose
        outputs differ in magnitude makes that trap easy to fall into:
        RealNVP returns a latent of order 1 beside a log-probability of
        order 1e4, so the absolute worst is set by the second, and
        dividing it by the first reads as a 4% error when every output
        agrees to 1e-6.

        ``relative=True`` scales each output's difference by that
        output's own magnitude — but only down to one, which is the
        other half of the same trap: VQ-VAE's latent has values around
        2e-3, and dividing float32 noise by that reads as 3e-4 when the
        difference is 5e-7. Dividing by ``max(scale, 1)`` is relative
        where relative means something and absolute where it does not,
        which is the same bargain a tolerance pair makes.

        Parameters
        ----------
        model : nn.Module
            The eager model this package was exported from.
        x : Tensor or tuple of Tensor or dict of str to Tensor
            Input to run through both. Host tensors.
        relative : bool, optional, default=False
            Scale each output's difference by that output's own largest
            magnitude, floored at one, before taking the worst.

        Returns
        -------
        float
            The worst ``max|coreml - eager|`` across the outputs, or the
            worst of those divided by each output's own scale when
            ``relative``. Expect ~1e-7 for a float32 export and ~1e-3
            relative for float16.

        Notes
        -----
        For an image export the pixel buffer is eight bits per channel,
        so anything but whole numbers in ``[0, 255]`` is rounded on the
        way in and the two sides see different pixels. That is refused
        rather than reported: for ``randn`` values the rounding is most
        of the signal and the answer would be around 3e-1, which reads
        as a broken export. Feed pixels and the comparison is the usual
        one.
        """
        if self.classifier is not None:
            raise TypeError(
                "lucid.coreml: a classifier's output is a label and a probability "
                "map; compare them with classify() rather than verify()"
            )
        if self.carries_state:
            raise TypeError(
                "lucid.coreml: this package carries state, so one prediction says "
                "nothing about whether it agrees — the eager model would have to "
                "be threaded through the same sequence. Run both over several "
                "steps and compare, with reset_state() between runs"
            )
        examples, by_keyword = _named_examples(x)
        if self.image_input is not None:
            # A pixel buffer is eight bits per channel, so an input that
            # is not already whole numbers in [0, 255] is rounded on the
            # way in and the two sides genuinely see different pixels.
            # For ``randn`` values that is most of the signal, and the
            # number this would return — around 3e-1 — reads as a broken
            # export rather than as the round-trip it measures.
            for name, tensor in examples:
                low = float(tensor.min().item())
                high = float(tensor.max().item())
                integral = bool(((tensor - tensor.round()).abs().max() < 1e-6).item())
                if low < 0.0 or high > 255.0 or not integral:
                    raise TypeError(
                        f"lucid.coreml: {name!r} is not pixel data (range "
                        f"[{low:.3g}, {high:.3g}], "
                        f"{'non-integral' if not integral else 'integral'}), and this "
                        "package takes an image. Core ML would quantise it to eight "
                        "bits per channel, so the comparison would measure that "
                        "rounding rather than the network. Feed whole numbers in "
                        "[0, 255] to compare the two models."
                    )
            # The package normalises the pixels itself, so the eager model
            # has to be shown the same normalised values or the comparison
            # is between two different inputs.
            examples = [
                (name, _apply_image_normalisation(tensor, self.image_input))
                for name, tensor in examples
            ]
        if by_keyword:
            reference = model(**dict(examples))
        else:
            reference = model(*(tensor for _, tensor in examples))
        expected = dict(_select_outputs(reference, None))

        got = self.predict(x)
        produced = got if isinstance(got, dict) else {self.output_names[0]: got}

        worst = 0.0
        carries_signal = False
        for name in self.output_names:
            wanted = expected.get(name)
            if wanted is None:
                raise KeyError(
                    f"lucid.coreml: the eager model no longer returns {name!r}, "
                    "which this package exports"
                )
            extreme = float(wanted.abs().max().item())
            if extreme != extreme:  # NaN: the reference has no finite answer
                raise ValueError(
                    f"lucid.coreml: the eager model's {name!r} contains NaN, so "
                    "there is nothing to compare the package against — the model "
                    "produces it before any export is involved, and a difference "
                    "against NaN is NaN whatever the package computed"
                )
            carries_signal = carries_signal or extreme > 0.0
            gap = float((produced[name] - wanted).abs().max().item())
            worst = max(worst, gap / max(extreme, 1.0) if relative else gap)
        if not carries_signal:
            # Comparing against an all-zero reference proves nothing: an
            # exporter that dropped every layer would also return zeros
            # and score perfectly. Several zoo models zero-initialise
            # their head, so this is reachable with an untrained factory
            # rather than being a theoretical case.
            #
            # Only when *every* output is zero, though. A model can have
            # one output that is legitimately zero and others that are
            # not — NICE's log-determinant is exactly zero because the
            # transform preserves volume, which is a fact about the
            # architecture and not a missing weight. Refusing the whole
            # comparison for it would hide the outputs that do carry
            # signal, and a zero reference still catches an export that
            # returns something else.
            raise ValueError(
                f"lucid.coreml: every output of the eager model "
                f"({', '.join(self.output_names)}) is all zeros, so this "
                "comparison cannot detect anything — load weights, or perturb "
                "the zero-initialised parameters, before verifying"
            )
        return worst

    def compute_plan(self) -> PlacementSummary:
        """Which device Core ML assigns each operation to.

        Requires macOS 14.4+; an empty plan there means *unknown*, not
        *unaccelerated*.
        """
        placements = _C_engine.coreml.compute_plan(
            self.path, _UNITS[self.compute_units]
        )
        return PlacementSummary(
            [(op, device) for op, device in placements],
            precision=self.precision,
            units=self.compute_units,
        )

    def close(self) -> None:
        """Release the compiled model and the artifacts Core ML cached."""
        self._handle.close()

    @override
    def __repr__(self) -> str:
        return (
            f"CoreMLModel({self.path!r}, precision={self.precision}, "
            f"units={self.compute_units.value})"
        )
