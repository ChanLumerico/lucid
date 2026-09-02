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
from lucid.coreml._spec import ComputeUnits, ImageInput

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


class PlacementSummary:
    """Where a model's operations are scheduled.

    ``const`` operations carry no device assignment — they are data, not
    computation — so they are counted separately and kept out of the
    fraction. Reporting 21% ANE for a model whose every computation runs
    on the ANE would be true of the raw operation list and useless.
    """

    def __init__(self, placements: list[tuple[str, str]]) -> None:
        self.placements = placements
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

    @override
    def __repr__(self) -> str:
        parts = ", ".join(f"{d}={n}" for d, n in sorted(self.compute.items()))
        return (
            f"PlacementSummary({parts}, constants={self.constants}, "
            f"ane={self.ane_fraction:.0%})"
        )


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
    ) -> None:
        self.path = path
        self.input_names = input_names
        self.output_names = output_names
        self.compute_units = compute_units
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
        self._handle = _C_engine.coreml.load_model(path, _UNITS[compute_units])

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
        images = (
            [(self.input_names[0], _spec.color_space(self.image_input.color))]
            if self.image_input is not None
            else []
        )
        raw = self._handle.predict(self._feed(x), self.output_names, images)
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

    def verify(self, model: Module, x: object) -> float:
        """Largest relative difference against the eager model.

        Shapes agreeing is not evidence: a package missing a layer has
        the right shape and returns plausible numbers. This runs both and
        compares values — every output, not just the first, since a
        detector that exported its class scores and dropped its boxes
        would otherwise pass.

        Parameters
        ----------
        model : nn.Module
            The eager model this package was exported from.
        x : Tensor or tuple of Tensor or dict of str to Tensor
            Input to run through both. Host tensors.

        Returns
        -------
        float
            The worst ``max|coreml - eager|`` across the outputs. Expect
            ~1e-7 for a float32 export and ~1e-3 relative for float16.

        Notes
        -----
        For an image export the pixel buffer is eight bits per channel,
        so a non-integral input is rounded on the way in and the two
        sides see slightly different pixels — around 1e-5 rather than
        1e-8 for the same model. Feed integral values to compare the
        network rather than the rounding.
        """
        examples, by_keyword = _named_examples(x)
        if self.image_input is not None:
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
        for name in self.output_names:
            wanted = expected.get(name)
            if wanted is None:
                raise KeyError(
                    f"lucid.coreml: the eager model no longer returns {name!r}, "
                    "which this package exports"
                )
            scale = float(wanted.abs().max().item())
            if scale == 0.0:
                # Comparing against an all-zero reference proves nothing:
                # an exporter that dropped every layer would also return
                # zeros and score perfectly.  Several zoo models
                # zero-initialise their head, so this is reachable with an
                # untrained factory rather than being a theoretical case.
                raise ValueError(
                    f"lucid.coreml: the eager model's {name!r} is all zeros, so "
                    "this comparison cannot detect anything — load weights, or "
                    "perturb the zero-initialised parameters, before verifying"
                )
            worst = max(worst, float((produced[name] - wanted).abs().max().item()))
        return worst

    def compute_plan(self) -> PlacementSummary:
        """Which device Core ML assigns each operation to.

        Requires macOS 14.4+; an empty plan there means *unknown*, not
        *unaccelerated*.
        """
        placements = _C_engine.coreml.compute_plan(
            self.path, _UNITS[self.compute_units]
        )
        return PlacementSummary([(op, device) for op, device in placements])

    def close(self) -> None:
        """Release the compiled model and the artifacts Core ML cached."""
        self._handle.close()

    @override
    def __repr__(self) -> str:
        return (
            f"CoreMLModel({self.path!r}, precision={self.precision}, "
            f"units={self.compute_units.value})"
        )
