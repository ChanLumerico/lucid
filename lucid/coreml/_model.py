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
from lucid.coreml._build import _select_output
from lucid.coreml._spec import ComputeUnits

if TYPE_CHECKING:
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
        input_name: str,
        output_name: str,
        *,
        compute_units: ComputeUnits = ComputeUnits.ALL,
        precision: str = "FLOAT32",
        output_shape: tuple[int, ...] | None = None,
    ) -> None:
        self.path = path
        self.input_name = input_name
        self.output_name = output_name
        self.compute_units = compute_units
        self.precision = precision
        # Core ML's multi-array has no rank-0 form, so a model whose output
        # is a scalar comes back shaped (1,).  Keeping the traced shape lets
        # ``predict`` hand back what the eager model would.
        self.output_shape = output_shape
        self._handle = _C_engine.coreml.load_model(path, _UNITS[compute_units])

    def predict(self, x: Tensor) -> Tensor:
        """Run the model on ``x``.

        The input must be a host tensor: Core ML reads host memory, and
        moving a Metal tensor here would hide a copy the caller did not
        ask for. Move it explicitly with ``.to("cpu")``.
        """
        if x.dtype in (lucid.int64, lucid.int32):
            # Core ML's multi-array has int32 and no int64, so an integer
            # input is narrowed here rather than at every call site. Token
            # ids and masks are nowhere near the range where that loses
            # anything.
            x = x if x.dtype == lucid.int32 else x.to(lucid.int32)
        out = _wrap(self._handle.predict(self.input_name, x._impl, self.output_name))
        if self.output_shape is not None and out.shape != self.output_shape:
            out = out.reshape(*self.output_shape)
        return out

    def verify(self, model: Module, x: Tensor) -> float:
        """Largest absolute difference against the eager model.

        Shapes agreeing is not evidence: a package missing a layer has
        the right shape and returns plausible numbers. This runs both and
        compares values.

        Parameters
        ----------
        model : nn.Module
            The eager model this package was exported from.
        x : Tensor
            Input to run through both. A host tensor.

        Returns
        -------
        float
            ``max|coreml - eager|``. Expect ~1e-7 for a float32 export
            and ~1e-3 relative for float16.
        """
        reference = _select_output(model(x), None)
        scale = float(reference.abs().max().item())
        if scale == 0.0:
            # Comparing against an all-zero reference proves nothing: an
            # exporter that dropped every layer would also return zeros
            # and score perfectly.  Several zoo models zero-initialise
            # their head, so this is reachable with an untrained factory
            # rather than being a theoretical case.
            raise ValueError(
                "lucid.coreml: the eager model returned all zeros, so this "
                "comparison cannot detect anything — load weights, or perturb "
                "the zero-initialised parameters, before verifying"
            )
        return float((self.predict(x) - reference).abs().max().item())

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
