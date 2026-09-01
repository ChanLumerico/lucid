"""CLI for the Core ML exporter.

Examples
--------
::

    # zoo model, default 224x224 input
    python -m tools.export_coreml resnet_18_cls --out /tmp/resnet18.mlpackage

    # target the Neural Engine (fp16 is what it wants)
    python -m tools.export_coreml resnet_18_cls --out /tmp/r18.mlpackage \
        --compute-units CPU_AND_NE --precision FLOAT16

    # check the export against the Lucid model it came from
    python -m tools.export_coreml resnet_18_cls --out /tmp/r18.mlpackage --verify
"""

from __future__ import annotations  # tooling only — outside lucid/ (H1 OK)

import argparse
import sys

import numpy as np

import coremltools as ct

import lucid
import lucid.models as models

from tools.export_coreml import export


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.export_coreml",
        description="Export a Lucid model zoo entry to a Core ML package.",
    )
    parser.add_argument("model", help="Registered factory name, e.g. 'resnet_18_cls'.")
    parser.add_argument("--out", required=True, help="Destination .mlpackage path.")
    parser.add_argument(
        "--shape",
        default="1,3,224,224",
        help="Comma-separated input shape (default: 1,3,224,224).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override the head's class count.",
    )
    parser.add_argument(
        "--compute-units",
        default="ALL",
        choices=[u.name for u in ct.ComputeUnit],
        help="Processors Core ML may use (default: ALL). CPU_AND_NE pins to "
        "the Neural Engine.",
    )
    parser.add_argument(
        "--precision",
        default="FLOAT32",
        choices=["FLOAT32", "FLOAT16"],
        help="Exported weight/activation precision (default: FLOAT32, which "
        "stays faithful to the Lucid model; FLOAT16 is what the ANE wants).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run both models on one random input and report the largest "
        "absolute difference.",
    )
    args = parser.parse_args(argv)

    shape = tuple(int(d) for d in args.shape.split(","))
    overrides = {} if args.num_classes is None else {"num_classes": args.num_classes}
    model = models.create_model(args.model, **overrides).eval()
    example = lucid.randn(*shape)

    print(f"[export] {args.model} {shape} → {args.out}")
    mlmodel = export(
        model,
        example,
        args.out,
        compute_units=getattr(ct.ComputeUnit, args.compute_units),
        compute_precision=getattr(ct.precision, args.precision),
    )
    print(f"[export] saved  units={args.compute_units} precision={args.precision}")

    if args.verify:
        reference = model(example)
        want = (
            reference if isinstance(reference, lucid.Tensor) else reference.logits
        ).numpy()
        key = list(mlmodel.input_description)[0]
        got = np.asarray(list(mlmodel.predict({key: example.numpy()}).values())[0])
        diff = float(np.abs(want - got.reshape(want.shape)).max())
        print(f"[verify] max|lucid - coreml| = {diff:.3e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
