"""Rewrite the step-cost budget: ``python -m lucid.test.unit.cost --update``.

Only when a difference :mod:`.test_step_cost` reported is the intended one —
a new op in a step, or one taken out.  The diff of ``step_cost.json`` is
then the record of what changed and by how much.
"""

import argparse
import json
from pathlib import Path

from lucid.test.unit.cost._workloads import measure_all

BUDGET = Path(__file__).with_name("step_cost.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="rewrite step_cost.json")
    args = parser.parse_args()
    measured = measure_all()
    text = json.dumps(measured, indent=2, sort_keys=True) + "\n"
    if args.update:
        BUDGET.write_text(text)
        print(f"wrote {BUDGET} ({len(measured)} workloads)")
    else:
        print(text)


if __name__ == "__main__":
    main()
