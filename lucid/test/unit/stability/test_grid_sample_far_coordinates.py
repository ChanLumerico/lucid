"""``grid_sample`` at coordinates far outside the image, NaN and infinite ones.

A grid coordinate of ``inf`` or ``1e10`` killed the interpreter with SIGBUS
on the CPU (LCD-420): the kernel cast the floored position to ``int``,
which saturated, added one, which overflowed, and the bounds check then let
a wild address through.  Sample positions now become indices by one rule
(``resample::index_of`` on the CPU, ``gpu_index_of`` on Metal): NaN reads
as 0 and anything past the int64 range saturates, which is how the
reference reads them too.

Every case runs in a child interpreter, as in ``test_no_process_crash.py``,
so a crash fails one case instead of the run.  The child reports the
sampled value and both gradients; the parent holds them to the reference.
"""

import json
import math
import os
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.stability

_COORDS = {
    "inf": math.inf,
    "-inf": -math.inf,
    "1e10": 1e10,
    "-1e10": -1e10,
    "nan": math.nan,
    "2^31": 2.0**31,
    "-2^31": -(2.0**31),
}
_MODES = ["bilinear", "nearest", "bicubic"]
_PADDINGS = ["zeros", "border", "reflection"]


def _cases() -> list[tuple[str, str, str, bool, str, int]]:
    """(id, mode, padding, align_corners, coordinate, axis it is on)."""
    out = []
    for mode in _MODES:
        for pad in _PADDINGS:
            for align in (False, True):
                for coord in _COORDS:
                    for axis in (0, 1):
                        name = f"{mode}-{pad}-{'corners' if align else 'edges'}-{coord}-{'xy'[axis]}"
                        out.append((name, mode, pad, align, coord, axis))
    return out


CASES = _cases()
_IMAGE = np.arange(16.0, dtype=np.float32).reshape(1, 1, 4, 4)


def _grid(coord: str, axis: int) -> np.ndarray:
    point = [0.25, 0.25]
    point[axis] = _COORDS[coord]
    return np.array([[[point]]], dtype=np.float32)


_CHILD = """
import json, math, sys
import numpy as np
import lucid
import lucid.nn.functional as F

D = sys.argv[1]
image = np.arange(16.0, dtype=np.float32).reshape(1, 1, 4, 4)

def listed(t):
    return [None if math.isnan(v) else v for v in np.asarray(t.to("cpu").numpy(), dtype=np.float64).ravel().tolist()]

for name, mode, pad, align, grid in json.loads(sys.stdin.read()):
    print("@@BEGIN " + name, flush=True)
    x = lucid.tensor(image, device=D, requires_grad=True)
    g = lucid.tensor(np.array(grid, dtype=np.float32), device=D, requires_grad=True)
    y = F.grid_sample(x, g, mode=mode, padding_mode=pad, align_corners=align)
    y.sum().backward()
    out = {"y": listed(y), "dx": listed(x.grad), "dg": listed(g.grad)}
    print("@@END " + name + " " + json.dumps(out), flush=True)
"""


def _payload() -> list[Any]:
    return [
        [name, mode, pad, align, _grid(coord, axis).tolist()]
        for name, mode, pad, align, coord, axis in CASES
    ]


def _run(device: str) -> dict[str, Any]:
    """Every case on ``device``; a child that dies is blamed on its case and
    restarted past it."""
    results: dict[str, Any] = {}
    pending = _payload()
    env = dict(os.environ, PYTHONWARNINGS="ignore")
    while pending:
        done = subprocess.run(
            [sys.executable, "-c", _CHILD, device],
            input=json.dumps(pending),
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
        current = None
        for line in done.stdout.splitlines():
            if line.startswith("@@BEGIN "):
                current = line[len("@@BEGIN ") :]
            elif line.startswith("@@END "):
                name, _, body = line[len("@@END ") :].partition(" ")
                results[name] = json.loads(body)
                current = None
        if done.returncode == 0:
            break
        blamed = current if current is not None else pending[0][0]
        results[blamed] = {"crash": f"exit {done.returncode}: {done.stderr[-400:]}"}
        pending = [case for case in pending if case[0] not in results]
    return results


_RESULTS: dict[str, dict[str, Any]] = {}


def _result(device: str, name: str) -> dict[str, Any]:
    if device not in _RESULTS:
        _RESULTS[device] = _run(device)
    return _RESULTS[device].get(name, {"crash": "the child never reached this case"})


def _array(values: list[float | None]) -> np.ndarray:
    return np.array([math.nan if v is None else v for v in values], dtype=np.float64)


def _reference(ref: Any, mode: str, pad: str, align: bool, grid: np.ndarray) -> dict:
    x = ref.tensor(_IMAGE, requires_grad=True)
    g = ref.tensor(grid, requires_grad=True)
    y = ref.nn.functional.grid_sample(
        x, g, mode=mode, padding_mode=pad, align_corners=align
    )
    y.sum().backward()
    return {
        "y": y.detach().double().numpy().ravel(),
        "dx": x.grad.double().numpy().ravel(),
        "dg": g.grad.double().numpy().ravel(),
    }


def _compared(mode: str, pad: str, coord: str) -> tuple[str, ...]:
    """Which results are held to the reference.

    The Python composites (bicubic, and the grid reflection) differ from the
    reference on non-finite coordinates in their gradients; those are pinned
    by the strict xfails below, not here.
    """
    if mode == "bicubic":
        return ("y",)
    if pad == "reflection" and coord == "nan":
        return ("y", "dx")
    return ("y", "dx", "dg")


@pytest.mark.parametrize(
    "name,mode,pad,align,coord,axis", CASES, ids=[case[0] for case in CASES]
)
def test_far_coordinates(
    ref: Any,
    device: str,
    name: str,
    mode: str,
    pad: str,
    align: bool,
    coord: str,
    axis: int,
) -> None:
    got = _result(device, name)
    assert (
        "crash" not in got
    ), f"{name} on {device} killed the interpreter: {got['crash']}"
    want = _reference(ref, mode, pad, align, _grid(coord, axis))
    for key in _compared(mode, pad, coord):
        np.testing.assert_allclose(
            _array(got[key]), want[key], atol=1e-5, equal_nan=True, err_msg=key
        )


@pytest.mark.xfail(
    strict=True,
    reason="LCD-420 follow-up: reflect() in lucid/nn/functional/_grid_sampling.py "
    "has no grid gradient at a NaN coordinate; the reference passes one through",
)
def test_reflection_grid_gradient_at_nan(ref: Any, device: str) -> None:
    name = "bilinear-reflection-edges-nan-x"
    got = _result(device, name)
    want = _reference(ref, "bilinear", "reflection", False, _grid("nan", 0))
    np.testing.assert_allclose(_array(got["dg"]), want["dg"], atol=1e-5, equal_nan=True)


@pytest.mark.xfail(
    strict=True,
    reason="LCD-420 follow-up: the bicubic composite scatters NaN into the input "
    "gradient at an infinite coordinate under zero padding; the reference gives 0",
)
def test_bicubic_input_gradient_at_infinity(ref: Any, device: str) -> None:
    name = "bicubic-zeros-edges-inf-y"
    got = _result(device, name)
    want = _reference(ref, "bicubic", "zeros", False, _grid("inf", 1))
    np.testing.assert_allclose(_array(got["dx"]), want["dx"], atol=1e-5, equal_nan=True)


def test_case_names_are_unique() -> None:
    names = [case[0] for case in CASES]
    assert len(names) == len(set(names))
