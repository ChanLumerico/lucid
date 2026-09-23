"""Eager against numpy, on CPU and on Metal, for every case with a reference.

The replay matrices compare compiled with eager, so they cannot see an op
eager gets wrong on both devices alike.  This one holds eager to an answer
computed independently — see :mod:`._numpy_oracle` for the table and for
the cases it deliberately leaves out.

Two rules keep numpy's own conventions from being mistaken for Lucid's
defects: values are compared after casting (numpy's ``floor`` of an integer
is float64, the reference framework's is integer), and the dtype *kind* is
enforced only for :data:`REAL_VALUED` ops — an integer input to ``arcsin``
must come back floating point, whatever width.

H4: numpy computes expected values here and nowhere else; Lucid values reach
it only through ``Tensor.numpy()``.
"""

import numpy as np
import pytest

import lucid
import lucid.test.unit.compile._op_matrix as M

from lucid.test.unit.compile._helpers import COMPILE_DEVICE
from lucid.test.unit.ops._numpy_oracle import NO_REF, REAL_VALUED, REFS

#: Compared bit for bit: a one-ULP step disappears inside any float tolerance.
EXACT = frozenset({"nextafter"})


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


DEVICES = ["cpu"] + (["metal"] if _metal_ok() else [])


class _Consts:
    """The case's constants, as numpy arrays, on the device under test."""

    def w(self, *shape: int, seed: int = 7) -> np.ndarray:
        return M._w(*shape, seed=seed).numpy()

    def idx(self, *vals: int) -> np.ndarray:
        return M._idx(*vals).numpy()

    def eye(self, n: int) -> np.ndarray:
        return M._eye(n).numpy()


def _leaves(out: object) -> list[object]:
    if isinstance(out, (tuple, list)):
        return [leaf for o in out for leaf in _leaves(o)]
    return [out]


def _params() -> list[tuple[str, str, str]]:
    return [
        (c.name, d, dev)
        for c in M.CASES
        if c.name in REFS and not c.random
        for d in c.dtypes
        for dev in DEVICES
    ]


def test_every_case_is_accounted_for() -> None:
    """A case is either held to a reference or listed with the reason it is not."""
    unaccounted = [
        c.name for c in M.CASES if c.name not in REFS and c.name not in NO_REF
    ]
    assert (
        not unaccounted
    ), f"cases with neither a reference nor a reason: {unaccounted}"


@pytest.mark.parametrize(("name", "dtype", "device"), _params(), ids=str)
def test_eager_matches_numpy(name: str, dtype: str, device: str) -> None:
    case = M.CASE_BY_NAME[name]
    x = M.make_input(case.kind, dtype, case.shape, 2)
    with M.on_device(device):
        x = x.to(device)
        try:
            got = case.fn(x)
        except Exception as e:  # noqa: BLE001 — eager decides which dtypes exist
            if dtype == "f32":
                pytest.fail(f"eager float32 raised {type(e).__name__}: {e}")
            pytest.skip(f"eager rejects {dtype}: {type(e).__name__}")
        try:
            want = REFS[name](x.numpy(), _Consts())
        except Exception as e:  # noqa: BLE001 — numpy has no answer for this input
            pytest.skip(f"numpy has no answer: {type(e).__name__}: {e}")

    g_leaves = [t for t in _leaves(got) if isinstance(t, lucid.Tensor)]
    w_leaves = [np.asarray(w) for w in _leaves(want)]
    assert len(g_leaves) == len(
        w_leaves
    ), f"{len(g_leaves)} outputs, want {len(w_leaves)}"
    for i, (g, w) in enumerate(zip(g_leaves, w_leaves)):
        gn = g.numpy()
        assert gn.shape == w.shape, f"out[{i}] shape {gn.shape}, want {w.shape}"
        if name in REAL_VALUED and dtype != "f32":
            assert np.issubdtype(
                gn.dtype, np.floating
            ), f"out[{i}] is {gn.dtype}: {name} of an integer cannot be an integer"
        if name in EXACT:
            assert np.array_equal(
                gn.view(np.int32), w.astype(gn.dtype).view(np.int32)
            ), f"out[{i}] not bit-exact: {gn.ravel()[:6]} vs {w.ravel()[:6]}"
        elif np.issubdtype(w.dtype, np.floating) or np.issubdtype(
            gn.dtype, np.floating
        ):
            assert np.allclose(
                gn.astype(np.float64),
                w.astype(np.float64),
                rtol=2e-4,
                atol=2e-5,
                equal_nan=True,
            ), (
                f"out[{i}] max|diff| "
                f"{np.nanmax(np.abs(gn.astype(np.float64) - w.astype(np.float64))):.3g}: "
                f"{gn.ravel()[:6].tolist()} vs {w.ravel()[:6].tolist()}"
            )
        else:
            assert np.array_equal(
                gn.astype(np.int64), w.astype(np.int64)
            ), f"out[{i}] differs: {gn.ravel()[:6].tolist()} vs {w.ravel()[:6].tolist()}"
