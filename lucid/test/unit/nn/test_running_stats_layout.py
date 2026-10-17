"""BatchNorm writes its running statistics into their own elements, wherever they sit.

The running-statistic update — fused into the training forward, and the
``batch_norm_update_running_stats`` entry — took each buffer's first C
elements from byte 0 as its values.  A running statistic that is a view at an
offset or through other strides (``bn.running_mean = base[:, 0]``) had the
first elements of its buffer written instead of its own: the base's other
elements were overwritten and the view's were left as they were.  On Metal a
view of part of a tensor was refused (LCD-491).  The optimizer step had the
same defect (LCD-476).  Both writers now go through one helper,
``lucid/_C/core/ElementWrite.h``.

The inputs differ per channel and per element, so a statistic written to the
wrong channel or element ends up with the wrong value.  Metal views of part
of a tensor are built with the engine's ``_make_view`` hook: on Metal, Python
indexing still copies (LCD-18).
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid.test._fixtures.devices import device_dtype_params

_C = 3
_BASE_SHAPE = (4, 6)
_STEPS = 3
_MOMENTUM = 0.3
_TOL = 1e-5

# Each BatchNorm, by its class name in either framework, with an input shape
# it takes.
_NORMS: dict[str, tuple[str, tuple[int, ...]]] = {
    "1d_flat": ("BatchNorm1d", (6, _C)),
    "1d": ("BatchNorm1d", (5, _C, 4)),
    "2d": ("BatchNorm2d", (4, _C, 2, 3)),
    "3d": ("BatchNorm3d", (2, _C, 2, 2, 3)),
}

# A (C,) statistic's (shape, stride, offset) over the 4 x 6 base, in elements.
_LAYOUTS: dict[str, tuple[tuple[int, ...], tuple[int, ...], int]] = {
    "column": ((_C,), (6,), 2),
    "diagonal": ((_C,), (7,), 0),
    "offset_run": ((_C,), (1,), 7),
    # Contiguous from the buffer's first byte, but only the start of it.
    "leading_run": ((_C,), (1,), 0),
}

_WHICH = ["running_mean", "running_var", "both"]


def _view(
    base: lucid.Tensor, shape: tuple[int, ...], stride: tuple[int, ...], offset: int
) -> lucid.Tensor:
    return _wrap(_C_engine.TensorImpl._make_view(base._impl, list(shape), list(stride), offset))


def _positions(geometry: tuple[tuple[int, ...], tuple[int, ...], int]) -> list[int]:
    """The flat base index of each element of the view, in row-major order."""
    shape, stride, offset = geometry
    return [
        offset + sum(i * s for i, s in zip(index, stride, strict=True))
        for index in np.ndindex(*shape)
    ]


def _covered(*layouts: tuple[tuple[int, ...], tuple[int, ...], int]) -> np.ndarray:
    """Which elements of the base the layouts read."""
    mask = np.zeros(int(np.prod(_BASE_SHAPE)), dtype=bool)
    for geometry in layouts:
        mask[_positions(geometry)] = True
    return mask.reshape(_BASE_SHAPE)


def _base_init(seed: int) -> np.ndarray:
    # Positive, so it reads as a variance too.
    rng = np.random.default_rng(seed)
    return rng.uniform(0.5, 1.5, _BASE_SHAPE).astype(np.float32)


def _input(step: int, shape: tuple[int, ...]) -> np.ndarray:
    rng = np.random.default_rng(100 + step)
    channel = [1] * len(shape)
    channel[1] = _C
    scale = np.array([0.5, 1.0, 2.0], dtype=np.float32).reshape(channel)
    shift = np.array([-1.0, 0.5, 3.0], dtype=np.float32).reshape(channel)
    return (rng.standard_normal(shape).astype(np.float32) * scale + shift).astype(np.float32)


def _flat(t: lucid.Tensor) -> np.ndarray:
    return np.asarray(t.detach().numpy(), dtype=np.float64)


def _stats(which: str) -> list[str]:
    return ["running_mean", "running_var"] if which == "both" else [which]


@pytest.mark.parametrize("norm", list(_NORMS))
@pytest.mark.parametrize("layout", list(_LAYOUTS))
@pytest.mark.parametrize("which", _WHICH)
def test_a_running_stat_view_is_updated_as_the_reference_updates_it(
    norm: str, layout: str, which: str, device: str, ref: Any
) -> None:
    name, x_shape = _NORMS[norm]
    geometry = _LAYOUTS[layout]
    bn = getattr(nn, name)(_C, momentum=_MOMENTUM).to(device)
    ref_bn = getattr(ref.nn, name)(_C, momentum=_MOMENTUM)
    inits, bases, ref_bases = {}, {}, {}
    for k, stat in enumerate(_stats(which)):
        inits[stat] = _base_init(k)
        bases[stat] = lucid.tensor(inits[stat].copy()).to(device)
        ref_bases[stat] = ref.from_numpy(inits[stat].copy())
        setattr(bn, stat, _view(bases[stat], *geometry))
        setattr(ref_bn, stat, ref_bases[stat].as_strided(*geometry))
    bn.train()
    ref_bn.train()
    for step in range(_STEPS):
        x = _input(step, x_shape)
        bn(lucid.tensor(x.copy()).to(device))
        ref_bn(ref.from_numpy(x.copy()))
        for stat in ("running_mean", "running_var"):
            np.testing.assert_allclose(
                _flat(getattr(bn, stat)),
                getattr(ref_bn, stat).numpy(),
                rtol=_TOL,
                atol=_TOL,
                err_msg=f"{stat} after step {step + 1}",
            )
    untouched = ~_covered(geometry)
    for stat, base in bases.items():
        after = _flat(base)
        np.testing.assert_allclose(after, ref_bases[stat].numpy(), rtol=_TOL, atol=_TOL)
        np.testing.assert_array_equal(after[untouched], inits[stat][untouched])
    assert int(bn.num_batches_tracked.item()) == _STEPS


@pytest.mark.parametrize("norm", list(_NORMS))
@pytest.mark.parametrize("layout", list(_LAYOUTS))
@pytest.mark.parametrize("device, dtype", device_dtype_params([lucid.float32, lucid.float64]))
def test_a_running_stat_view_updates_as_a_dense_one(
    norm: str, layout: str, device: str, dtype: lucid.dtype
) -> None:
    # Bit for bit: the same update over the same values, only the layout of
    # the buffers differs.
    name, x_shape = _NORMS[norm]
    viewed = getattr(nn, name)(_C, momentum=_MOMENTUM, dtype=dtype).to(device)
    dense = getattr(nn, name)(_C, momentum=_MOMENTUM, dtype=dtype).to(device)
    for k, stat in enumerate(("running_mean", "running_var")):
        base = lucid.tensor(_base_init(k), dtype=dtype).to(device)
        setattr(viewed, stat, _view(base, *_LAYOUTS[layout]))
        setattr(dense, stat, getattr(viewed, stat).clone())
    for step in range(_STEPS):
        x = lucid.tensor(_input(step, x_shape), dtype=dtype).to(device)
        viewed(x)
        dense(x)
    for stat in ("running_mean", "running_var"):
        np.testing.assert_array_equal(_flat(getattr(viewed, stat)), _flat(getattr(dense, stat)))


# What a user writes on the CPU, as in the LCD-491 report: each maps to a
# geometry over the 4 x 6 base.
_NATURAL: dict[str, tuple[Callable[[Any], Any], tuple[tuple[int, ...], tuple[int, ...], int]]] = {
    "column": (lambda x: x[:3, 2], ((_C,), (6,), 2)),
    "transpose": (lambda x: x.t()[4][:3], ((_C,), (6,), 4)),
    "offset_slice": (lambda x: x.reshape(-1)[7:10], ((_C,), (1,), 7)),
    "leading_slice": (lambda x: x.reshape(-1)[:3], ((_C,), (1,), 0)),
}


@pytest.mark.parametrize("norm", list(_NORMS))
@pytest.mark.parametrize("layout", list(_NATURAL))
def test_a_running_stat_made_by_indexing_is_updated_into_its_base(
    norm: str, layout: str, ref: Any
) -> None:
    name, x_shape = _NORMS[norm]
    make, geometry = _NATURAL[layout]
    init = _base_init(0)
    base = lucid.tensor(init.copy())
    ref_base = ref.from_numpy(init.copy())
    bn = getattr(nn, name)(_C, momentum=_MOMENTUM)
    ref_bn = getattr(ref.nn, name)(_C, momentum=_MOMENTUM)
    bn.running_mean = make(base)
    ref_bn.running_mean = make(ref_base)
    for step in range(_STEPS):
        x = _input(step, x_shape)
        bn(lucid.tensor(x.copy()))
        ref_bn(ref.from_numpy(x.copy()))
    after = _flat(base)
    np.testing.assert_allclose(after, ref_base.numpy(), rtol=_TOL, atol=_TOL)
    untouched = ~_covered(geometry)
    np.testing.assert_array_equal(after[untouched], init[untouched])


def test_two_running_stats_over_one_base_each_take_their_own_elements(
    device: str, ref: Any
) -> None:
    mean_at = ((_C,), (6,), 1)
    var_at = ((_C,), (6,), 4)
    init = _base_init(0)
    base = lucid.tensor(init.copy()).to(device)
    ref_base = ref.from_numpy(init.copy())
    bn = nn.BatchNorm2d(_C, momentum=_MOMENTUM).to(device)
    ref_bn = ref.nn.BatchNorm2d(_C, momentum=_MOMENTUM)
    bn.running_mean, bn.running_var = _view(base, *mean_at), _view(base, *var_at)
    ref_bn.running_mean = ref_base.as_strided(*mean_at)
    ref_bn.running_var = ref_base.as_strided(*var_at)
    for step in range(_STEPS):
        x = _input(step, _NORMS["2d"][1])
        bn(lucid.tensor(x.copy()).to(device))
        ref_bn(ref.from_numpy(x.copy()))
    after = _flat(base)
    np.testing.assert_allclose(after, ref_base.numpy(), rtol=_TOL, atol=_TOL)
    untouched = ~_covered(mean_at, var_at)
    np.testing.assert_array_equal(after[untouched], init[untouched])


@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_the_functional_updates_running_stat_views_in_their_base(
    layout: str, device: str, ref: Any
) -> None:
    geometry = _LAYOUTS[layout]
    inits = [_base_init(0), _base_init(1)]
    bases = [lucid.tensor(i.copy()).to(device) for i in inits]
    ref_bases = [ref.from_numpy(i.copy()) for i in inits]
    rm, rv = (_view(b, *geometry) for b in bases)
    ref_rm, ref_rv = (b.as_strided(*geometry) for b in ref_bases)
    for step in range(_STEPS):
        x = _input(step, _NORMS["1d"][1])
        F.batch_norm(lucid.tensor(x.copy()).to(device), rm, rv, training=True, momentum=_MOMENTUM)
        ref.nn.functional.batch_norm(
            ref.from_numpy(x.copy()), ref_rm, ref_rv, training=True, momentum=_MOMENTUM
        )
    untouched = ~_covered(geometry)
    for base, ref_base, init in zip(bases, ref_bases, inits, strict=True):
        after = _flat(base)
        np.testing.assert_allclose(after, ref_base.numpy(), rtol=_TOL, atol=_TOL)
        np.testing.assert_array_equal(after[untouched], init[untouched])


@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_the_engine_update_entry_writes_a_view_as_a_dense_buffer(layout: str, device: str) -> None:
    # ``batch_norm_update_running_stats``: the update ``_update_running_stats``
    # makes when it is called with a finite momentum.
    geometry = _LAYOUTS[layout]
    inits = [_base_init(0), _base_init(1)]
    bases = [lucid.tensor(i.copy()).to(device) for i in inits]
    views = [_view(b, *geometry) for b in bases]
    dense = [v.clone() for v in views]
    for step in range(_STEPS):
        x = lucid.tensor(_input(step, _NORMS["2d"][1])).to(device)
        for rm, rv in (views, dense):
            _C_engine.nn.batch_norm_update_running_stats(
                rm._impl, rv._impl, x._impl, [0, 2, 3], _MOMENTUM, True
            )
    for base, d, init in zip(bases, dense, inits, strict=True):
        # The dense result in the view's elements, every other one as it was.
        want = init.astype(np.float64).reshape(-1)
        want[_positions(geometry)] = _flat(d)
        assert not np.array_equal(want, init.reshape(-1))
        np.testing.assert_array_equal(_flat(base).reshape(-1), want)


@pytest.mark.parametrize("expanded", ["running_mean", "running_var"])
def test_a_running_stat_whose_elements_overlap_is_refused_and_neither_moves(
    expanded: str, device: str
) -> None:
    # Every element of an expanded view is the base's one element: there is
    # no single value to write.  The other statistic is a view too, so the
    # refusal comes after one write-back and has to put it back.
    bn = nn.BatchNorm1d(_C).to(device)
    init = _base_init(0)
    base = lucid.tensor(init.copy()).to(device)
    one = lucid.ones(1, device=device)
    other = "running_var" if expanded == "running_mean" else "running_mean"
    setattr(bn, other, _view(base, *_LAYOUTS["column"]))
    setattr(bn, expanded, _view(one, (_C,), (0,), 0))
    with pytest.raises(RuntimeError, match="overlap"):
        bn(lucid.tensor(_input(0, _NORMS["1d"][1])).to(device))
    np.testing.assert_array_equal(_flat(base), init)
    assert _flat(one).tolist() == [1.0]
    assert int(bn.num_batches_tracked.item()) == 0


# The module paths that update the statistics in Python rebind the buffer to
# a new tensor rather than writing into it, so a view's base is never
# updated.  The engine writer is not involved; these are pinned until the
# Python side is fixed.
_PYTHON_REBINDS: dict[str, Callable[[Any], Any]] = {
    "instance_norm": lambda nn_: nn_.InstanceNorm1d(_C, track_running_stats=True),
    "batch_norm_cumulative": lambda nn_: nn_.BatchNorm1d(_C, momentum=None),
}


@pytest.mark.xfail(
    strict=True,
    reason="LCD-491 FOUND: the Python running-stat updates rebind the buffer "
    "(InstanceNorm, BatchNorm momentum=None)",
)
@pytest.mark.parametrize("module", list(_PYTHON_REBINDS))
def test_a_python_side_running_stat_update_reaches_a_view_base(module: str, ref: Any) -> None:
    init = _base_init(0)
    base = lucid.tensor(init.copy())
    ref_base = ref.from_numpy(init.copy())
    m = _PYTHON_REBINDS[module](nn)
    ref_m = _PYTHON_REBINDS[module](ref.nn)
    m.running_mean = _view(base, *_LAYOUTS["column"])
    ref_m.running_mean = ref_base.as_strided(*_LAYOUTS["column"])
    x = _input(0, _NORMS["1d"][1])
    m(lucid.tensor(x.copy()))
    ref_m(ref.from_numpy(x.copy()))
    np.testing.assert_allclose(_flat(base), ref_base.numpy(), rtol=_TOL, atol=_TOL)
