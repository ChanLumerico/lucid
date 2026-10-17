"""``Module.to`` takes a floating or complex dtype, and casts only such tensors.

``nn.Linear(2, 2).to(lucid.int64)`` cast the weights to int64 and dropped
their ``requires_grad`` without a word, and ``bn.to("cpu", lucid.float64)``
widened ``num_batches_tracked`` to float64: the integer buffers were spared
only for a dtype given alone (LCD-486).  The reference refuses a bool or
integer dtype with ``TypeError`` and applies a dtype to the floating and
complex tensors only, however the call is spelled.
"""

import pytest

import lucid
import lucid.nn as nn


@pytest.mark.parametrize(
    "target",
    [
        lambda lib: (lib.int64,),
        lambda lib: (lib.bool,),
        lambda lib: (lib.zeros(1, dtype=lib.int64),),
        lambda lib: ("cpu", lib.int32),
    ],
    ids=["int64", "bool", "integer tensor", "device and int32"],
)
def test_an_integral_dtype_is_refused(target):
    module = nn.Linear(2, 2)
    weight = module.weight
    with pytest.raises(TypeError, match="only accepts floating point or complex"):
        module.to(*target(lucid))
    assert module.weight is weight and weight.dtype == lucid.float32


@pytest.mark.parity
@pytest.mark.parametrize(
    "target",
    [lambda lib: (lib.int64,), lambda lib: (lib.zeros(1, dtype=lib.int64),)],
    ids=["int64", "integer tensor"],
)
def test_the_reference_refuses_the_same_dtypes(ref, target):
    with pytest.raises(TypeError, match="only accepts floating point or complex"):
        ref.nn.Linear(2, 2).to(*target(ref))


_SPELLINGS = [
    lambda lib: (lib.float64,),
    lambda lib: ("cpu", lib.float64),
    lambda lib: ({"device": "cpu", "dtype": lib.float64},),
    lambda lib: (lib.zeros(1, dtype=lib.float64),),
]
_SPELLING_IDS = ["dtype", "device and dtype", "keywords", "other tensor"]


def _to(module, spelling):
    args = spelling
    if args and isinstance(args[0], dict):
        return module.to(**args[0])
    return module.to(*args)


@pytest.mark.parametrize("spelling", _SPELLINGS, ids=_SPELLING_IDS)
def test_an_integer_buffer_keeps_its_dtype_however_the_call_is_spelled(spelling):
    bn = _to(nn.BatchNorm1d(3), spelling(lucid))
    assert bn.num_batches_tracked.dtype == lucid.int64
    assert bn.weight.dtype == lucid.float64 and bn.running_mean.dtype == lucid.float64
    assert bn.weight.requires_grad


@pytest.mark.parity
@pytest.mark.parametrize("spelling", _SPELLINGS, ids=_SPELLING_IDS)
def test_the_buffer_dtypes_match_the_reference(ref, spelling):
    mine = _to(nn.BatchNorm1d(3), spelling(lucid))
    theirs = _to(ref.nn.BatchNorm1d(3), spelling(ref))
    for name in ("weight", "running_mean", "num_batches_tracked"):
        assert str(getattr(mine, name).dtype).split(".")[-1] == str(
            getattr(theirs, name).dtype
        ).split(".")[-1]


def test_a_device_move_takes_the_integer_buffers_along(device):
    bn = nn.BatchNorm1d(3).to(device, lucid.float16)
    assert bn.num_batches_tracked.dtype == lucid.int64
    assert bn.num_batches_tracked.device == lucid.device(device)
    assert bn.weight.dtype == lucid.float16 and bn.weight.device == lucid.device(device)


def test_a_call_of_no_to_form_is_refused():
    with pytest.raises(TypeError):
        nn.Linear(2, 2).to(device=3)
    with pytest.raises(TypeError):
        nn.Linear(2, 2).to(object())
