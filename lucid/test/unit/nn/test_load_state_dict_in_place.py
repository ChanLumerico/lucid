"""load_state_dict copies into the tensors a module already holds.

Rebinding each parameter and buffer to a new impl left everything that held
the old one — a view of the weight, an alias, a hook — reading the old
values, and replaced the buffer objects a caller may have kept.
"""

import lucid
import lucid.nn as nn


def _flat(t: lucid.Tensor) -> list[float]:
    return [float(v) for v in t.reshape(-1).tolist()]


def test_a_view_of_a_parameter_sees_the_loaded_values() -> None:
    m = nn.Linear(2, 2)
    view = m.weight.detach().reshape(-1)
    before = m.weight._impl
    m.load_state_dict({k: lucid.full(v.shape, 3.0) for k, v in m.state_dict().items()})
    assert m.weight._impl is before
    assert _flat(view) == [3.0] * 4


def test_the_parameter_does_not_share_storage_with_its_source() -> None:
    m = nn.Linear(2, 2)
    source = {k: lucid.full(v.shape, 3.0) for k, v in m.state_dict().items()}
    m.load_state_dict(source)
    source["weight"].fill_(7.0)
    assert _flat(m.weight) == [3.0] * 4


def test_buffers_are_loaded_in_place_too() -> None:
    bn = nn.BatchNorm1d(3)
    running_mean = bn.running_mean
    state = dict(bn.state_dict())
    state["running_mean"] = lucid.full((3,), 5.0)
    bn.load_state_dict(state)
    assert bn.running_mean is running_mean
    assert _flat(running_mean) == [5.0] * 3
