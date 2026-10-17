"""Engine device errors name the device as the public API does: ``metal``.

Every DeviceMismatch the engine raises takes the device's name from one
function, which said ``gpu`` — a name no Lucid call accepts back, where
``tensor.device`` and every ``device=`` argument say ``metal`` (LCD-307).
"""

import re
from collections.abc import Callable

import pytest

import lucid
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="needs Metal")


def _pair() -> tuple[lucid.Tensor, lucid.Tensor]:
    return lucid.ones(2, 3), lucid.ones(2, 3, device="metal")


#: One op per place that builds the message: the binary kernel, matmul, the
#: nn layer checks, the in-place binary path.
_MIXED: dict[str, Callable[[lucid.Tensor, lucid.Tensor], object]] = {
    "add": lambda c, m: c + m,
    "matmul": lambda c, m: c @ m.mT,
    "layer_norm": lambda c, m: F.layer_norm(c, (3,), weight=m[0], bias=None),
    "add_": lambda c, m: c.add_(m),
}


@pytest.mark.parametrize("op", list(_MIXED))
def test_a_device_mismatch_says_metal(op: str) -> None:
    cpu, metal = _pair()
    with pytest.raises(_C_engine.DeviceMismatch) as info:
        _MIXED[op](cpu, metal)
    message = str(info.value)
    assert "metal" in message, message
    assert not re.search(r"\bgpu\b", message, re.IGNORECASE), message
