"""Core ML tests inside a virtual machine.

A hosted CI runner is a macOS virtual machine, and the Core ML runtime
inside it is not the one on a device: there is no Neural Engine, one
comparison lands a thousand times further from eager than on hardware,
and two unrelated packages killed the test process by SIGTRAP with
nothing in the log.  So under a hypervisor, loading a package — the step
every prediction, verification and compute plan goes through — skips the
test instead.  Tracing and writing the package still run there; executing
it is left to hardware.

Session-scoped, so a module- or class-scoped fixture that loads a
package is refused too rather than slipping in before a per-test patch.
"""

from collections.abc import Iterator

import pytest

from lucid._C import engine as _C_engine
from lucid.test.unit.coreml._helpers import under_hypervisor


def _refuse(*args: object, **kwargs: object) -> object:
    pytest.skip("the Core ML runtime is not exercised inside a VM; runs on hardware")


@pytest.fixture(autouse=True, scope="session")
def _no_core_ml_runtime_in_a_vm() -> Iterator[None]:
    if not under_hypervisor() or not hasattr(_C_engine, "coreml"):
        yield
        return
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_C_engine.coreml, "load_model", _refuse)
        yield
