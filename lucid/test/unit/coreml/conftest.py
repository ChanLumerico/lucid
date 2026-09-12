"""Core ML tests inside a virtual machine.

A hosted CI runner is a macOS virtual machine, and the Core ML runtime
inside it is not the one on a device.  There is no Neural Engine, and the
GPU goes unused: the probe in ``.github/workflows/coreml-vm-probe.yml``
got the same answer, to the last digit, from CPU_ONLY, CPU_AND_GPU and
ALL — every package on BNNS's CPU path.  That path lands one comparison
a thousand times further from eager than on hardware and traps at
prediction on two unrelated packages: SIGTRAP, nothing in the log, and
the rest of the suite gone with the process.  Other compute units do not
route around it.

So under a hypervisor, loading a package — the step
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
