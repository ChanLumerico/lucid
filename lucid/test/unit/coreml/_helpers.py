"""What the machine running the Core ML tests can and cannot do.

A hosted CI runner is a virtual machine, and Core ML inside one is not
Core ML on a device.  There is no Neural Engine, so every placement
claim reads 0.0; and the paravirtual GPU computes some graphs far less
accurately than a real one — ZFNet, the zoo's one model with local
response normalisation, came back 1.7e-3 from eager there, against
2.2e-6 on an M1 Pro's GPU and 1.7e-6 on its CPU.  Tests ask these
questions directly rather than inferring the machine from the answers
they are there to check.
"""

import functools
import subprocess

import pytest

import lucid.coreml as cml


@functools.cache
def under_hypervisor() -> bool:
    """Whether this process is running inside a virtual machine.

    ``kern.hv_vmm_present`` is 1 under a hypervisor and 0 on hardware.
    """
    try:
        done = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "kern.hv_vmm_present"],
            capture_output=True,
            text=True,
            check=True,
        )
    except OSError, subprocess.CalledProcessError:
        return False
    return done.stdout.strip() == "1"


@functools.cache
def has_neural_engine() -> bool:
    """Whether Core ML has a Neural Engine to place work on here.

    Asked of Core ML's own device list rather than read off a compute
    plan: a plan is what the placement tests check, and inferring the
    hardware from it would skip exactly the case that matters — an
    engine that is there and unused.
    """
    try:
        from coremltools.models.compute_device import (  # noqa: PLC0415
            MLComputeDevice,
            MLNeuralEngineComputeDevice,
        )
    except ImportError:
        return True  # cannot tell, so assert as if it were there
    return any(
        isinstance(device, MLNeuralEngineComputeDevice)
        for device in MLComputeDevice.get_all_compute_devices()
    )


def require_neural_engine() -> None:
    if not has_neural_engine():
        pytest.skip("no Neural Engine on this machine; a hosted CI runner is a VM")


def translation_units() -> cml.ComputeUnits:
    """Where to run a package whose job is to show the translation holds.

    ``ALL`` on hardware, so the GPU path a deployment takes is checked
    too.  ``CPU_ONLY`` inside a virtual machine, whose GPU is not one any
    deployment runs on: comparing against it measures the paravirtual
    device, not the export.
    """
    return cml.ComputeUnits.CPU_ONLY if under_hypervisor() else cml.ComputeUnits.ALL
