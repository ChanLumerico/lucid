"""A fallback to eager should be findable, not just correct.

``compile`` builds an MPSGraph when it can and runs the model eagerly
when it cannot. The result is right either way, which is what made the
silence tolerable — and also what made it invisible: a caller who asked
for a compiled model got an eager one with nothing to read anywhere, not
even under ``LUCID_COMPILE_VERBOSE=1``, which narrates everything else
about the build.

The engine already says why. ``compile_trace`` writes a reason and the
binding raises it; the Python layer caught that exception to fall back
and dropped the message with it.
"""

import subprocess
import sys
import textwrap

import pytest

import lucid
from lucid._C import engine as _C_engine


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to("metal")
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")


PROGRAM = textwrap.dedent("""
    import lucid, lucid.nn as nn, lucid.nn.functional as F
    from lucid._C import engine as e

    class Inverse(nn.Module):
        def forward(self, x):
            return lucid.linalg.inv(x)

    model = Inverse().eval().to("metal")
    e.compile.session_cache_clear()
    compiled = lucid.compile.compile(model)
    compiled(lucid.randn(4, 4).to("metal"))
    print("CACHE", e.compile.session_cache_size())
    """)


def _run(verbose: bool) -> subprocess.CompletedProcess:
    import os

    environment = dict(os.environ)
    if verbose:
        environment["LUCID_COMPILE_VERBOSE"] = "1"
    else:
        environment.pop("LUCID_COMPILE_VERBOSE", None)
    return subprocess.run(
        [sys.executable, "-c", PROGRAM],
        capture_output=True,
        text=True,
        env=environment,
        timeout=600,
    )


class TestFallbackIsReported:
    def test_verbose_names_the_operation_and_the_reason(self) -> None:
        done = _run(verbose=True)
        assert done.returncode == 0, done.stderr
        # It really did fall back — otherwise the assertion below would
        # pass for a run that compiled and said nothing.
        assert "CACHE 0" in done.stdout
        assert "eager fallback" in done.stderr
        assert "inv" in done.stderr

    def test_it_stays_quiet_by_default(self) -> None:
        """A fallback is correct, so it is not a warning on every call."""
        done = _run(verbose=False)
        assert done.returncode == 0, done.stderr
        assert "eager fallback" not in done.stderr


class TestRegistrationIsNotCompilation:
    def test_a_registered_emitter_may_still_decline(self) -> None:
        """``emitter_registered`` answers a narrower question than it looks.

        It is a registry lookup. An emitter may be registered and still
        refuse the variant it is handed, which is how six operations
        report as supported and run eagerly.
        """
        assert _C_engine.compile.emitter_registered("inv")

        import lucid.nn as nn

        class Inverse(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return lucid.linalg.inv(x)

        model = Inverse().eval().to("metal")
        _C_engine.compile.session_cache_clear()
        compiled = lucid.compile.compile(model)
        compiled(lucid.randn(4, 4).to("metal"))
        assert _C_engine.compile.session_cache_size() == 0
