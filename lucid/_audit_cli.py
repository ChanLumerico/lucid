"""Console-script shim for ``lucid-audit``.

The audit lives in ``lucid.test.audit``, and ``lucid.test*`` is excluded
from the published wheel on purpose — see ``[tool.setuptools.packages.
find]``.  Pointing ``[project.scripts]`` straight at it would install a
``lucid-audit`` command for every wheel user and have it die on
``ModuleNotFoundError``, which reads as a broken package rather than as
a tool that was never shipped.

This module *is* shipped, so the command always exists; in a wheel it
explains itself and exits, and in a source checkout it forwards.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit sweep, or say why it is not here.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments.  ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        The sweep's exit status — ``0`` no defect survived, ``1`` one
        did, ``2`` the harness itself broke or is not installed.
    """
    try:
        from lucid.test.audit.__main__ import main as _run
    except ModuleNotFoundError:
        print(
            "lucid-audit is a development tool and is not part of the "
            "published wheel.\nRun it from a source checkout:\n"
            "    git clone https://github.com/ChanLumerico/lucid\n"
            "    pip install -e ."
        )
        return 2
    return _run(argv)


__all__ = ["main"]
