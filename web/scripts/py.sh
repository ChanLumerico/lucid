#!/usr/bin/env bash
# Run one of this directory's Python scripts with the right interpreter.
#
# The npm scripts used to call bare ``python3``, which made them depend on
# whether the shell that invoked them had the project's virtualenv active.
# From an activated shell the docs build works; from one that is not --- a
# background job, a `zsh script.sh`, an editor task runner --- the same
# command dies at the first step with ``ERROR: griffe not installed``, and
# the failure looks like a missing dependency rather than a missing PATH
# entry.  It is the same symptom ``.githooks/pre-commit`` hardened against
# by resolving ``.venv/bin/python3`` absolutely, and this applies that fix
# to the other caller of the same scripts.
#
# CI is the case that keeps the PATH fallback: the Docs workflow installs
# Python with ``actions/setup-python`` and has no ``.venv`` at all, so the
# venv must be preferred when present rather than required.
#
# Usage:  ./scripts/py.sh scripts/build-usedby.py [args…]

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${here}/../.." && pwd)"

if [[ -x "${repo_root}/.venv/bin/python3" ]]; then
    exec "${repo_root}/.venv/bin/python3" "$@"
elif command -v python3 >/dev/null 2>&1; then
    exec python3 "$@"
fi

echo "py.sh: no python3 found (checked ${repo_root}/.venv/bin and PATH)" >&2
exit 1
