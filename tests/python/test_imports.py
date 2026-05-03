"""
Tests for import correctness and namespace integrity.
"""

import time
import pytest
import builtins


class TestImportTime:
    def test_import_under_300ms(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-c", "import time; t=time.time(); import lucid; print(time.time()-t)"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        elapsed = float(result.stdout.strip())
        assert elapsed < 0.3, f"Import took {elapsed:.3f}s (limit: 0.3s)"


class TestNamespaceIntegrity:
    def test_no_builtin_shadowing(self):
        import lucid
        # float, int, bool should NOT be in lucid's namespace as dtype aliases
        # (they're only in lucid.dtypes)
        assert not hasattr(lucid, "float") or lucid.float is builtins.float, \
            "lucid.float should not shadow builtins.float"

    def test_bool_not_shadowed(self):
        # After 'from lucid import *', bool should still be Python's bool
        import lucid
        ns: dict = {}
        exec("from lucid import *", ns)
        assert "bool" not in ns or ns["bool"] is builtins.bool, \
            "'from lucid import *' shadows bool"

    def test_float_not_shadowed(self):
        import lucid
        ns: dict = {}
        exec("from lucid import *", ns)
        assert "float" not in ns or ns["float"] is builtins.float, \
            "'from lucid import *' shadows float"

    def test_dtypes_module(self):
        import lucid.dtypes as dt
        assert dt.float is dt.float32
        assert dt.int is dt.int32
        assert dt.bool is dt.bool_

    def test_all_submodules_accessible(self):
        import lucid
        submodules = ["nn", "optim", "autograd", "linalg", "metal", "backends",
                      "utils", "einops", "profiler", "amp"]
        for name in submodules:
            assert hasattr(lucid, name) or name in lucid.__all__, \
                f"lucid.{name} not accessible"


class TestVersionABI:
    def test_version_exists(self):
        import lucid
        assert hasattr(lucid, "__version__")
        assert lucid.__version__ == "3.0.0"
