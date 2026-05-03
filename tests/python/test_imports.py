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
    def test_dtype_aliases_accessible(self):
        import lucid
        # lucid.float / lucid.int / lucid.bool mirror torch.float / torch.int / torch.bool
        assert lucid.float is lucid.float32, "lucid.float should be float32"
        assert lucid.int is lucid.int32,     "lucid.int should be int32"
        assert lucid.bool is lucid.bool_,    "lucid.bool should be bool_"

    def test_star_import_does_not_shadow_builtins(self):
        # 'from lucid import *' must NOT bring float/int/bool into scope —
        # they are module attributes (like torch.float) but excluded from __all__.
        import lucid
        ns: dict = {}
        exec("from lucid import *", ns)
        assert "float" not in ns or ns["float"] is builtins.float, \
            "'from lucid import *' shadows float"
        assert "int" not in ns or ns["int"] is builtins.int, \
            "'from lucid import *' shadows int"
        assert "bool" not in ns or ns["bool"] is builtins.bool, \
            "'from lucid import *' shadows bool"

    def test_dtypes_module(self):
        import lucid.dtypes as dt
        assert dt.float is dt.float32
        assert dt.int is dt.int32
        assert dt.bool is dt.bool_

    def test_all_submodules_accessible(self):
        import lucid
        submodules = ["nn", "optim", "autograd", "linalg", "metal", "backends",
                      "utils", "einops", "profiler", "amp", "testing"]
        for name in submodules:
            assert hasattr(lucid, name) or name in lucid.__all__, \
                f"lucid.{name} not accessible"


class TestVersionABI:
    def test_version_exists(self):
        import lucid
        assert hasattr(lucid, "__version__")
        assert lucid.__version__ == "3.0.0"
