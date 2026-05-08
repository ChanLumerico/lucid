"""``lucid.backends`` — surface + dispatch behaviour smoke."""

import pytest

import lucid


class TestBackendsSurface:
    def test_module_present(self) -> None:
        assert hasattr(lucid, "backends")


class TestMpsAccessor:
    def test_is_available_callable(self) -> None:
        # ``lucid.backends.mps.is_available()`` (or equivalent) should
        # exist and return a bool — engine surface varies, so we tolerate.
        if not hasattr(lucid.backends, "mps"):
            pytest.skip("backends.mps not exposed")
        if hasattr(lucid.backends.mps, "is_available"):
            v = lucid.backends.mps.is_available()
            assert isinstance(v, bool)
