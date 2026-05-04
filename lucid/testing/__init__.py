"""lucid.testing — backward-compatibility shim. Use lucid.test instead."""

from lucid.test._comparison import assert_close

__all__ = ["assert_close"]
