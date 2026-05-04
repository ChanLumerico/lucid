"""lucid.test — testing utilities and pytest test suite.

Public API (mirrors lucid.testing for backward compatibility):
    assert_close   Numerically compare two tensors element-wise.
"""

from lucid.test._comparison import assert_close

__all__ = ["assert_close"]
