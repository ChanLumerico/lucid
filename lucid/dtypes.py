"""
lucid.dtypes: named aliases for all scalar data types.

Import from here when you need the short form:
    from lucid.dtypes import float, int, bool

This avoids shadowing Python builtins in the top-level namespace.
"""

from lucid._dtype import (
    dtype,
    float16,
    float32,
    float64,
    bfloat16,
    int8,
    int16,
    int32,
    int64,
    bool_,
    complex64,
    half,
    double,
    short,
    long,
)

# Short aliases — only accessible via lucid.dtypes.* to avoid shadowing builtins
float = float32
int = int32
bool = bool_

__all__ = [
    "dtype",
    "float16",
    "float32",
    "float64",
    "bfloat16",
    "int8",
    "int16",
    "int32",
    "int64",
    "bool_",
    "complex64",
    "half",
    "double",
    "short",
    "long",
    "float",
    "int",
    "bool",
]
