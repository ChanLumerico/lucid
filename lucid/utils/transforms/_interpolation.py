"""Interpolation modes for resampling transforms.

A small ``str``-valued enum so callers get autocomplete + validation
while the value still flows straight into
:func:`lucid.nn.functional.interpolate` (which takes a string).
"""

import enum


class Interpolation(str, enum.Enum):
    """Resampling mode for resize-style transforms.

    Subclasses ``str`` so members compare equal to their wire value
    (``Interpolation.BILINEAR == "bilinear"``) and pass through to the
    engine's ``interpolate`` without conversion.
    """

    NEAREST = "nearest"
    LINEAR = "linear"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    TRILINEAR = "trilinear"
    AREA = "area"

    def __str__(self) -> str:
        return self.value


# Modes for which ``interpolate`` accepts an ``align_corners`` flag.
_ALIGN_CORNERS_MODES = frozenset(
    {
        Interpolation.LINEAR,
        Interpolation.BILINEAR,
        Interpolation.BICUBIC,
        Interpolation.TRILINEAR,
    }
)


def as_interpolation(mode: "str | Interpolation") -> Interpolation:
    """Coerce a string or enum to an :class:`Interpolation` (validating)."""
    if isinstance(mode, Interpolation):
        return mode
    try:
        return Interpolation(mode)
    except ValueError:
        valid = ", ".join(m.value for m in Interpolation)
        raise ValueError(
            f"unknown interpolation {mode!r}; expected one of: {valid}"
        ) from None
