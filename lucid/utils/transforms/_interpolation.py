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


# OpenCV ``INTER_*`` codes → Lucid modes, so Albumentations-style
# ``interpolation=cv2.INTER_LINEAR`` (an int) is accepted verbatim.
_CV2_CODES = {
    0: Interpolation.NEAREST,  # INTER_NEAREST
    1: Interpolation.BILINEAR,  # INTER_LINEAR
    2: Interpolation.AREA,  # INTER_AREA
    3: Interpolation.BICUBIC,  # INTER_CUBIC
    4: Interpolation.BICUBIC,  # INTER_LANCZOS4 → nearest available
}


def as_interpolation(mode: "str | int | Interpolation") -> Interpolation:
    """Coerce a string, OpenCV int code, or enum to an :class:`Interpolation`."""
    if isinstance(mode, Interpolation):
        return mode
    if isinstance(mode, bool):  # guard: bool is an int subclass
        raise ValueError(f"invalid interpolation {mode!r}")
    if isinstance(mode, int):
        if mode not in _CV2_CODES:
            raise ValueError(
                f"unknown OpenCV interpolation code {mode}; "
                f"expected one of {sorted(_CV2_CODES)}"
            )
        return _CV2_CODES[mode]
    try:
        return Interpolation(mode)
    except ValueError:
        valid = ", ".join(m.value for m in Interpolation)
        raise ValueError(
            f"unknown interpolation {mode!r}; expected one of: {valid}"
        ) from None
