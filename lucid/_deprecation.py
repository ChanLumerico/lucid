"""Retiring a public name without breaking the code that calls it.

Lucid follows Semantic Versioning.  Within a major version nothing public
disappears, and nothing public changes how it is called, without warning
first: a name on its way out keeps working for at least
:data:`MIN_MINOR_RELEASES` minor releases and says so every time it is
used.  3.x broke that promise several times — ``mobilenet_v1`` became
``mobilenet``, ``GenerationMixin`` became ``CausalLMMixin`` and 43 names
left ``lucid.models``, each in a minor release — and code written a month
earlier stopped importing.

:func:`deprecated` marks a function or class.  The released-surface test
(``lucid/test/unit/audit/test_released_surface.py``) holds the other half:
a name in the last release's surface may leave only once the release it
was marked to leave in has come.

Renaming a function keeps the old name as a deprecated wrapper::

    @deprecated(since="3.16.0", removal="3.18.0", alternative="lucid.models.mobilenet")
    def mobilenet_v1(*args, **kwargs):
        return mobilenet(*args, **kwargs)

Renaming a class keeps the old name as a deprecated subclass, so the new
class itself is left untouched::

    @deprecated(since="3.16.0", removal="3.18.0", alternative="CausalLMMixin")
    class GenerationMixin(CausalLMMixin):
        pass
"""

import warnings
from collections.abc import Callable

__all__ = ["LucidDeprecationWarning", "MIN_MINOR_RELEASES", "deprecated"]

#: A deprecated name stays at least this many minor releases, unless its
#: removal waits for the next major version.
MIN_MINOR_RELEASES = 2


class LucidDeprecationWarning(FutureWarning):
    """A public name is on its way out.

    A ``FutureWarning`` rather than a ``DeprecationWarning``: Python hides
    the latter outside ``__main__``, and the person who has to act on this
    is the one running the code, not only the one developing the library.
    """


def _version(text: str) -> tuple[int, int, int]:
    parts = text.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"expected a MAJOR.MINOR.PATCH version, got {text!r}")
    major, minor, patch = (int(part) for part in parts)
    return major, minor, patch


def _check_window(since: str, removal: str) -> None:
    first, last = _version(since), _version(removal)
    if last[0] > first[0]:
        return
    if last[0] < first[0] or last[1] - first[1] < MIN_MINOR_RELEASES:
        raise ValueError(
            f"a name deprecated in {since} stays for at least "
            f"{MIN_MINOR_RELEASES} minor releases or until the next major "
            f"version; {removal} is too soon"
        )


def deprecated[T: Callable[..., object]](
    *, since: str, removal: str, alternative: str | None = None
) -> Callable[[T], T]:
    """Mark a public function or class as on its way out.

    Every call (or instantiation, or subclassing) then raises a
    :class:`LucidDeprecationWarning` naming the release it goes in and what
    to use instead.  The marked object carries ``__lucid_deprecation__``,
    which the released-surface snapshot records so the removal can be
    held to the date promised here.

    Parameters
    ----------
    since : str
        The release that first warns, ``MAJOR.MINOR.PATCH``.
    removal : str
        The first release the name may be gone from.  At least
        :data:`MIN_MINOR_RELEASES` minor releases after ``since``, or a
        later major version.
    alternative : str, optional
        What to use instead, as the caller would spell it.

    Raises
    ------
    ValueError
        If either version is malformed or ``removal`` comes too soon.
    """
    _check_window(since, removal)

    def mark(target: T) -> T:
        name = getattr(target, "__qualname__", repr(target))
        message = (
            f"{name} is deprecated since Lucid {since} and will be removed in {removal}"
        )
        if alternative:
            message += f"; use {alternative} instead"
        marked = warnings.deprecated(message, category=LucidDeprecationWarning)(target)
        marked.__lucid_deprecation__ = {  # type: ignore[attr-defined]
            "since": since,
            "removal": removal,
        }
        return marked

    return mark
