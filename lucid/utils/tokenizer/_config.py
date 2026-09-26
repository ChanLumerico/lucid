"""Validated reads out of a Hugging Face ``tokenizer.json`` block.

Every pipeline stage in a ``tokenizer.json`` — normalizer, pre-tokenizer,
post-processor, decoder — is a JSON object tagged with a ``"type"`` and
carrying that type's options.  The ``*_from_config`` factories in the
sibling modules all read those objects the same way, so the reading lives
here once.

Each helper raises :class:`ValueError` naming the block it was reading
rather than letting a malformed file surface as a ``KeyError`` or a
``TypeError`` three calls later.  A tokenizer that loads with a stage
silently misread is worse than one that refuses to load: the first
produces plausible ids that are wrong, the second produces an error
message.
"""

import re

#: Unicode ``White_Space`` — what Hugging Face means by whitespace when it
#: splits or strips.  Not :meth:`str.isspace`, which also accepts the
#: information separators U+001C..U+001F.
WHITE_SPACE = (
    "\t\n\v\f\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005"
    "\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
)


def as_object(value: object, where: str) -> dict[str, object]:
    """Return ``value`` as a JSON object or raise.

    Parameters
    ----------
    value : object
        A decoded JSON value.
    where : str
        What was being read, for the error message.

    Returns
    -------
    dict of str to object
        ``value`` itself, narrowed.

    Raises
    ------
    ValueError
        If ``value`` is not a JSON object.
    """
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected a JSON object, got {type(value).__name__}")
    return {str(k): v for k, v in value.items()}


def as_list(value: object, where: str) -> list[object]:
    """Return ``value`` as a JSON array or raise.

    Parameters
    ----------
    value : object
        A decoded JSON value.
    where : str
        What was being read, for the error message.

    Returns
    -------
    list of object
        ``value`` itself, narrowed.

    Raises
    ------
    ValueError
        If ``value`` is not a JSON array.
    """
    if not isinstance(value, list):
        raise ValueError(f"{where}: expected a JSON array, got {type(value).__name__}")
    return list(value)


def type_of(config: dict[str, object], where: str) -> str:
    """Return the ``"type"`` tag of a pipeline block.

    Parameters
    ----------
    config : dict of str to object
        One pipeline block.
    where : str
        What was being read, for the error message.

    Returns
    -------
    str
        The tag, e.g. ``"Metaspace"``.

    Raises
    ------
    ValueError
        If the block has no string ``"type"``.
    """
    tag = config.get("type")
    if not isinstance(tag, str):
        raise ValueError(f"{where}: block has no 'type' tag: {config!r}")
    return tag


def get_str(config: dict[str, object], key: str, where: str, default: str) -> str:
    """Read a string option, falling back to ``default`` when absent.

    Parameters
    ----------
    config : dict of str to object
        One pipeline block.
    key : str
        Option name.
    where : str
        What was being read, for the error message.
    default : str
        Value used when the key is missing or ``null``.

    Returns
    -------
    str
        The option's value.

    Raises
    ------
    ValueError
        If the option is present but not a string.
    """
    value = config.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{where}: option {key!r} must be a string, got {value!r}")
    return value


def get_bool(config: dict[str, object], key: str, where: str, default: bool) -> bool:
    """Read a boolean option, falling back to ``default`` when absent.

    Parameters
    ----------
    config : dict of str to object
        One pipeline block.
    key : str
        Option name.
    where : str
        What was being read, for the error message.
    default : bool
        Value used when the key is missing or ``null``.

    Returns
    -------
    bool
        The option's value.

    Raises
    ------
    ValueError
        If the option is present but not a boolean.
    """
    value = config.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{where}: option {key!r} must be a boolean, got {value!r}")
    return value


def get_int(config: dict[str, object], key: str, where: str, default: int) -> int:
    """Read an integer option, falling back to ``default`` when absent.

    Parameters
    ----------
    config : dict of str to object
        One pipeline block.
    key : str
        Option name.
    where : str
        What was being read, for the error message.
    default : int
        Value used when the key is missing or ``null``.

    Returns
    -------
    int
        The option's value.

    Raises
    ------
    ValueError
        If the option is present but not an integer.
    """
    value = config.get(key)
    if value is None:
        return default
    # ``bool`` is an ``int`` subclass; a flag where a count belongs is a
    # malformed file, not the number 1.
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{where}: option {key!r} must be an integer, got {value!r}")
    return value


def pattern_from_config(value: object, where: str) -> str | re.Pattern[str]:
    """Decode a ``{"String": ...}`` / ``{"Regex": ...}`` pattern object.

    Parameters
    ----------
    value : object
        The ``pattern`` option of a ``Replace`` or ``Split`` block.
    where : str
        What was being read, for the error message.

    Returns
    -------
    str or re.Pattern
        A literal string for ``{"String": s}``, a compiled expression for
        ``{"Regex": r}``.

    Raises
    ------
    ValueError
        If the object is neither form, or the expression does not compile
        under :mod:`re`.

    Notes
    -----
    Hugging Face tokenizers compile these expressions with Oniguruma.  The
    patterns published checkpoints use — whitespace runs, character
    classes, the GPT-2 split expression minus its ``\\p{...}`` escapes —
    mean the same thing under :mod:`re`; one that does not compile here is
    reported rather than approximated.
    """
    obj = as_object(value, where)
    if "String" in obj:
        literal = obj["String"]
        if not isinstance(literal, str):
            raise ValueError(f"{where}: 'String' pattern must be a string")
        return literal
    if "Regex" in obj:
        source = obj["Regex"]
        if not isinstance(source, str):
            raise ValueError(f"{where}: 'Regex' pattern must be a string")
        try:
            return re.compile(source)
        except re.error as exc:
            raise ValueError(
                f"{where}: pattern {source!r} does not compile as a Python "
                f"regular expression ({exc}); this tokenizer cannot be "
                f"reproduced without it"
            ) from exc
    raise ValueError(f"{where}: pattern must be {{'String': ...}} or {{'Regex': ...}}")


def pattern_to_config(pattern: str | re.Pattern[str]) -> dict[str, object]:
    """Inverse of :func:`pattern_from_config`.

    Parameters
    ----------
    pattern : str or re.Pattern
        A literal or a compiled expression.

    Returns
    -------
    dict of str to object
        ``{"String": s}`` or ``{"Regex": source}``.
    """
    if isinstance(pattern, str):
        return {"String": pattern}
    return {"Regex": pattern.pattern}
