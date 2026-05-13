"""Cross-task shared utilities used by multiple model families.

These helpers are task-agnostic and may be imported by any model
regardless of its output task (classification, detection, segmentation …).
"""


def make_divisible(
    v: float,
    divisor: int = 8,
    min_value: int | None = None,
) -> int:
    """Round ``v`` to the nearest multiple of ``divisor``.

    The result is at least ``min_value`` (falls back to ``divisor`` when
    *None*).  The 0.9 × v guard prevents the value from being rounded
    down excessively — if the adjusted value is more than 10 % below ``v``
    an extra ``divisor`` is added.

    This is the canonical implementation used by MobileNet, EfficientNet,
    SE-ResNet, SK-ResNet, ResNeSt and any other family that needs channel
    counts aligned to a power-of-two-friendly grid.

    Args:
        v:         Raw (possibly non-integer) channel count.
        divisor:   Alignment granularity.  Defaults to 8.
        min_value: Hard lower bound on the result.  ``None`` → ``divisor``.

    Returns:
        Rounded integer channel count.
    """
    min_val: int = min_value if min_value is not None else divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
