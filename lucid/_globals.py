import threading
from lucid._dtype import dtype, float32 as _f32
from lucid._device import device as _dev

_lock: threading.Lock = threading.Lock()
_default_dtype: dtype = _f32
_default_device_str: str = "cpu"


def set_default_dtype(d: dtype) -> None:
    """Set the dtype used by factory functions when none is supplied.

    Affects ``lucid.tensor``, ``lucid.zeros``, ``lucid.ones``,
    ``lucid.randn`` and every other constructor that defaults to
    ``dtype=None``.  Existing tensors are not converted.  Thread-safe
    (guarded by a module-level lock); the change becomes visible to
    every thread on the next factory call.

    Parameters
    ----------
    d : lucid.dtype
        New default — typically :data:`lucid.float32` (the framework
        default) or :data:`lucid.float64` for higher-precision research
        runs.  Also invalidates the dispatch-side default-dtype cache
        so subsequent kwargs normalisation re-resolves the engine enum.
    """
    global _default_dtype
    with _lock:
        _default_dtype = d
    # 3.2.2: invalidate the dispatch-side default-dtype cache so that
    # subsequent ``normalize_factory_kwargs`` calls re-resolve the
    # engine enum.  Imported lazily to avoid circular dependency at
    # module load time.
    from lucid._dispatch import _invalidate_default_dtype_cache

    _invalidate_default_dtype_cache()


def get_default_dtype() -> dtype:
    """Return the dtype currently used by factory functions.

    Reads the same global the dispatch path consults — so a freshly
    constructed ``lucid.zeros(3)`` (with no explicit ``dtype``) lands
    on the value returned here.

    Returns
    -------
    lucid.dtype
        Current default dtype.  Initialised to :data:`lucid.float32`;
        mutated by :func:`set_default_dtype`.
    """
    return _default_dtype


def set_default_device(d: _dev | str) -> None:
    """Set the device used by factory functions when none is supplied.

    Same scope as :func:`set_default_dtype` but for placement.
    Existing tensors are not moved; subsequent factory calls observe
    the new default on the next invocation.  Thread-safe under a
    module-level lock.  Invalidates two dispatch-side caches — the
    converter fast path and the kwargs normaliser — so the next
    factory call re-resolves placement from scratch.

    Parameters
    ----------
    d : lucid.device or str
        New default placement.  Either a :class:`~lucid._device.device`
        instance or a string like ``"cpu"`` / ``"metal"``.
    """
    global _default_device_str
    with _lock:
        _default_device_str = d if isinstance(d, str) else d.type
    # Invalidate BOTH device caches:
    #   - ``lucid._factories.converters`` for the ndarray-fast-path
    #   - ``lucid._dispatch`` for ``normalize_factory_kwargs``
    # Imported lazily to avoid circular dependency at module load time.
    from lucid._factories.converters import _invalidate_default_device_cache as _c_inv
    from lucid._dispatch import _invalidate_default_device_cache as _d_inv

    _c_inv()
    _d_inv()


def get_default_device() -> str:
    """Return the default device string used by factory functions.

    Mirror of :func:`get_default_dtype` for placement.

    Returns
    -------
    str
        Either ``"cpu"`` or ``"metal"``.  Initialised to ``"cpu"``;
        mutated by :func:`set_default_device`.
    """
    return _default_device_str
