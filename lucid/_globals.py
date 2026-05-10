import threading
from lucid._dtype import dtype, float32 as _f32
from lucid._device import device as _dev

_lock: threading.Lock = threading.Lock()
_default_dtype: dtype = _f32
_default_device_str: str = "cpu"


def set_default_dtype(d: dtype) -> None:
    """Set the default dtype used by factory functions."""
    global _default_dtype
    with _lock:
        _default_dtype = d


def get_default_dtype() -> dtype:
    """Return the current default dtype."""
    return _default_dtype


def set_default_device(d: _dev | str) -> None:
    """Set the default device used by factory functions."""
    global _default_device_str
    with _lock:
        _default_device_str = d if isinstance(d, str) else d.type


def get_default_device() -> str:
    """Return the current default device as a string."""
    return _default_device_str
