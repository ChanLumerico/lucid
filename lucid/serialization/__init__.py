"""
lucid.serialization: save and load tensors and modules.
"""

import io
import pickle
import warnings
from typing import Any, Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


# ── Allowed types for weights_only=True ──────────────────────────────────────

_SAFE_CLASSES = frozenset({
    "builtins.dict", "builtins.list", "builtins.tuple",
    "builtins.int", "builtins.float", "builtins.str", "builtins.bool",
    "builtins.bytes", "builtins.NoneType",
    "numpy.ndarray", "numpy.dtype",
})


class _SafeUnpickler(pickle.Unpickler):
    """Restricted Unpickler that only allows safe types (weights_only=True mode)."""

    def find_class(self, module: str, name: str) -> Any:
        full = f"{module}.{name}"
        if full in _SAFE_CLASSES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"weights_only=True: refusing to deserialize {full!r}. "
            "Pass weights_only=False to allow arbitrary objects "
            "(potential security risk)."
        )

    def persistent_load(self, pid: Any) -> Any:
        return _LucidUnpickler.persistent_load(self, pid)


# ── Pickler / Unpickler ───────────────────────────────────────────────────────

class _LucidPickler(pickle.Pickler):
    def persistent_id(self, obj: Any) -> Any:
        from lucid._tensor.tensor import Tensor as _T
        if isinstance(obj, _T):
            arr = np.ascontiguousarray(np.asarray(obj._impl.data_as_python()))
            return (
                "tensor",
                list(obj.shape),
                obj.dtype._name,
                "metal" if obj.is_metal else "cpu",
                arr.tobytes(),
                str(arr.dtype),
            )
        return None


class _LucidUnpickler(pickle.Unpickler):
    def persistent_load(self, pid: Any) -> Any:
        if isinstance(pid, tuple) and pid[0] == "tensor":
            from lucid._C import engine as _C_engine
            from lucid._dispatch import _wrap
            from lucid._dtype import to_engine_dtype
            _, shape, dtype_name, device_str, raw_bytes, np_dtype_str = pid
            arr = np.frombuffer(raw_bytes, dtype=np.dtype(np_dtype_str)).reshape(shape).copy()
            eng_device = _C_engine.Device.GPU if device_str == "metal" else _C_engine.Device.CPU
            impl = _C_engine.TensorImpl(arr, eng_device, False)
            return _wrap(impl)
        raise pickle.UnpicklingError(f"Unknown persistent id: {pid}")


# ── Public API ────────────────────────────────────────────────────────────────

def save(obj: Any, f: str | bytes | io.IOBase, *, pickle_protocol: int = 4) -> None:
    """Save an object to a file or file-like object.

    Tensors are serialized as raw bytes (efficient). Modules are saved as-is
    (their state_dict is part of the object graph).

    Args:
        obj:             Object to save (Tensor, Module, dict, etc.)
        f:               File path (str/bytes) or file-like object.
        pickle_protocol: Pickle protocol version (default: 4).
    """
    buf = io.BytesIO()
    pickler = _LucidPickler(buf, protocol=pickle_protocol)
    pickler.dump({"_lucid_format": 1, "obj": obj})
    data = buf.getvalue()

    if isinstance(f, (str, bytes)):
        with open(f, "wb") as fp:
            fp.write(data)
    else:
        f.write(data)  # type: ignore[union-attr]


def load(
    f: str | bytes | io.IOBase,
    *,
    map_location: str | Callable[..., Any] | dict[str, str] | None = None,
    weights_only: bool = True,
) -> Any:
    """Load an object saved with lucid.save().

    Args:
        f:             File path or file-like object.
        map_location:  Remap tensor devices on load. Accepts:
                       - str: device name ('cpu', 'metal')
                       - dict: {'metal': 'cpu', 'cpu': 'cpu'}
                       - callable: map_location(tensor, location_string) → Tensor
        weights_only:  If True (default), only allow safe types (tensors, dicts,
                       lists, scalars). Prevents arbitrary code execution.
                       Pass False to allow loading Module objects etc.

    Returns:
        The deserialized object.
    """
    if isinstance(f, (str, bytes)):
        with open(f, "rb") as fp:
            data = fp.read()
    else:
        data = f.read()  # type: ignore[union-attr]

    buf = io.BytesIO(data)

    if weights_only:
        unpickler: pickle.Unpickler = _SafeUnpickler(buf)
    else:
        warnings.warn(
            "lucid.load() with weights_only=False allows arbitrary object "
            "deserialization which may execute untrusted code.",
            UserWarning,
            stacklevel=2,
        )
        unpickler = _LucidUnpickler(buf)

    container = unpickler.load()

    if not isinstance(container, dict) or container.get("_lucid_format") != 1:
        raise RuntimeError("File is not a valid Lucid checkpoint")

    obj = container["obj"]

    if map_location is not None:
        obj = _apply_map_location(obj, map_location)

    return obj


def _apply_map_location(
    obj: Any,
    map_location: str | Callable[..., Any] | dict[str, str],
) -> Any:
    """Recursively apply map_location to all Tensors in obj."""
    from lucid._tensor.tensor import Tensor as _T

    if isinstance(obj, _T):
        if callable(map_location) and not isinstance(map_location, str):
            device_str = "metal" if obj.is_metal else "cpu"
            return map_location(obj, device_str)
        if isinstance(map_location, dict):
            device_str = "metal" if obj.is_metal else "cpu"
            target = map_location.get(device_str, device_str)
            return obj.to(target)
        # String: simple device remap
        return obj.to(map_location)  # type: ignore[arg-type]

    if isinstance(obj, dict):
        return {k: _apply_map_location(v, map_location) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        remapped = [_apply_map_location(v, map_location) for v in obj]
        return type(obj)(remapped)
    return obj


__all__ = ["save", "load"]
