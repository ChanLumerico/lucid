"""
lucid.serialization: save and load tensors and modules.
"""

import io
import pickle
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


class _LucidPickler(pickle.Pickler):
    def persistent_id(self, obj: Any) -> Any:
        from lucid._tensor.tensor import Tensor as _T
        if isinstance(obj, _T):
            arr = np.asarray(obj._impl.data_as_python())
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
            from lucid._dtype import _NAME_TO_DTYPE, to_engine_dtype
            _, shape, dtype_name, device_str, raw_bytes, np_dtype_str = pid
            arr = np.frombuffer(raw_bytes, dtype=np.dtype(np_dtype_str)).reshape(shape).copy()
            eng_dtype = to_engine_dtype(dtype_name)
            eng_device = _C_engine.Device.GPU if device_str == "metal" else _C_engine.Device.CPU
            impl = _C_engine.TensorImpl(arr, eng_device, False)
            return _wrap(impl)
        raise pickle.UnpicklingError(f"Unknown persistent id: {pid}")


def save(obj: Any, f: str | bytes | io.IOBase, *, pickle_protocol: int = 4) -> None:
    """
    Save an object to a file.

    Serializes Tensor objects efficiently using their raw data.
    Modules are saved via their state_dict.

    Args:
        obj:             Object to save (Tensor, Module, dict, etc.)
        f:               File path or file-like object
        pickle_protocol: Pickle protocol version (default: 4)
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
    map_location: Any = None,
) -> Any:
    """
    Load an object from a file.

    Args:
        f:            File path or file-like object
        map_location: Optional device to map tensors to (e.g., "cpu", "metal")
    """
    if isinstance(f, (str, bytes)):
        with open(f, "rb") as fp:
            data = fp.read()
    else:
        data = f.read()  # type: ignore[union-attr]

    buf = io.BytesIO(data)
    unpickler = _LucidUnpickler(buf)
    container = unpickler.load()

    if not isinstance(container, dict) or container.get("_lucid_format") != 1:
        raise RuntimeError("File is not a valid Lucid checkpoint")

    obj = container["obj"]

    if map_location is not None:
        from lucid._tensor.tensor import Tensor as _T

        def _remap(o: Any) -> Any:
            if isinstance(o, _T):
                return o.to(map_location)
            if isinstance(o, dict):
                return {k: _remap(v) for k, v in o.items()}
            return o

        obj = _remap(obj)

    return obj


__all__ = ["save", "load"]
