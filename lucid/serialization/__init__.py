"""
lucid.serialization: save and load tensors and modules.

The wire format goes through the engine's ``to_bytes`` / ``from_bytes``
contract — no numpy round-trip, so checkpoints can be loaded in a
numpy-free environment.  Format version 3 records the dtype as the
canonical Lucid name (``"float32"``, ``"int64"``, …) instead of the
numpy ``str(arr.dtype)`` it used in v1/v2.
"""

import io
import pickle
import warnings
from typing import Any, Callable, TYPE_CHECKING

from lucid._tensor.tensor import Tensor as _T
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid._dtype import _resolve_dtype_name, to_engine_dtype

# ── Allowed types for weights_only=True ──────────────────────────────────────

_SAFE_CLASSES = frozenset(
    {
        "builtins.dict",
        "builtins.list",
        "builtins.tuple",
        "builtins.int",
        "builtins.float",
        "builtins.str",
        "builtins.bool",
        "builtins.bytes",
        "builtins.NoneType",
        "collections.OrderedDict",
    }
)


class _SafeUnpickler(pickle.Unpickler):
    """Restricted Unpickler that only allows safe types (weights_only=True mode)."""

    def find_class(self, module: str, name: str) -> type:
        full = f"{module}.{name}"
        if full in _SAFE_CLASSES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"weights_only=True: refusing to deserialize {full!r}. "
            "Pass weights_only=False to allow arbitrary objects "
            "(potential security risk)."
        )

    def persistent_load(self, pid: object) -> object:
        return _LucidUnpickler.persistent_load(self, pid)


# ── Pickler / Unpickler ───────────────────────────────────────────────────────


class _LucidPickler(pickle.Pickler):
    def persistent_id(self, obj: object) -> object:
        if isinstance(obj, _T):
            return (
                "tensor_v3",
                list(obj.shape),
                obj.dtype._name,
                "metal" if obj.is_metal else "cpu",
                obj._impl.to_bytes(),
            )
        return None


def _restore_tensor(
    shape: list[int], dtype_name: str, device_str: str, raw_bytes: bytes
) -> object:
    eng_device = _C_engine.Device.GPU if device_str == "metal" else _C_engine.Device.CPU
    eng_dtype = to_engine_dtype(_resolve_dtype_name(dtype_name))
    impl = _C_engine.TensorImpl.from_bytes(
        raw_bytes, list(shape), eng_dtype, eng_device, False
    )
    return _wrap(impl)


class _LucidUnpickler(pickle.Unpickler):
    def persistent_load(self, pid: object) -> object:
        if isinstance(pid, tuple) and pid:
            tag = pid[0]
            if tag == "tensor_v3":
                _, shape, dtype_name, device_str, raw_bytes = pid
                return _restore_tensor(shape, dtype_name, device_str, raw_bytes)
            if tag == "tensor":
                # v1/v2 backward-compat path — those checkpoints stored a
                # numpy dtype string and went through ``np.frombuffer``.
                # We translate the wire dtype to the Lucid name and skip
                # the numpy import entirely.
                _, shape, dtype_name, device_str, raw_bytes, _np_dtype_str = pid
                return _restore_tensor(shape, dtype_name, device_str, raw_bytes)
        raise pickle.UnpicklingError(f"Unknown persistent id: {pid}")


# ── Public API ────────────────────────────────────────────────────────────────


def save(obj: object, f: str | bytes | io.IOBase, *, pickle_protocol: int = 4) -> None:
    """Save an object to a file or file-like object.

    If ``obj`` is a ``dict`` (or subclass) carrying a ``_metadata`` attribute
    set by ``Module.state_dict()``, the metadata is preserved across the
    round-trip and re-attached on ``load()``.
    """
    sd_metadata: object | None = getattr(obj, "_metadata", None)
    container: dict = {"_lucid_format": 2, "obj": obj}
    if sd_metadata is not None:
        container["_state_dict_metadata"] = sd_metadata

    buf = io.BytesIO()
    pickler = _LucidPickler(buf, protocol=pickle_protocol)
    pickler.dump(container)
    data = buf.getvalue()

    if isinstance(f, (str, bytes)):
        with open(f, "wb") as fp:
            fp.write(data)
    else:
        f.write(data)  # type: ignore[union-attr]


def load(
    f: str | bytes | io.IOBase,
    *,
    map_location: str | Callable[[str, str], str] | dict[str, str] | None = None,
    weights_only: bool = True,
) -> object:
    """Load an object saved with lucid.save()."""
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

    if not isinstance(container, dict) or container.get("_lucid_format") not in (1, 2):
        raise RuntimeError("File is not a valid Lucid checkpoint")

    obj = container["obj"]

    # Restore the `_metadata` attribute on state_dicts saved with format v2.
    sd_metadata = container.get("_state_dict_metadata")
    if sd_metadata is not None and isinstance(obj, dict):
        try:
            obj._metadata = sd_metadata  # type: ignore[attr-defined]
        except AttributeError:
            # Plain dict can't hold attrs — re-wrap in OrderedDict.
            from collections import OrderedDict as _OD

            new_obj = _OD(obj)
            new_obj._metadata = sd_metadata  # type: ignore[attr-defined]
            obj = new_obj

    if map_location is not None:
        obj = _apply_map_location(obj, map_location)
        if sd_metadata is not None and isinstance(obj, dict):
            try:
                obj._metadata = sd_metadata  # type: ignore[attr-defined]
            except AttributeError:
                pass

    return obj


def _apply_map_location(
    obj: object,
    map_location: str | Callable[..., Any] | dict[str, str],
) -> object:
    """Recursively apply map_location to all Tensors in obj."""
    if isinstance(obj, _T):
        if callable(map_location) and not isinstance(map_location, str):
            device_str = "metal" if obj.is_metal else "cpu"
            return map_location(obj, device_str)
        if isinstance(map_location, dict):
            device_str = "metal" if obj.is_metal else "cpu"
            target = map_location.get(device_str, device_str)
            return obj.to(target)
        return obj.to(map_location)  # type: ignore[arg-type]

    if isinstance(obj, dict):
        return {k: _apply_map_location(v, map_location) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        remapped = [_apply_map_location(v, map_location) for v in obj]
        return type(obj)(remapped)
    return obj


__all__ = ["save", "load"]
