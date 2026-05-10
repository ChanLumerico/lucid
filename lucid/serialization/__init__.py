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
from typing import Any, Callable, TYPE_CHECKING, cast

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
            return cast(type, super().find_class(module, name))
        raise pickle.UnpicklingError(
            f"weights_only=True: refusing to deserialize {full!r}. "
            "Pass weights_only=False to allow arbitrary objects "
            "(potential security risk)."
        )

    def persistent_load(self, pid: object) -> object:
        return _LucidUnpickler.persistent_load(self, pid)  # type: ignore[arg-type]


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
    container: dict[str, object] = {"_lucid_format": 2, "obj": obj}
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
        f.write(data)


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
        data = f.read()

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
        return obj.to(map_location)

    if isinstance(obj, dict):
        return {k: _apply_map_location(v, map_location) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        remapped = [_apply_map_location(v, map_location) for v in obj]
        return type(obj)(remapped)
    return obj


def save_sharded(
    obj: object,
    path: str | bytes,
    *,
    shard_size_mb: float = 1024.0,
    pickle_protocol: int = 4,
) -> None:
    """Save an object to a sharded checkpoint directory.

    If *obj* is a ``dict`` (e.g. a Module ``state_dict()``) it is split into
    multiple shard files so that each shard stays under *shard_size_mb* MiB.
    A JSON index (``index.json``) records which keys live in which shard,
    making the directory self-describing.

    For non-dict objects the function falls back to a single-shard write that
    is still compatible with ``load_sharded``.

    Parameters
    ----------
    obj:
        Object to save.  State dicts (``OrderedDict`` / ``dict`` of tensors)
        are the primary use-case.
    path:
        Destination *directory*.  Created if it does not exist.
    shard_size_mb:
        Target maximum size per shard file in MiB.  Tensors that individually
        exceed this limit are placed in their own shard.
    pickle_protocol:
        Pickle protocol forwarded to each per-shard ``save()`` call.
    """
    import json
    import os

    path_str = path.decode() if isinstance(path, bytes) else str(path)
    os.makedirs(path_str, exist_ok=True)

    if not isinstance(obj, dict):
        fname = "shard-00000-of-00001.lucid"
        save(obj, os.path.join(path_str, fname), pickle_protocol=pickle_protocol)
        index: dict[str, object] = {
            "_lucid_sharded": 1,
            "shards": [{"file": fname, "keys": None}],
        }
        with open(os.path.join(path_str, "index.json"), "w", encoding="utf-8") as fp:
            json.dump(index, fp, indent=2)
        return

    limit_bytes = int(shard_size_mb * 1024 * 1024)

    # Bucket (key, value) pairs into shards without splitting any single tensor.
    shards: list[dict[str, object]] = []
    cur: dict[str, object] = {}
    cur_bytes: int = 0

    for k, v in obj.items():
        v_sz = (v.numel() * v.element_size()) if isinstance(v, _T) else 0
        # Flush current shard if adding this tensor would exceed the limit
        # (but never emit an empty shard — always include at least one entry).
        if cur and cur_bytes + v_sz > limit_bytes:
            shards.append(cur)
            cur = {}
            cur_bytes = 0
        cur[k] = v
        cur_bytes += v_sz

    if cur:
        shards.append(cur)
    if not shards:
        shards = [{}]

    n = len(shards)
    index_shards: list[dict[str, object]] = []
    for i, shard_dict in enumerate(shards):
        fname = f"shard-{i:05d}-of-{n:05d}.lucid"
        save(shard_dict, os.path.join(path_str, fname), pickle_protocol=pickle_protocol)
        index_shards.append({"file": fname, "keys": list(shard_dict.keys())})

    index = {"_lucid_sharded": 1, "shards": index_shards}
    sd_meta = getattr(obj, "_metadata", None)
    if sd_meta is not None:
        try:
            # _metadata is a plain dict[str, dict] — JSON-serializable.
            index["_state_dict_metadata"] = sd_meta
        except Exception:
            pass

    with open(os.path.join(path_str, "index.json"), "w", encoding="utf-8") as fp:
        json.dump(index, fp, indent=2)


def load_sharded(
    path: str | bytes,
    *,
    map_location: str | Callable[[str, str], str] | dict[str, str] | None = None,
    weights_only: bool = True,
) -> object:
    """Load a sharded checkpoint saved with ``save_sharded``.

    Reads ``index.json`` from *path*, then loads each shard file in order and
    merges the results into a single ``OrderedDict``.  *map_location* and
    *weights_only* are forwarded to every per-shard ``load()`` call.

    Parameters
    ----------
    path:
        Directory that contains ``index.json`` and the shard files.
    map_location:
        Device remapping forwarded to ``load()``.
    weights_only:
        If ``True`` (default) only tensor-safe types are deserialised.
    """
    import json
    import os
    from collections import OrderedDict

    path_str = path.decode() if isinstance(path, bytes) else str(path)
    index_path = os.path.join(path_str, "index.json")

    with open(index_path, encoding="utf-8") as fp:
        index = json.load(fp)

    if not isinstance(index, dict) or index.get("_lucid_sharded") != 1:
        raise RuntimeError(
            f"{index_path!r} is not a valid Lucid sharded checkpoint index"
        )

    shards_meta: list[dict[str, object]] = index["shards"]

    # Single non-dict shard (fallback path written for non-dict objects).
    if len(shards_meta) == 1 and shards_meta[0].get("keys") is None:
        fname = str(shards_meta[0]["file"])
        return load(
            os.path.join(path_str, fname),
            map_location=map_location,
            weights_only=weights_only,
        )

    result: OrderedDict[str, object] = OrderedDict()
    for shard_meta in shards_meta:
        fname = str(shard_meta["file"])
        shard = load(
            os.path.join(path_str, fname),
            map_location=map_location,
            weights_only=weights_only,
        )
        if isinstance(shard, dict):
            result.update(shard)

    sd_meta = index.get("_state_dict_metadata")
    if sd_meta is not None and isinstance(sd_meta, dict):
        try:
            result._metadata = sd_meta  # type: ignore[attr-defined]
        except AttributeError:
            pass

    return result


__all__ = ["save", "load", "save_sharded", "load_sharded"]
