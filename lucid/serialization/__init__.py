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
    r"""Serialise ``obj`` to a Lucid-format checkpoint file.

    Writes a pickle stream augmented with a custom persistent-id
    protocol that captures every embedded :class:`Tensor` as a typed
    byte blob rather than a numpy array. The resulting ``.lucid`` file
    can be reloaded in environments without numpy installed.

    Parameters
    ----------
    obj : object
        Object graph to serialise. ``dict`` / ``OrderedDict`` state
        dicts are the primary use case; arbitrary picklable objects are
        also supported.
    f : str, bytes, or file-like
        Destination path or open binary file handle.
    pickle_protocol : int, optional
        Pickle protocol version forwarded to the underlying
        ``pickle.Pickler``. Default ``4``.

    Returns
    -------
    None

    Notes
    -----
    State dicts produced by ``Module.state_dict()`` carry a hidden
    ``_metadata`` attribute storing version information per submodule.
    That attribute is detected, packed alongside ``obj`` in the
    container, and reattached on :func:`load`.

    Examples
    --------
    >>> import lucid
    >>> sd = {"w": lucid.randn(3, 3)}
    >>> lucid.serialization.save(sd, "ckpt.lucid")
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
    r"""Load an object saved by :func:`save` or :func:`save_safetensors`.

    By default uses a restricted unpickler that only permits a small
    whitelist of safe primitive types — sufficient for state dicts but
    not for arbitrary objects. Disable this with ``weights_only=False``
    only for trusted files; arbitrary pickle deserialisation can execute
    code.

    Parameters
    ----------
    f : str, bytes, or file-like
        Source path or open binary file handle. Paths ending in
        ``.safetensors`` are delegated to :func:`load_safetensors`.
    map_location : str, dict, callable, or None, optional
        Device remapping applied to every loaded tensor. A bare string
        moves all tensors to that device; a ``dict`` maps source device
        names to targets; a callable receives ``(tensor, source_device)``
        and returns the relocated tensor.
    weights_only : bool, optional
        If ``True`` (default), restrict deserialisation to a safe type
        whitelist. Set to ``False`` only for fully trusted checkpoints.

    Returns
    -------
    object
        The deserialised object, with ``_metadata`` reattached for
        state-dict round-trips.

    Notes
    -----
    Implements two formats: the current persistent-id format
    (``"tensor_v3"``) plus a backward-compatible path for v1/v2
    checkpoints that previously round-tripped through numpy.

    Examples
    --------
    >>> import lucid
    >>> sd = lucid.serialization.load("ckpt.lucid")
    """
    if isinstance(f, (str, bytes)):
        path_str = f.decode() if isinstance(f, bytes) else str(f)
        if path_str.endswith(".safetensors"):
            return load_safetensors(path_str)
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
    r"""Save ``obj`` as a sharded, self-describing checkpoint directory.

    Splits a state dict across multiple shard files capped at
    ``shard_size_mb`` MiB each and emits a JSON index listing which keys
    live in which shard. Useful for very large models where a single
    monolithic file is impractical to host, partial-load, or
    download-resume.

    Parameters
    ----------
    obj : object
        Object to save. ``dict`` / ``OrderedDict`` state dicts are
        sharded; other types are written as a single shard for
        compatibility with :func:`load_sharded`.
    path : str or bytes
        Destination *directory*. Created if it does not exist.
    shard_size_mb : float, optional
        Soft cap on per-shard size in MiB. A tensor exceeding the cap
        gets its own shard. Default ``1024.0``.
    pickle_protocol : int, optional
        Pickle protocol forwarded to every per-shard :func:`save`.
        Default ``4``.

    Returns
    -------
    None

    Notes
    -----
    The packing algorithm is a first-fit pass that never splits a
    single tensor across shards — preserving the property that any
    given key resolves to exactly one file. The index records the
    optional ``_metadata`` attribute when present on a state dict.

    Examples
    --------
    >>> import lucid
    >>> sd = {"w": lucid.randn(4096, 4096)}
    >>> lucid.serialization.save_sharded(sd, "ckpt_dir", shard_size_mb=64)
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
    r"""Load a sharded checkpoint directory written by :func:`save_sharded`.

    Reads ``index.json``, then deserialises each shard with :func:`load`
    and merges the results into a single ``OrderedDict`` preserving key
    order. The ``_metadata`` attribute, if stored in the index, is
    reattached to the merged dictionary.

    Parameters
    ----------
    path : str or bytes
        Directory containing ``index.json`` and the per-shard files.
    map_location : str, dict, callable, or None, optional
        Device remapping forwarded to each per-shard :func:`load` call.
    weights_only : bool, optional
        If ``True`` (default), restrict deserialisation to a safe type
        whitelist.

    Returns
    -------
    OrderedDict or object
        The merged state dict; for single-shard non-dict objects, the
        underlying object as written.

    Notes
    -----
    Shard order is taken from the index file — the function does not
    rely on filesystem listing order, so reproducibility is preserved
    across hosts.

    Examples
    --------
    >>> import lucid
    >>> sd = lucid.serialization.load_sharded("ckpt_dir")
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


def _require_safetensors() -> object:
    """Import safetensors.numpy, raising a helpful error if not installed."""
    try:
        import safetensors.numpy as _st

        return _st
    except ImportError:
        raise ImportError(
            "The 'safetensors' package is required for this operation.\n"
            "Install it with:  pip install safetensors"
        ) from None


def save_safetensors(
    state_dict: dict[str, object],
    path: str,
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    r"""Save a flat state dict as a SafeTensors file.

    SafeTensors is the recommended interchange format for sharing model
    weights: the file layout is a small JSON header followed by raw
    tensor bytes, so loading is fast, zero-copy where possible, and free
    of pickle's code-execution surface area. Use this whenever a
    checkpoint may be shared with untrusted parties.

    Parameters
    ----------
    state_dict : dict of str to Tensor
        Flat mapping from parameter name to :class:`Tensor`. Nested
        dicts or non-tensor values raise ``TypeError``.
    path : str
        Destination file path; use a ``.safetensors`` suffix by
        convention.
    metadata : dict of str to str, optional
        Free-form string metadata stored in the file header (model
        version, training framework, etc.).

    Returns
    -------
    None

    Notes
    -----
    Requires the optional ``safetensors`` Python package
    (``pip install safetensors``). The numpy backend used here does not
    accept bfloat16 — cast such tensors to float32 first. Zero-rank
    scalars are promoted to shape ``(1,)`` on write and squeezed back to
    ``()`` on load via a private metadata key.

    Examples
    --------
    >>> import lucid
    >>> sd = {"w": lucid.randn(3, 3)}
    >>> lucid.serialization.save_safetensors(sd, "weights.safetensors")
    """
    _st = _require_safetensors()

    np_tensors: dict[str, object] = {}
    scalar_keys: list[str] = []
    for name, value in state_dict.items():
        if not isinstance(value, _T):
            raise TypeError(
                f"save_safetensors: state_dict[{name!r}] is "
                f"{type(value).__name__}, expected Tensor"
            )
        if value.dtype._name == "bfloat16":
            raise TypeError(
                f"save_safetensors: tensor {name!r} has dtype bfloat16, "
                "which is not supported by the numpy safetensors backend. "
                "Cast to float32 first: tensor.to(lucid.float32)"
            )
        arr = value.numpy()
        if arr.ndim == 0:
            # SafeTensors does not support 0-d tensors; promote to (1,) and tag.
            arr = arr.reshape((1,))
            scalar_keys.append(name)
        np_tensors[name] = arr

    combined_meta: dict[str, str] = dict(metadata) if metadata else {}
    if scalar_keys:
        combined_meta["__lucid_scalar_keys__"] = ",".join(scalar_keys)
    _st.save_file(np_tensors, path, metadata=combined_meta)  # type: ignore[attr-defined]


def load_safetensors(
    path: str,
    *,
    device: str = "cpu",
) -> dict[str, object]:
    r"""Load a SafeTensors file into a flat state dict.

    Reads the entire file lazily through the ``safetensors`` Python
    package, converts each tensor to a Lucid :class:`Tensor`, optionally
    relocates to the requested device, and restores any zero-rank
    scalars that were promoted to shape ``(1,)`` during :func:`save_safetensors`.

    Parameters
    ----------
    path : str
        Path to a ``.safetensors`` file.
    device : str, optional
        Target device for every loaded tensor: ``"cpu"`` (default) or
        ``"metal"``.

    Returns
    -------
    dict of str to Tensor
        Flat state dict suitable for ``model.load_state_dict()``.

    Notes
    -----
    Requires the optional ``safetensors`` Python package
    (``pip install safetensors``). The header metadata is consulted to
    recover the original 0-d shape of scalar entries written through
    :func:`save_safetensors`.

    Examples
    --------
    >>> import lucid
    >>> sd = lucid.serialization.load_safetensors("weights.safetensors")
    """
    from safetensors import safe_open as _safe_open
    from lucid._factories.converters import from_numpy as _from_numpy

    _require_safetensors()

    import lucid as _lucid

    result: dict[str, object] = {}
    with _safe_open(path, framework="np") as _f:  # type: ignore[no-untyped-call]
        meta: dict[str, str] = _f.metadata() or {}
        scalar_keys: set[str] = set(
            meta.get("__lucid_scalar_keys__", "").split(",")
        ) - {""}
        for name in _f.keys():
            arr = _f.get_tensor(name)
            t = _from_numpy(arr)
            if name in scalar_keys:
                # from_numpy can't produce 0-d tensors; squeeze (1,) → ()
                t = _lucid.squeeze(t)
            if device in ("metal", "gpu"):
                t = t.to("metal")
            result[name] = t
    return result


__all__ = [
    "save",
    "load",
    "save_sharded",
    "load_sharded",
    "save_safetensors",
    "load_safetensors",
]
