from typing import Any

from lucid._tensor.tensor import Tensor as _Tensor


def _is_namedtuple(value: Any) -> bool:
    t = type(value)
    return (
        isinstance(value, tuple)
        and hasattr(t, "_fields")
        and isinstance(t._fields, tuple)
    )


def flatten_output(value: Any) -> tuple[list[_Tensor], Any]:
    if isinstance(value, _Tensor):
        return [value], ("leaf",)

    if _is_namedtuple(value):
        nt_type = type(value)
        flat: list[_Tensor] = []
        child_specs: list[Any] = []
        for item in value:
            child_flat, child_spec = flatten_output(item)
            flat.extend(child_flat)
            child_specs.append(child_spec)
        return flat, ("namedtuple", nt_type, child_specs)

    if isinstance(value, tuple):
        flat = []
        child_specs = []
        for item in value:
            child_flat, child_spec = flatten_output(item)
            flat.extend(child_flat)
            child_specs.append(child_spec)
        return flat, ("tuple", child_specs)

    if isinstance(value, list):
        flat = []
        child_specs = []
        for item in value:
            child_flat, child_spec = flatten_output(item)
            flat.extend(child_flat)
            child_specs.append(child_spec)
        return flat, ("list", child_specs)

    if isinstance(value, dict):
        keys = list(value.keys())
        flat = []
        child_specs = []
        for k in keys:
            child_flat, child_spec = flatten_output(value[k])
            flat.extend(child_flat)
            child_specs.append(child_spec)
        return flat, ("dict", keys, child_specs)

    return [], ("const", value)


def unflatten_output(flat: list[_Tensor], treespec: Any) -> Any:
    _, result = _unflatten(flat, treespec, 0)
    return result


def _unflatten(flat: list[_Tensor], treespec: Any, offset: int) -> tuple[int, Any]:
    kind = treespec[0]

    if kind == "leaf":
        return offset + 1, flat[offset]

    if kind == "const":
        return offset, treespec[1]

    if kind in ("tuple", "list"):
        _, child_specs = treespec
        items = []
        for cs in child_specs:
            offset, item = _unflatten(flat, cs, offset)
            items.append(item)
        result = tuple(items) if kind == "tuple" else items
        return offset, result

    if kind == "dict":
        _, keys, child_specs = treespec
        result = {}
        for k, cs in zip(keys, child_specs):
            offset, val = _unflatten(flat, cs, offset)
            result[k] = val
        return offset, result

    if kind == "namedtuple":
        _, nt_type, child_specs = treespec
        items = []
        for cs in child_specs:
            offset, item = _unflatten(flat, cs, offset)
            items.append(item)
        return offset, nt_type(*items)

    raise ValueError(f"Unknown treespec kind: {kind!r}")
