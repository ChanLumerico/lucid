import pickle
from pathlib import Path
from collections import OrderedDict
from typing import Literal

import lucid
from lucid._tensor import Tensor
from lucid.nn import Module


__all__ = ["save", "load"]

_LucidPortable = Tensor | Module | OrderedDict

FORMAT_VERSION: int = 1.0

EXTENSIONS = Literal[".lct", ".lcm", ".lcd"]


def save(obj: _LucidPortable, path: Path | str) -> Path:
    path = Path(path) if isinstance(path, str) else path
    if path.suffix == "":
        if isinstance(obj, Tensor):
            path = path.with_suffix(".lct")
        elif isinstance(obj, Module):
            path = path.with_suffix(".lcm")
        elif isinstance(obj, OrderedDict):
            path = path.with_suffix(".lcd")
        else:
            raise TypeError(
                "Cannot infer file extension: "
                "provide full path or use a recognized type "
                "(Tensor, Module, state_dict)."
            )

    suffix: EXTENSIONS = path.suffix
    if suffix == ".lct":
        if not isinstance(obj, Tensor):
            raise TypeError("Expected a Tensor for .lct file.")
        data = {"type": "Tensor", "format_version": FORMAT_VERSION, "content": obj}

    elif suffix == ".lcd":
        if isinstance(obj, Module):
            obj = obj.state_dict()
        if not isinstance(obj, OrderedDict):
            raise TypeError("Expected a state_dict (OrderedDict) for .lcd file.")
        data = {"type": "OrderedDict", "format_version": FORMAT_VERSION, "content": obj}

    elif suffix == ".lcm":
        if not isinstance(obj, Module):
            raise TypeError("Expected an nn.Module for .lcm file.")
        data = {"type": "Module", "format_version": FORMAT_VERSION, "content": obj}

    else:
        raise ValueError(f"Unsupported file extension: {suffix}")

    with open(path, "wb") as f:
        pickle.dump(data, f)

    return path.resolve()


def load(path: Path | str) -> _LucidPortable:
    path = Path(path) if isinstance(path, str) else path
    suffix: EXTENSIONS = path.suffix

    if suffix not in {".lct", ".lcd", ".lcm"}:
        raise ValueError(f"Unsupported file extension: {suffix}")

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict) or "type" not in obj or "content" not in obj:
        raise ValueError("Invalid Lucid file format.")

    match suffix:
        case ".lct":
            if obj["type"] != "Tensor":
                raise TypeError("Expected 'Tensor' content in .lct file.")
            return obj["content"]

        case ".lcd":
            if obj["type"] != "OrderedDict":
                raise TypeError("Expected 'OrderedDict' content in .lcd file.")
            return obj["content"]

        case ".lcm":
            if obj["type"] != "Module":
                raise TypeError("Expected 'Module' content in .lcm file.")
            return obj["content"]

        case _:
            raise ValueError("Unexpected error during loading.")
