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

EXTENSIONS = Literal[".lct", ".lcd"]


def save(obj: _LucidPortable, path: Path | str) -> Path:
    path = Path(path) if isinstance(path, str) else path
    if path.suffix == "":
        if isinstance(obj, Tensor):
            path = path.with_suffix(".lct")
        elif isinstance(obj, (Module, OrderedDict)):
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
        data = {
            "type": "Tensor",
            "format_version": FORMAT_VERSION,
            "content": obj.numpy(),
        }

    elif suffix == ".lcd":
        if isinstance(obj, Module):
            obj = obj.state_dict()
        if not isinstance(obj, OrderedDict):
            raise TypeError("Expected a state_dict (OrderedDict) for .lcd file.")

        data = {"type": "OrderedDict", "format_version": FORMAT_VERSION, "content": obj}

    else:
        raise ValueError(f"Unsupported file extension: {suffix}")

    with open(path, "wb") as f:
        pickle.dump(data, f)

    return path.resolve()


def load(path: Path | str) -> _LucidPortable:
    path = Path(path) if isinstance(path, str) else path

    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    file_type = data.get("type")
    version = data.get("format_version")

    if version != FORMAT_VERSION:
        raise ValueError(f"Incompatible format version: {version} != {FORMAT_VERSION}")

    if file_type == "Tensor":
        array = data["content"]
        return Tensor(array)

    elif file_type == "OrderedDict":
        return data["content"]

    else:
        raise ValueError(f"Unsupported data type in file: {file_type}")
