import pickle
from pathlib import Path
from collections import OrderedDict
from typing import Literal

from lucid._tensor import Tensor
from lucid.nn import Module
from lucid.types import _NumPyArray


__all__ = ["save", "load"]

_LucidPortable = Tensor | Module | OrderedDict

FORMAT_VERSION: float = 1.0

EXTENSIONS = Literal[".lct", ".lcd", ".safetensors"]


def save(obj: _LucidPortable, path: Path | str, safetensors: bool = False) -> Path:
    if getattr(obj, "device", None) == "gpu":
        raise RuntimeError(
            f"Only CPU objects ({type(obj).__name__}) " "are able to be saved."
        )

    path = Path(path) if isinstance(path, str) else path
    if safetensors:
        path = path.with_suffix(".safetensors")

    if path.suffix == "":
        if isinstance(obj, Tensor):
            path = path.with_suffix(".lct")
        elif isinstance(obj, (Module, OrderedDict)):
            path = (
                path.with_suffix(".safetensors")
                if safetensors
                else path.with_suffix(".lcd")
            )
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

    elif suffix == ".safetensors":
        try:
            from safetensors.numpy import save_file
        except Exception as e:
            raise ImportError(
                "safetensors is required to save .safetensors files. "
                "Install with `pip install safetensors`."
            ) from e

        if isinstance(obj, Module):
            obj = obj.state_dict()
        if not isinstance(obj, OrderedDict):
            raise TypeError(
                "Expected a state_dict (OrderedDict) for .safetensors file."
            )

        save_file(obj, str(path))
        return path.resolve()

    else:
        raise ValueError(f"Unsupported file extension: {suffix}")

    with open(path, "wb") as f:
        pickle.dump(data, f)

    return path.resolve()


def load(path: Path | str) -> _LucidPortable:
    path = Path(path) if isinstance(path, str) else path

    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    suffix: str = path.suffix

    if suffix == ".safetensors":
        try:
            from safetensors.numpy import load_file
        except Exception as e:
            raise ImportError(
                "safetensors is required to load .safetensors files. "
                "Install with `pip install safetensors`."
            ) from e

        content = load_file(str(path))
        return OrderedDict((k, v) for k, v in content.items())

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
