from lucid.models.conv import *


import lucid
import lucid.nn as nn

from lucid._tensor import Tensor
from lucid.types import _ShapeLike


def summarize(
    model: nn.Module,
    input_shape: _ShapeLike,
    recurse: bool = True,
    truncate_from: int | None = None,
) -> None:

    def _register_hook(module: nn.Module) -> None:

        def _hook(_module: nn.Module, input_: Tensor, output: Tensor) -> None:
            layer_name = type(_module).__name__
            input_shape = input_[0].shape if isinstance(input_, tuple) else input.shape
            output_shape = output.shape if isinstance(output, Tensor) else None
            param_size = _module.parameter_size
            layer_count = len(_module._modules)

            module_summary.append(
                dict(
                    layer_name=layer_name,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    param_size=param_size,
                    layer_count=layer_count,
                )
            )

        hooks.append(module.register_forward_hook(_hook))

    def _recursive_register(module: nn.Module) -> None:
        for _, submodule in module._modules.items():
            _register_hook(submodule)
            if recurse:
                _recursive_register(submodule)

    hooks = []
    module_summary = []
    _recursive_register(module=model)

    dummy_input = lucid.zeros(input_shape)
    model(dummy_input)

    title = f"Summary of {type(model).__name__}"
    print(f"{title:^90}")
    print("=" * 90)
    print(f"{"Layer":<25}{"Input Shape":<25}", end="")
    print(f"{"Output Shape":<25}{"Parameter Size":<12}")
    print("=" * 90)

    total_layers = sum(layer["layer_count"] for layer in module_summary)
    total_params = sum(layer["param_size"] for layer in module_summary)

    if truncate_from is not None:
        truncated_lines = len(module_summary) - truncate_from
        module_summary = module_summary[:truncate_from]

    for layer in module_summary:
        print(
            f"{layer["layer_name"]:<25}{str(layer["input_shape"]):<25}",
            f"{str(layer["output_shape"]):<25}{layer["param_size"]:<12,}",
            sep="",
        )

    if truncate_from is not None:
        print(f"\n{f"... and more {truncated_lines} layer(s)":^90}")

    print("=" * 90)
    print(f"Total Layers(Submodules): {total_layers:,}")
    print(f"Total Parameters: {total_params:,}")
    print("=" * 90)

    for hook in hooks:
        hook()
