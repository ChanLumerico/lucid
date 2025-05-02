from pathlib import Path
from typing import Literal
import json

import lucid
import lucid.nn as nn

from lucid._tensor import Tensor
from lucid.types import _ShapeLike


def _get_input_shape(args: tuple[Tensor | tuple | list]) -> _ShapeLike | None:
    for arg in args:
        if isinstance(arg, Tensor):
            return arg.shape
        elif isinstance(arg, (tuple, list)):
            for a in arg:
                if isinstance(a, Tensor):
                    return a.shape
    return None


def _format_number(num: int | float, decimals: int = 2) -> str:
    units = ["", "K", "M", "B", "T"]
    mag = 0
    while abs(num) >= 1000 and mag < len(units) - 1:
        num /= 1000.0
        mag += 1

    return f"{num:.{decimals}f}{units[mag]}"


def summarize(
    model: nn.Module,
    input_shape: _ShapeLike | list[_ShapeLike],
    recurse: bool = True,
    truncate_from: int | None = None,
    test_backward: bool = False,
    do_eval: bool = False,
    **model_kwargs,
) -> None:
    PIPELINE: str = r"│   "
    BRANCH: str = r"├── "

    def _register_hook(module: nn.Module, depth: int) -> None:
        def _hook(
            _module: nn.Module, input_arg: tuple, output: Tensor | tuple[Tensor]
        ) -> None:
            input_shape = _get_input_shape(input_arg)
            output_shape = output.shape if isinstance(output, Tensor) else None

            param_size = _module.parameter_size
            layer_name = type(_module).__name__
            layer_count = len(_module._modules)

            if depth == 1:
                layer_name = BRANCH + layer_name
            elif depth > 1:
                layer_name = PIPELINE * (depth - 2) + BRANCH + layer_name
            if len(layer_name) > 34:
                layer_name = layer_name[:32] + "..."

            summary_ = dict(
                layer_name=layer_name,
                input_shape=input_shape,
                output_shape=output_shape,
                param_size=param_size,
                layer_count=layer_count,
            )
            module_summary.append(summary_)

        hooks.append(module.register_forward_hook(_hook))

    def _recursive_register(module: nn.Module, depth: int = 0) -> None:
        _register_hook(module, depth)
        for _, submodule in module._modules.items():
            if recurse:
                _recursive_register(submodule, depth=depth + 1)

    hooks = []
    module_summary = []
    _recursive_register(module=model)

    dummy_inputs = []
    if isinstance(input_shape, list):
        for in_shape in input_shape:
            dummy_inputs.append(lucid.random.rand(in_shape, device=model.device))
    else:
        dummy_inputs.append(lucid.random.rand(input_shape, device=model.device))

    with lucid.count_flops():
        outputs = model(*dummy_inputs, **model_kwargs)

    total_flops = 0
    outputs_tuple = outputs if isinstance(outputs, tuple) else (outputs,)
    for out in outputs_tuple:
        total_flops = max(total_flops, out.flops)
        if do_eval:
            out.eval()
        if test_backward:
            out.backward()

    module_summary.reverse()
    title = f"Summary of {type(model).__name__}"
    if model._alt_name:
        title += f"({model._alt_name})"

    print(f"{title:^95}")
    print("=" * 95)
    print(f"{"Layer":<36}{"Input Shape":<22}", end="")
    print(f"{"Output Shape":<22}{"Parameter Size":<12}")
    print("=" * 95)

    total_layers = sum(layer["layer_count"] for layer in module_summary)
    total_params = model.parameter_size

    if truncate_from is not None:
        truncated_lines = len(module_summary) - truncate_from
        if truncated_lines > 0:
            module_summary = module_summary[:truncate_from]

    for layer in module_summary:
        print(
            f"{layer['layer_name']:<36}{str(layer['input_shape']):<22}",
            f"{str(layer['output_shape']):<22}",
            sep="",
            end="",
        )
        if layer["param_size"]:
            print(f"{layer['param_size']:<12,}")
        else:
            print("-")

    if truncate_from is not None and truncated_lines > 0:
        print(f"\n{f"... and more {truncated_lines} layer(s)":^95}")

    print("=" * 95)
    print(f"Total Layers(Submodules): {total_layers:,}")
    print(f"Total Parameters: {total_params:,} ({_format_number(total_params)})")
    print(f"Total FLOPs: {total_flops:,} ({_format_number(total_flops)})")
    print("=" * 95)

    for hook in hooks:
        hook()


def get_model_names(registry_path: Path = lucid.REGISTRY_PATH) -> list[str]:
    model_name_list = []
    try:
        with open(registry_path, "r") as file:
            registry = json.load(file)

        for model_info in registry.values():
            model_name_list.append(model_info["name"])

    except FileNotFoundError:
        print(f"File not found: {registry_path}")

    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {registry_path}")

    except KeyError as e:
        print(f"Missing key {e} in one of the registry entries.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    model_name_list.sort()
    return model_name_list
