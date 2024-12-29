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
    PIPELINE: str = r"│   "
    BRANCH: str = r"├── "

    def _register_hook(module: nn.Module, depth: int) -> None:

        def _hook(_module: nn.Module, input_: Tensor, output: Tensor) -> None:
            layer_name = type(_module).__name__
            input_shape = input_[0].shape if isinstance(input_, tuple) else input_.shape
            output_shape = output.shape if isinstance(output, Tensor) else None
            param_size = _module.parameter_size
            layer_count = len(_module._modules)

            if depth == 1:
                layer_name = BRANCH + layer_name
            elif depth > 1:
                layer_name = PIPELINE * (depth - 2) + BRANCH + layer_name

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
        for _, submodule in module._modules.items():
            if recurse:
                _recursive_register(submodule, depth=depth + 1)
            _register_hook(submodule, depth=depth)

    hooks = []
    module_summary = []
    _recursive_register(module=model)

    dummy_input = lucid.zeros(input_shape)
    model(dummy_input)

    title = f"Summary of {type(model).__name__}"
    print(f"{title:^95}")
    print("=" * 95)
    print(f"{"Layer":<30}{"Input Shape":<25}", end="")
    print(f"{"Output Shape":<25}{"Parameter Size":<12}")
    print("=" * 95)

    total_layers = sum(layer["layer_count"] for layer in module_summary)
    total_params = model.parameter_size

    if truncate_from is not None:
        truncated_lines = len(module_summary) - truncate_from
        module_summary = module_summary[:truncate_from]

    for layer in module_summary:
        print(
            f"{layer['layer_name']:<30}{str(layer['input_shape']):<25}",
            f"{str(layer['output_shape']):<25}{layer['param_size']:<12,}",
            sep="",
        )

    if truncate_from is not None:
        print(f"\n{f"... and more {truncated_lines} layer(s)":^95}")

    print("=" * 95)
    print(f"Total Layers(Submodules): {total_layers:,}")
    print(f"Total Parameters: {total_params:,}")
    print("=" * 95)

    for hook in hooks:
        hook()
