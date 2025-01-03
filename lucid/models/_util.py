import lucid
import lucid.nn as nn

from lucid._tensor import Tensor
from lucid.types import _ShapeLike


def summarize(
    model: nn.Module,
    input_shape: _ShapeLike,
    recurse: bool = True,
    truncate_from: int | None = None,
    test_backward: bool = False,
) -> None:
    PIPELINE: str = r"│   "
    BRANCH: str = r"├── "
    LAST_BRANCH: str = r"└── "

    def _register_hook(module: nn.Module, depth: int, is_last: bool) -> None:

        def _hook(_module: nn.Module, input_: Tensor, output: Tensor) -> None:
            layer_name = type(_module).__name__
            input_shape = input_[0].shape if isinstance(input_, tuple) else input_.shape
            output_shape = output.shape if isinstance(output, Tensor) else None
            param_size = _module.parameter_size
            layer_count = len(_module._modules)

            prefix = LAST_BRANCH if is_last else BRANCH
            if depth == 1:
                layer_name = prefix + layer_name
            elif depth > 1:
                layer_name = PIPELINE * (depth - 2) + prefix + layer_name
            if len(layer_name) > 30:
                layer_name = layer_name[:26] + "..."

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
        submodules = list(module._modules.items())
        for idx, (_, submodule) in enumerate(submodules):
            is_last = idx == 0
            _register_hook(submodule, depth + 1, is_last)
            if recurse:
                _recursive_register(submodule, depth=depth + 1)

    hooks = []
    module_summary = []

    _recursive_register(module=model)

    dummy_input = lucid.zeros(input_shape)
    out = model(dummy_input)
    if test_backward:
        out.backward()

    module_summary.reverse()

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
