from dataclasses import dataclass, field
from typing import Any, Callable

from lucid.types import _DeviceType


@dataclass
class ShapeSpec:
    shape: tuple[int, ...]
    dtype: Any
    device: _DeviceType


@dataclass
class IRValue:
    value_id: int
    shape_spec: ShapeSpec
    live_tensor: Any = field(default=None, repr=False)


@dataclass
class IRNode:
    node_id: int
    op_class: type
    op_init_args: tuple
    op_init_kwargs: dict

    input_ids: list[int]
    output_ids: list[int]

    non_tensor_args: tuple
    non_tensor_kwargs: dict

    device: _DeviceType
    has_gradient: bool
    n_in: int | None
    n_ret: int | None

    fused_forward_kernel: Callable | None = None


@dataclass
class IRGraph:
    input_ids: list[int]
    output_ids: list[int]
    values: dict[int, IRValue]

    nodes: list[IRNode]
    param_ids: set[int]

    @property
    def constant_ids(self) -> set[int]:
        produced: set[int] = set()
        for node in self.nodes:
            produced.update(node.output_ids)
        return set(self.values.keys()) - set(self.input_ids) - self.param_ids - produced
