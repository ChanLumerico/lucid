import inspect
import threading
from typing import Any

from lucid._jit.ir import IRGraph, IRNode, IRValue, ShapeSpec

_trace_local = threading.local()


def is_tracing() -> bool:
    return getattr(_trace_local, "active", False)


def _get_active_tracer() -> TracingContext | None:
    return getattr(_trace_local, "tracer", None)


def _capture_op_init_state(op_self: Any) -> tuple[tuple, dict]:
    base_attrs = {"result", "_inplace", "_inplace_target", "_flops"}
    try:
        sig = inspect.signature(type(op_self).__init__)
        params = list(sig.parameters.values())[1:]

        positional_vals = []
        for p in params:
            if p.name in base_attrs:
                continue
            if hasattr(op_self, p.name):
                positional_vals.append(getattr(op_self, p.name))

        return tuple(positional_vals), {}

    except (ValueError, TypeError):
        return (), {}


class TracingContext:
    def __init__(self) -> None:
        self._value_counter: int = 0
        self._tensor_to_vid: dict[int, int] = {}

        self.values: dict[int, IRValue] = {}
        self.nodes: list[IRNode] = []

        self._node_counter: int = 0
        self.input_ids: list[int] = []
        self.param_ids: set[int] = set()

    def register_tensor(
        self, tensor: Any, *, is_input: bool = False, is_param: bool = False
    ) -> int:
        tid = id(tensor)
        if tid in self._tensor_to_vid:
            return self._tensor_to_vid[tid]

        vid = self._value_counter
        self._value_counter += 1

        spec = ShapeSpec(
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        self.values[vid] = IRValue(value_id=vid, shape_spec=spec, live_tensor=tensor)
        self._tensor_to_vid[tid] = vid

        if is_input:
            self.input_ids.append(vid)
        if is_param:
            self.param_ids.add(vid)

        return vid

    def record_op(
        self,
        op_self: Any,
        input_tensors: tuple,
        output_tensors: tuple,
        non_tensor_args: tuple,
        non_tensor_kwargs: dict,
        device: str,
        has_gradient: bool,
        n_in: int | None,
        n_ret: int | None,
    ) -> None:
        input_ids = [self.register_tensor(t) for t in input_tensors]

        output_ids = []
        for t in output_tensors:
            tid = id(t)
            if tid in self._tensor_to_vid:
                output_ids.append(self._tensor_to_vid[tid])
            else:
                vid = self._value_counter
                self._value_counter += 1
                spec = ShapeSpec(
                    shape=tuple(t.shape),
                    dtype=t.dtype,
                    device=t.device,
                )
                self.values[vid] = IRValue(value_id=vid, shape_spec=spec, live_tensor=t)
                self._tensor_to_vid[tid] = vid
                output_ids.append(vid)

        op_init_args, op_init_kwargs = _capture_op_init_state(op_self)

        node = IRNode(
            node_id=self._node_counter,
            op_class=type(op_self),
            op_init_args=op_init_args,
            op_init_kwargs=op_init_kwargs,
            input_ids=input_ids,
            output_ids=output_ids,
            non_tensor_args=non_tensor_args,
            non_tensor_kwargs=non_tensor_kwargs,
            device=device,
            has_gradient=has_gradient,
            n_in=n_in,
            n_ret=n_ret,
        )
        self.nodes.append(node)
        self._node_counter += 1

    def finalize(self, output_tensors: tuple) -> IRGraph:
        output_ids = []
        for t in output_tensors:
            tid = id(t)
            if tid in self._tensor_to_vid:
                output_ids.append(self._tensor_to_vid[tid])

        return IRGraph(
            input_ids=list(self.input_ids),
            output_ids=output_ids,
            values=self.values,
            nodes=self.nodes,
            param_ids=self.param_ids,
        )
