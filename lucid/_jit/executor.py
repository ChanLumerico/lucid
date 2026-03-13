import weakref
from typing import Any

from lucid._jit.ir import IRGraph, IRNode
from lucid._backend.core import BackwardOperation
from lucid._tensor.tensor import Tensor as _Tensor


class CompiledPlan:
    def __init__(self, graph: IRGraph, training: bool) -> None:
        self.graph = graph
        self.training = training
        self.exec_order: list[IRNode] = list(graph.nodes)

    def __repr__(self) -> str:
        return (
            f"CompiledPlan(nodes={len(self.exec_order)}, " f"training={self.training})"
        )


class ForwardExecutor:
    def execute(
        self,
        plan: CompiledPlan,
        inputs: tuple,
        param_map: dict[int, Any],
        grad_enabled: bool,
    ) -> tuple:
        graph = plan.graph
        value_map: dict[int, Any] = {}

        for vid, tensor in zip(graph.input_ids, inputs):
            value_map[vid] = tensor

        value_map.update(param_map)

        for vid in graph.constant_ids:
            iv = graph.values.get(vid)
            if iv is not None and iv.live_tensor is not None:
                value_map[vid] = iv.live_tensor

        requires_grad_any = grad_enabled and (
            any(getattr(t, "requires_grad", False) for t in inputs)
            or any(getattr(t, "requires_grad", False) for t in param_map.values())
        )

        for node in plan.exec_order:
            input_tensors = tuple(value_map[vid] for vid in node.input_ids)
            all_args = input_tensors + node.non_tensor_args

            op = node.op_class(*node.op_init_args, **node.op_init_kwargs)
            op_class = type(op)
            if node.device == "gpu":
                raw_func = getattr(op_class.gpu, "__wrapped__", None)
            else:
                raw_func = getattr(op_class.cpu, "__wrapped__", None)

            if raw_func is not None:
                raw_result = raw_func(op, *all_args, **node.non_tensor_kwargs)
            else:
                raw_result = op(
                    *input_tensors, *node.non_tensor_args, **node.non_tensor_kwargs
                )
                n_ret = node.n_ret
                if n_ret == 1 or not isinstance(raw_result, tuple):
                    raw_results = (raw_result,)
                else:
                    raw_results = raw_result
                for out_id, result_tensor in zip(node.output_ids, raw_results):
                    value_map[out_id] = result_tensor
                continue

            n_ret = node.n_ret
            if n_ret == 1:
                pairs = (raw_result,)
            elif n_ret is None:
                if (
                    isinstance(raw_result, tuple)
                    and len(raw_result) == 2
                    and isinstance(raw_result[0], _Tensor)
                ):
                    pairs = (raw_result,)
                else:
                    pairs = raw_result
            else:
                pairs = raw_result

            result_needs_grad = requires_grad_any and node.has_gradient and grad_enabled

            for out_id, (result_tensor, grad_func) in zip(node.output_ids, pairs):
                result_tensor.requires_grad = result_needs_grad

                if plan.training and result_needs_grad:
                    tensor_refs = tuple(weakref.ref(t) for t in input_tensors)
                    result_tensor._backward_op = BackwardOperation(
                        forward_op_ref=weakref.ref(op),
                        grad_func=grad_func,
                        tensor_refs=tensor_refs,
                        versions=tuple(t._version for t in input_tensors),
                        device=node.device,
                    )
                    result_tensor._prev = list(input_tensors)
                    result_tensor._op = op

                value_map[out_id] = result_tensor

        outputs = tuple(value_map[vid] for vid in graph.output_ids)
        return outputs
