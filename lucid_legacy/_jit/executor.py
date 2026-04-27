from dataclasses import dataclass

import lucid
from lucid._jit.ir import IRGraph, IRNode
from lucid._tensor.tensor import Tensor as _Tensor


class CompiledPlan:
    def __init__(
        self, graph: IRGraph, training: bool, output_treespec: object = None
    ) -> None:
        self.graph = graph
        self.training = training
        self.output_treespec = output_treespec
        self.exec_order: list[IRNode] = list(graph.nodes)

    def __repr__(self) -> str:
        return (
            f"CompiledPlan(nodes={len(self.exec_order)}, " f"training={self.training})"
        )


@dataclass
class ForwardResult:
    grad_func_map: dict
    value_map: dict
    exec_order: list
    leaf_vids: frozenset
    node_device_map: dict


class BackwardExecutor:
    def execute(
        self,
        forward_result: ForwardResult,
        output_vids: list,
        output_upstream_grads: dict,
        retain_grad: bool = False,
    ) -> None:
        grad_map: dict = {}
        for vid in output_vids:
            g = output_upstream_grads.get(vid)
            if g is None:
                t = forward_result.value_map.get(vid)
                if t is not None:
                    g = t.grad
            if g is not None:
                grad_map[vid] = g

        for node in reversed(forward_result.exec_order):
            if not node.has_gradient:
                continue

            device = forward_result.node_device_map[node.node_id]

            for out_id in node.output_ids:
                upstream = grad_map.get(out_id)
                if upstream is None:
                    continue
                grad_func = forward_result.grad_func_map.get(out_id)
                if grad_func is None:
                    continue

                result_tensor = forward_result.value_map[out_id]
                result_tensor.grad = upstream

                raw_grads = grad_func()
                if not isinstance(raw_grads, tuple):
                    raw_grads = (raw_grads,)

                for in_vid, g in zip(node.input_ids, raw_grads):
                    if g is None:
                        continue

                    in_tensor = forward_result.value_map.get(in_vid)
                    if in_tensor is None or not in_tensor.requires_grad:
                        continue

                    matched = lucid._match_grad_shape(in_tensor.data, g, device=device)
                    if in_vid in forward_result.leaf_vids:
                        lucid._set_tensor_grad(in_tensor, matched)
                    else:
                        existing = grad_map.get(in_vid)
                        grad_map[in_vid] = (
                            matched if existing is None else existing + matched
                        )

                if not retain_grad:
                    result_tensor.grad = None


class ForwardExecutor:
    def execute(
        self,
        plan: CompiledPlan,
        inputs: tuple,
        param_map: dict,
        grad_enabled: bool,
    ) -> tuple:
        graph = plan.graph
        value_map: dict = {}

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

        build_backward = plan.training and requires_grad_any and grad_enabled
        grad_func_map: dict = {}

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

                if build_backward and result_needs_grad:
                    grad_func_map[out_id] = grad_func

                value_map[out_id] = result_tensor

        outputs = tuple(value_map[vid] for vid in graph.output_ids)

        fwd_result = None
        if build_backward:
            fwd_result = ForwardResult(
                grad_func_map=grad_func_map,
                value_map=value_map,
                exec_order=plan.exec_order,
                leaf_vids=frozenset(graph.input_ids) | graph.param_ids,
                node_device_map={n.node_id: n.device for n in plan.exec_order},
            )

        return outputs, fwd_result
