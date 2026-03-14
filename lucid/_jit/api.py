import functools
import weakref
from typing import Any, Callable, overload

import lucid
from lucid._jit.tracer import TracingContext, _trace_local
from lucid._jit.passes import (
    run_passes,
    DEFAULT_INFERENCE_PASSES,
    DEFAULT_TRAINING_PASSES,
)
from lucid._jit.executor import (
    CompiledPlan,
    ForwardExecutor,
    ForwardResult,
    BackwardExecutor,
)
from lucid._jit.cache import PlanCache, CacheKey
from lucid._tensor.tensor import Tensor
from lucid._backend.core import BackwardOperation
from lucid._jit.ir import IRGraph


def _collect_tensor_inputs(args: tuple, kwargs: dict) -> tuple:
    result = []
    for a in args:
        if isinstance(a, Tensor):
            result.append(a)
    for v in kwargs.values():
        if isinstance(v, Tensor):
            result.append(v)
    return tuple(result)


def _trace_and_compile(
    fn: Callable,
    args: tuple,
    kwargs: dict,
    key: CacheKey,
    param_tensors: tuple = (),
) -> CompiledPlan:
    ctx = TracingContext()
    inputs = _collect_tensor_inputs(args, kwargs)

    for t in inputs:
        ctx.register_tensor(t, is_input=True)
    for p in param_tensors:
        ctx.register_tensor(p, is_param=True)

    _trace_local.active = True
    _trace_local.tracer = ctx
    try:
        outputs = fn(*args, **kwargs)
    finally:
        _trace_local.active = False
        _trace_local.tracer = None

    outputs_tuple = outputs if isinstance(outputs, tuple) else (outputs,)
    graph = ctx.finalize(outputs_tuple)

    training = key.grad_enabled and key.training_mode
    pass_classes = DEFAULT_TRAINING_PASSES if training else DEFAULT_INFERENCE_PASSES
    graph = run_passes(graph, [p() for p in pass_classes])

    return CompiledPlan(graph, training=training)


def _attach_compiled_backward(
    outputs: tuple,
    fwd_result: ForwardResult,
    graph: IRGraph,
) -> None:
    leaf_tensors = [
        fwd_result.value_map[vid]
        for vid in fwd_result.leaf_vids
        if fwd_result.value_map.get(vid) is not None
        and getattr(fwd_result.value_map[vid], "requires_grad", False)
    ]
    leaf_refs = tuple(weakref.ref(t) for t in leaf_tensors)
    leaf_versions = tuple(t._version for t in leaf_tensors)

    for out_tensor, out_vid in zip(outputs, graph.output_ids):
        if not getattr(out_tensor, "requires_grad", False):
            continue

        out_ref = weakref.ref(out_tensor)

        def _make_closure(_out_ref, _out_vid, _fwd):
            def _closure():
                _out = _out_ref()
                if _out is None or _out.grad is None:
                    return
                retain = getattr(_out, "keep_grad", False)
                BackwardExecutor().execute(
                    _fwd,
                    output_vids=[_out_vid],
                    output_upstream_grads={_out_vid: _out.grad},
                    retain_grad=retain,
                )

            return _closure

        out_tensor._prev = leaf_tensors
        out_tensor._backward_op = BackwardOperation(
            forward_op_ref=None,
            grad_func=None,
            tensor_refs=leaf_refs,
            versions=leaf_versions,
            device=None,
            custom_closure=_make_closure(out_ref, out_vid, fwd_result),
        )


class JITFunction:
    def __init__(
        self,
        fn: Callable,
        *,
        max_cache_entries: int = 8,
    ) -> None:
        self._fn = fn
        self._cache = PlanCache(max_entries=max_cache_entries)
        self._executor = ForwardExecutor()
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        inputs = _collect_tensor_inputs(args, kwargs)
        key = CacheKey.from_inputs(
            inputs,
            grad_enabled=lucid.grad_enabled(),
            training_mode=False,
        )
        plan = self._cache.get(key)

        if plan is None:
            plan = _trace_and_compile(self._fn, args, kwargs, key)
            self._cache.put(key, plan)

        return self._run_plan(plan, inputs, {}, lucid.grad_enabled())

    def _run_plan(
        self,
        plan: CompiledPlan,
        inputs: tuple,
        param_map: dict,
        grad_enabled: bool,
    ) -> Any:
        outputs, fwd_result = self._executor.execute(
            plan, inputs, param_map, grad_enabled
        )
        if fwd_result is not None:
            _attach_compiled_backward(outputs, fwd_result, plan.graph)
        return outputs[0] if len(outputs) == 1 else outputs

    def invalidate_cache(self) -> None:
        self._cache.invalidate()

    def __repr__(self) -> str:
        return f"JITFunction({self._fn.__name__}, cache_size={len(self._cache)})"


class JITModule:
    def __init__(
        self,
        module: Any,
        *,
        max_cache_entries: int = 8,
    ) -> None:
        self._module = module
        self._cache = PlanCache(max_entries=max_cache_entries)
        self._executor = ForwardExecutor()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        module = self._module
        inputs = _collect_tensor_inputs(args, kwargs)
        key = CacheKey.from_inputs(
            inputs,
            grad_enabled=lucid.grad_enabled(),
            training_mode=module.training,
        )

        plan = self._cache.get(key)
        if plan is None:
            param_tensors = tuple(module.parameters())
            plan = _trace_and_compile(
                module.forward, args, kwargs, key, param_tensors=param_tensors
            )
            self._cache.put(key, plan)

        for hook, with_kwargs in module._forward_pre_hooks:
            if with_kwargs:
                result = hook(module, args, kwargs)
                if result is not None:
                    args, kwargs = result
            else:
                result = hook(module, args)
                if result is not None:
                    args = result

        param_map = self._build_param_map(plan.graph)
        outputs, fwd_result = self._executor.execute(
            plan, inputs, param_map, lucid.grad_enabled()
        )
        output = outputs[0] if len(outputs) == 1 else outputs

        if fwd_result is not None:
            _attach_compiled_backward(outputs, fwd_result, plan.graph)

        for hook, with_kwargs in module._forward_hooks:
            if with_kwargs:
                result = hook(module, args, kwargs, output)
            else:
                result = hook(module, args, output)
            if result is not None:
                output = result

        return output

    def _build_param_map(self, graph) -> dict:
        result = {}
        for vid in graph.param_ids:
            iv = graph.values.get(vid)
            if iv is not None and iv.live_tensor is not None:
                result[vid] = iv.live_tensor
        return result

    def invalidate_cache(self) -> None:
        self._cache.invalidate()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def __repr__(self) -> str:
        return (
            f"JITModule({type(self._module).__name__}, "
            f"cache_size={len(self._cache)})"
        )


@overload
def compile(
    target: Callable,
    *,
    max_cache_entries: int = 8,
) -> JITFunction: ...


@overload
def compile(
    target: Any,
    *,
    max_cache_entries: int = 8,
) -> JITModule: ...


def compile(
    target: Any,
    *,
    max_cache_entries: int = 8,
) -> JITFunction | JITModule:
    from lucid.nn.module import Module

    if isinstance(target, Module):
        return JITModule(target, max_cache_entries=max_cache_entries)
    elif callable(target):
        return JITFunction(target, max_cache_entries=max_cache_entries)
    else:
        raise TypeError(
            f"lucid.compile() expects an nn.Module or a callable, "
            f"got {type(target)}"
        )
