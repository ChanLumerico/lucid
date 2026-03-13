from abc import ABC, abstractmethod

from lucid._jit.ir import IRGraph


class Pass(ABC):
    name: str = ""

    @abstractmethod
    def run(self, graph: IRGraph) -> IRGraph: ...


class DeadNodeElimPass(Pass):
    name = "dead_node_elim"

    def run(self, graph: IRGraph) -> IRGraph:
        live: set[int] = set(graph.output_ids)
        for node in reversed(graph.nodes):
            if any(vid in live for vid in node.output_ids):
                live.update(node.input_ids)

        graph.nodes = [
            n for n in graph.nodes if any(vid in live for vid in n.output_ids)
        ]
        return graph


class NoGradStripPass(Pass):
    name = "no_grad_strip"

    def run(self, graph: IRGraph) -> IRGraph:
        for node in graph.nodes:
            node.has_gradient = False
        return graph


def run_passes(graph: IRGraph, passes: list[Pass]) -> IRGraph:
    for p in passes:
        graph = p.run(graph)
    return graph


DEFAULT_INFERENCE_PASSES: list[type[Pass]] = [DeadNodeElimPass, NoGradStripPass]
DEFAULT_TRAINING_PASSES: list[type[Pass]] = [DeadNodeElimPass]
