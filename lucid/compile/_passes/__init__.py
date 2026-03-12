"""
lucid.compile._passes
---------------------
Optimisation passes that transform an :class:`~lucid.compile._ir.IRGraph`
into a semantically equivalent but more efficient representation.

Available passes
~~~~~~~~~~~~~~~~
.. list-table::
   :widths: 30 70

   * - :func:`dead_code_elimination`
     - Remove nodes that are not reachable from the declared outputs.
   * - :func:`constant_folding`
     - Pre-evaluate sub-graphs whose all inputs are frozen constants.
   * - :func:`operator_fusion`
     - Merge compatible producer–consumer op pairs (e.g. linear+relu).

Pass pipeline
~~~~~~~~~~~~~
:func:`run_default_passes` applies the passes in the recommended order for the
``"default"`` compile mode.  Higher-level modes add constant folding on top.
"""

from lucid.compile._passes.dead_code_elimination import dead_code_elimination
from lucid.compile._passes.constant_folding import constant_folding
from lucid.compile._passes.operator_fusion import operator_fusion, register_fusion_rule

from lucid.compile._ir import IRGraph

__all__ = [
    "dead_code_elimination",
    "constant_folding",
    "operator_fusion",
    "register_fusion_rule",
    "run_default_passes",
    "run_max_passes",
]


def run_default_passes(graph: IRGraph) -> IRGraph:
    """Apply the standard optimisation pipeline (DCE → operator fusion → DCE).

    This pipeline is safe to run at every compilation without risk of
    incorrect results.  It does *not* fold constants, which requires the
    model to be in eval mode.

    Parameters
    ----------
    graph:
        Source graph (not mutated).

    Returns
    -------
    IRGraph
        Optimised graph.
    """
    graph = dead_code_elimination(graph)
    graph = operator_fusion(graph)
    graph = dead_code_elimination(graph)
    return graph


def run_max_passes(graph: IRGraph) -> IRGraph:
    """Apply the full optimisation pipeline including constant folding.

    Intended for inference-only workloads where weights are frozen.
    Running this during training may produce stale folded constants if
    weights change between compilations.

    Parameters
    ----------
    graph:
        Source graph (not mutated).

    Returns
    -------
    IRGraph
        Maximally optimised graph.
    """
    graph = dead_code_elimination(graph)
    graph = constant_folding(graph)
    graph = operator_fusion(graph)
    graph = dead_code_elimination(graph)
    return graph
