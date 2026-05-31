"""Shared BatchNorm running-stats compile plumbing.

A fused-momentum BatchNorm (``track_running_stats=True`` and a finite
``momentum``) traces as a 5-input / 3-output ``batch_norm`` node: the running
buffers come in as ``inputs[3]/[4]`` and the EMA-updated stats come out as
``outputs[1]/[2]``.  The compile entry points surface those extra outputs and
write them back into the live module buffers each run so ``model.eval()`` reads
fresh stats (see [[retro-3-5-bn-runningstats-compile-writeback]]).

Two running-stats configurations CANNOT ride the graph and must be handled by
the caller (eager fallback for ``make_step`` / ``CompiledModule``; a clear error
for ``fused_step`` / ``compiled_step`` which have no fallback):

* ``track_running_stats=True`` + ``momentum=None`` (cumulative moving average) —
  the update needs ``num_batches_tracked`` as a host scalar (a GPU sync), so it
  can't be lowered into a pure graph.  :func:`model_has_cumulative_bn` detects it.

A ``track_running_stats=False`` BatchNorm, by contrast, keeps NO buffers and uses
batch statistics in both train and eval — it traces as a plain 3-input node with
nothing to write back, so it compiles unchanged.  The distinction between these
two 3-input cases is invisible in the trace IR, so the discriminator is the model
itself, not the trace.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lucid._C.engine import TensorImpl
    from lucid.nn.module import Module

__all__ = ["model_has_cumulative_bn", "model_has_tracking_bn", "bn_writeback_targets"]

_BN_NAMES: tuple[str, ...] = ("batch_norm", "batch_norm1d", "batch_norm3d")


def model_has_cumulative_bn(model: Module) -> bool:
    """Return ``True`` if any submodule keeps running stats via cumulative moving
    average (``track_running_stats=True`` and ``momentum is None``).

    Such a layer's running-stats update can't be lowered into the compiled graph
    (it reads ``num_batches_tracked`` as a host scalar), so the caller must fall
    back to eager (``make_step`` / ``CompiledModule``) or raise (``fused_step`` /
    ``compiled_step``).  Duck-typed so it also catches InstanceNorm-style layers
    without importing every norm class.
    """
    for m in model.modules():
        if getattr(m, "track_running_stats", False) and getattr(m, "momentum", 0.1) is None:
            return True
    return False


def model_has_tracking_bn(model: Module) -> bool:
    """Return ``True`` if any submodule tracks running stats
    (``track_running_stats=True``, any momentum).

    Used by the forward-only :class:`CompiledModule`: in training mode such a
    layer's EMA must advance, but the forward-only compile path has no write-back
    hook, so the caller falls back to eager.  (In eval mode BatchNorm dispatches
    the distinct ``batch_norm_eval`` op, which is unaffected.)
    """
    for m in model.modules():
        if getattr(m, "track_running_stats", False):
            return True
    return False


def bn_writeback_targets(
    graph: object, ext: dict[int, TensorImpl]
) -> list[tuple[int, int, TensorImpl]]:
    """Collect the running-stat write-back targets for every fused-momentum BN.

    One ``(feed_id, out_id, buffer_impl)`` triple is produced per running-stat
    slot (``running_mean`` then ``running_var``) of each 5-input / 3-output
    ``batch_norm`` node.  A 3-input node (``track_running_stats=False``) keeps no
    buffers and contributes nothing.

    Parameters
    ----------
    graph : TraceGraph
        The recorded trace whose ``ops`` are scanned for BatchNorm nodes.
    ext : dict[int, TensorImpl]
        The trace's external-feed map (trace id to live TensorImpl), used to
        resolve each running-stat feed id to the module buffer to write back.

    Returns
    -------
    list[tuple[int, int, TensorImpl]]
        ``(feed_id, out_id, buffer_impl)`` per slot, where ``feed_id`` is the
        trace id of the live buffer fed in (``inputs[3]/[4]``), ``out_id`` the
        EMA output (``outputs[1]/[2]``), and ``buffer_impl`` the live module
        buffer the new value must be copied/swapped into.
    """
    targets: list[tuple[int, int, TensorImpl]] = []
    ops = getattr(graph, "ops", [])
    for node in ops:
        if node.name not in _BN_NAMES:
            continue
        if len(node.inputs) >= 5 and len(node.outputs) >= 3:
            for in_idx, out_idx in ((3, 1), (4, 2)):
                feed_id = int(node.inputs[in_idx])
                impl = ext.get(feed_id)
                if impl is None:
                    continue
                targets.append((feed_id, int(node.outputs[out_idx].id), impl))
    return targets
