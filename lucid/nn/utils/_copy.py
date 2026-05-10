"""
lucid.nn.utils.copy_parameters_and_buffers — synchronise a destination
module's parameter and buffer values to match a source module.

Useful for utilities such as EMA, model averaging, and weight mirroring
between models that share the same architecture but live in different
training contexts.
"""

from lucid.nn.module import Module


def copy_parameters_and_buffers(source: Module, dest: Module) -> None:
    """Copy all parameter / buffer values from ``source`` into ``dest``.

    Both modules must expose the same set of parameter and buffer names;
    a ``KeyError`` is raised if a name found on ``source`` is missing on
    ``dest``.  Tensor data is copied in-place via ``Tensor.detach()`` and
    storage assignment so that ``dest``'s autograd graph (if any) is not
    disturbed by the copy.

    Parameters
    ----------
    source : Module
        Module whose parameters / buffers provide the values to copy.
    dest : Module
        Module whose parameters / buffers receive the copied values.

    Notes
    -----
    Shape and dtype of corresponding entries must match.  Devices may
    differ — the source values are routed through ``.to(device)`` to
    land on ``dest``'s device first.
    """
    src_params = dict(source.named_parameters())
    dst_params = dict(dest.named_parameters())
    for name, p_src in src_params.items():
        if name not in dst_params:
            raise KeyError(
                f"copy_parameters_and_buffers: parameter {name!r} present "
                f"on source but missing on dest"
            )
        p_dst = dst_params[name]
        if p_dst.shape != p_src.shape:
            raise ValueError(
                f"copy_parameters_and_buffers: parameter {name!r} shape "
                f"mismatch (src={p_src.shape}, dst={p_dst.shape})"
            )
        # Move source to the destination's device, then write into the
        # destination's storage in place.
        src_on_dst = p_src.to(device=p_dst.device).detach()
        p_dst._impl.copy_from(src_on_dst._impl)

    src_buffers = dict(source.named_buffers())
    dst_buffers = dict(dest.named_buffers())
    for name, b_src in src_buffers.items():
        if name not in dst_buffers:
            raise KeyError(
                f"copy_parameters_and_buffers: buffer {name!r} present "
                f"on source but missing on dest"
            )
        b_dst = dst_buffers[name]
        if b_dst.shape != b_src.shape:
            raise ValueError(
                f"copy_parameters_and_buffers: buffer {name!r} shape "
                f"mismatch (src={b_src.shape}, dst={b_dst.shape})"
            )
        src_on_dst = b_src.to(device=b_dst.device).detach()
        b_dst._impl.copy_from(src_on_dst._impl)
