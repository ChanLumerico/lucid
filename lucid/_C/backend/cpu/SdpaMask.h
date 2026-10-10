// lucid/_C/backend/cpu/SdpaMask.h
//
// How the CPU attention kernel reads its mask, worked out from the mask
// storage itself.
//
// The mask is the one attention operand whose dtype and extent the scores do
// not fix: it is a bool keep-mask or an additive one, and it covers one score
// slice or every slice.  The kernel used to take both from its caller's
// ``mask_dtype`` / ``mask_numel`` arguments.  So a half mask that had not
// been widened with q/k/v was read as float, and a mask that broadcast over
// only some leading axes was read past its end.  The storage knows both
// facts, so this reads them from the storage and refuses anything the kernel
// cannot read.

#pragma once

#include <cstddef>
#include <string>

#include "../../core/Dtype.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Storage.h"

namespace lucid::backend::cpu {

struct SdpaMask {
    // A bool keep-mask (``true`` attends) rather than an additive one.
    bool keep = false;
    // Elements between consecutive score slices: ``0`` when one slice is
    // shared by every slice.
    std::size_t slice_step = 0;
};

// ``slice`` is ``Lq * Lk``, ``slices`` the flattened leading extent.
//
// Raises
// ------
// DtypeMismatch
//     An additive mask whose dtype is not the scores' ``dt``.
// InvalidArgument
//     A mask holding neither one slice nor all of them.
inline SdpaMask
sdpa_mask_layout(const CpuStorage& mask, Dtype dt, std::size_t slice, std::size_t slices) {
    SdpaMask layout;
    layout.keep = mask.dtype == Dtype::Bool;
    if (!layout.keep && mask.dtype != dt)
        ErrorBuilder("cpu_backend::sdpa").dtype_mismatch(dt, mask.dtype, "additive attn_mask");
    const std::size_t n = mask.nbytes / dtype_size(mask.dtype);
    if (n == slice)
        return layout;
    if (n != slice * slices)
        ErrorBuilder("cpu_backend::sdpa")
            .invalid_argument("attn_mask has " + std::to_string(n) +
                              " elements; the CPU kernel takes one score slice (" +
                              std::to_string(slice) + ") or every slice (" +
                              std::to_string(slice * slices) + ")");
    layout.slice_step = slice;
    return layout;
}

}  // namespace lucid::backend::cpu
