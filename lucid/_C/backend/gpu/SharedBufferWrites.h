// lucid/_C/backend/gpu/SharedBufferWrites.h
//
// Ordering a host write into a Metal shared buffer after the GPU work that
// reads it.
//
// A Metal shared buffer is the one MLX buffer Lucid writes in place: every
// other MLX array is a value nothing mutates.  An MLX op records a graph node
// and runs when a value is read, so a result built over the buffer before a
// write would otherwise read the bytes the write left there.  Kept apart from
// MlxBridge.h, which some forty translation units include, for the one caller
// in core that writes into shared buffers.

#pragma once

#include <vector>

#include <mlx/array.h>

#include "../../api.h"
#include "../../core/Storage.h"

namespace lucid::gpu {

// Run every pending GPU array whose graph reads ``sh``'s bytes, then wait for
// the device, so a host write into ``sh`` that follows changes none of them.
//
// Parameters
// ----------
// sh : const SharedStorage&
//     The buffer about to be written.
// with : std::vector<mlx::core::array>, optional
//     Arrays to run in the same pass — the value about to be written, which
//     would otherwise cost a second trip to the device.
//
// Notes
// -----
// A pending array reads the buffer when its graph reaches an array that has
// run and holds the buffer's ``array::Data``: the pinned alias
// (``sh.gpu_alias``) and whatever MLX derived from it without a copy
// (reshape, slice, contiguous).  Only arrays :func:`wrap_mlx_array` noted as
// pending are roots of the search, so the cost is one walk over the pending
// graphs plus the evaluation of the readers it finds; other pending work stays
// lazy.  A result that already ran and shares the buffer's bytes is an alias,
// not a reader, and sees the write.
LUCID_API void settle_buffer_readers(const SharedStorage& sh,
                                     std::vector<::mlx::core::array> with = {});

}  // namespace lucid::gpu
