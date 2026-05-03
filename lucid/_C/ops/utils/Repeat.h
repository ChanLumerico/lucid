// lucid/_C/ops/utils/Repeat.h
//
// Declares the tiling operations repeat and tile.  Both replicate input data
// along one or more dimensions and carry autograd support for floating-point
// inputs.

#pragma once

#include <cstdint>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Repeat the contents of `a` along `axis` exactly `repeats` times.  The
// output shape is identical to the input except that dimension `axis` is
// multiplied by `repeats`.  Backward: sum the repeated gradient blocks back
// into the original shape via Dispatcher::repeat_backward.
LUCID_API TensorImplPtr repeat_op(const TensorImplPtr& a, std::int64_t repeats, int axis);

// Tile `a` according to `reps`, where reps[d] specifies how many times to
// repeat along dimension d.  If reps is longer than a's rank, a is
// conceptually left-padded with size-1 dimensions before tiling.  Backward:
// sum the tiled gradient blocks via Dispatcher::tile_backward, which uses the
// padded input shape and the repetition counts to locate each contribution.
LUCID_API TensorImplPtr tile_op(const TensorImplPtr& a, std::vector<std::int64_t> reps);

}  // namespace lucid
