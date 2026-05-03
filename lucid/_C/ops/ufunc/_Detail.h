// lucid/_C/ops/ufunc/_Detail.h
//
// Internal utilities shared by hand-written ufunc implementations (Var, Trace,
// Scan) that do not go through the standard UnaryOp/ReduceOp CRTP templates.
// These helpers belong in the `lucid::ufunc_detail` namespace so accidental
// name collisions with the top-level `lucid` namespace are impossible.
//
// Do not include this header from public-facing headers; it is intended only
// for ufunc .cpp translation units.

#pragma once

#include <cstring>

#include "../../core/Allocator.h"
#include "../../core/Helpers.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/fwd.h"

namespace lucid::ufunc_detail {

// Bring helpers::allocate_cpu and helpers::fresh into this namespace so that
// local using-declarations in .cpp files stay concise.
using ::lucid::helpers::allocate_cpu;
using ::lucid::helpers::fresh;

}  // namespace lucid::ufunc_detail
