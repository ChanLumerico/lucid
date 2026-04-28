#pragma once

// =====================================================================
// ufunc internal helpers — shared by Var/Trace/Scan and any future ufunc
// additions. Header-only inline functions. Not user-facing.
// =====================================================================

#include <cstring>

#include "../../core/Allocator.h"
#include "../../core/Helpers.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/fwd.h"

namespace lucid::ufunc_detail {

// Re-exports of the canonical helpers in `core/Helpers.h`.
using ::lucid::helpers::allocate_cpu;
using ::lucid::helpers::fresh;

}  // namespace lucid::ufunc_detail
