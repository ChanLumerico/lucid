#pragma once

#include <cstring>

#include "../../core/Allocator.h"
#include "../../core/Helpers.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/fwd.h"

namespace lucid::ufunc_detail {

using ::lucid::helpers::allocate_cpu;
using ::lucid::helpers::fresh;

}  // namespace lucid::ufunc_detail
