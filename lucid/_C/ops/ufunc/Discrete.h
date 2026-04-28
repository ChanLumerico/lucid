#pragma once

// =====================================================================
// Lucid C++ engine — discrete / non-differentiable unary ops.
// =====================================================================
//
// All four ops have zero gradient (or no gradient at all on integer inputs).
// kHasGradient = false on the CRTP base skips graph wiring entirely.
//
//   round(x)   — banker's rounding (half-to-even, matches np.round)
//   floor(x)
//   ceil(x)
//   invert(x)  — bitwise NOT (integer dtypes)

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

#define LUCID_DECLARE_DISCRETE(CLASS, FN)                                                    \
    class LUCID_API CLASS##Backward : public UnaryOp<CLASS##Backward> {                      \
    public:                                                                                  \
        static constexpr bool kSavesInput = false;                                           \
        static constexpr bool kHasGradient = false;                                          \
        static const OpSchema schema_v1;                                                     \
        static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt); \
        static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt); \
        Storage grad_formula(const Storage& g);                                              \
    };                                                                                       \
    LUCID_API TensorImplPtr FN##_op(const TensorImplPtr& a);

LUCID_DECLARE_DISCRETE(Round, round)
LUCID_DECLARE_DISCRETE(Floor, floor)
LUCID_DECLARE_DISCRETE(Ceil, ceil)
LUCID_DECLARE_DISCRETE(Invert, invert)

#undef LUCID_DECLARE_DISCRETE

}  // namespace lucid
