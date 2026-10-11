// lucid/_C/compile/OpEmitters/_SignedZero.h
//
// Where a zero constant meets an addition inside a compiled graph (LCD-446).
//
// MPSGraph rewrites ``x + c`` and ``x - c`` to ``x`` whenever ``c`` is a
// constant splat of either zero — at every optimization level, before anything
// runs.  IEEE 754 agrees only for the zero that is the identity: ``x + (-0)``
// and ``x - (+0)`` are ``x`` for every ``x``, but ``(-0) + (+0)`` and
// ``(-0) - (-0)`` are ``+0``, so the rewrite hands back ``-0`` where eager
// gives ``+0``.  ``0 - x``, ``x * 1``, ``x * 0`` and ``x / 1`` were measured
// too and MPSGraph keeps them exact.
//
// A trace's zero constants come from the ``full`` / ``zeros`` factories (a
// Python scalar operand is materialised by ``full``).  Those emitters mark a
// floating ±0 fill in the builder's saved-state table, keyed by their output
// id under slot names no VJP uses; the add and sub emitters look the mark up
// and, when the operand is the zero that is not the identity, spell the sum as
// ``x + min(|x|, +0)``.  The second term is +0 for every non-NaN ``x`` but is
// not a constant, so MPSGraph has nothing to fold and the hardware addition
// gives IEEE's answer.  A select on ``x == 0`` would be one op cheaper, but
// MPSGraph's autodiff — the fallback when a manual VJP is missing — returns a
// wrong gradient through it.
//
// A zero that reaches the add through another traced op (an ``astype``, a
// ``reshape``) carries no mark and still folds.
//
// Objective-C++ only.

#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>

#include "../../core/Dtype.h"
#include "../TraceIR.h"
#include "OpEmitter.h"

namespace lucid::compile {

inline constexpr const char* kPosZeroSplatSlot = "ieee_pos_zero_splat";
inline constexpr const char* kNegZeroSplatSlot = "ieee_neg_zero_splat";

// Record that ``out_id`` is a splat of ``fill``, if that is a floating zero.
inline void mark_zero_splat(
    BuilderContext& ctx, TensorId out_id, Dtype dtype, double fill, MPSGraphTensor* splat) {
    if (!is_floating_point(dtype) || fill != 0.0)
        return;
    ctx.stash_saved(out_id, std::signbit(fill) ? kNegZeroSplatSlot : kPosZeroSplatSlot,
                    (__bridge void*)splat);
}

inline bool is_pos_zero_splat(const BuilderContext& ctx, TensorId id) {
    return ctx.resolve_saved(id, kPosZeroSplatSlot) != nullptr;
}

inline bool is_neg_zero_splat(const BuilderContext& ctx, TensorId id) {
    return ctx.resolve_saved(id, kNegZeroSplatSlot) != nullptr;
}

// ``x + pos_zero`` as IEEE computes it, where ``pos_zero`` is a +0 splat (it
// also carries the broadcast shape of the sum).
inline MPSGraphTensor* add_pos_zero(MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* pos_zero) {
    MPSGraphTensor* zero_unless_nan = [g minimumWithPrimaryTensor:[g absoluteWithTensor:x name:nil]
                                                  secondaryTensor:pos_zero
                                                             name:nil];
    return [g additionWithPrimaryTensor:x secondaryTensor:zero_unless_nan name:@"add_pos_zero"];
}

}  // namespace lucid::compile
