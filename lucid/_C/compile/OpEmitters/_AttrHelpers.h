// lucid/_C/compile/OpEmitters/_AttrHelpers.h
//
// Shared inline helpers for emitter ``.mm`` files: typed accessors for
// the :type:`AttributeValue` variant on :class:`OpNode`, plus a small
// utility to build an ``NSArray<NSNumber*>*`` axis list from a
// :class:`Shape`.
//
// Each .mm in a sub-package (core/ linalg/ shape/ reduce/ nn/ misc/)
// includes this header to avoid duplicating the same five getters at
// the top of every file.  Marked ``inline`` so a copy is emitted in
// each translation unit and the linker picks one — no ODR conflicts.
//
// Leading underscore in the filename keeps the header sorted to the
// top of directory listings and signals "internal to this directory"
// to anyone browsing the tree.
//
// Header is pure C++/Objective-C++ — safe to include from ``.mm``
// files but not from plain ``.cpp`` (since it touches NSArray /
// NSNumber).

#pragma once

#import <Foundation/Foundation.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <variant>
#include <vector>

#include "../TraceIR.h"
#include "OpEmitter.h"

namespace lucid::compile {

// Pull an ``int64_t`` attribute from ``node`` by key, falling back to
// ``def`` if the key is absent or the stored variant alternative
// doesn't match.  Used for axis / dim / kernel_size / padding / etc.
inline std::int64_t int_attr(const OpNode& node, const char* key, std::int64_t def) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end())
        return def;
    const auto* p = std::get_if<std::int64_t>(&it->second);
    return p ? *p : def;
}

// Pull a ``bool`` attribute (e.g. ``keepdim``, ``align_corners``).
inline bool bool_attr(const OpNode& node, const char* key, bool def) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end())
        return def;
    const auto* p = std::get_if<bool>(&it->second);
    return p ? *p : def;
}

// Pull a ``double`` attribute (e.g. ``eps``, ``ord``, ``scale``).
[[maybe_unused]] inline double double_attr(const OpNode& node, const char* key, double def) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end())
        return def;
    const auto* p = std::get_if<double>(&it->second);
    return p ? *p : def;
}

// Pull a ``vector<int64_t>`` attribute (stride / padding / dilation /
// kernel_size / axis_list).  Returns ``nullptr`` if absent or the
// stored alternative isn't a vector — caller decides whether to
// fall back to a default or bail out.
[[maybe_unused]] inline const std::vector<std::int64_t>* int_vec_attr(const OpNode& node,
                                                                      const char* key) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end())
        return nullptr;
    return std::get_if<std::vector<std::int64_t>>(&it->second);
}

// Build ``[0, 1, …, rank-1]`` as an Objective-C array of NSNumber.
// Used for "reduce over every axis" code paths.
[[maybe_unused]] inline NSArray<NSNumber*>* shape_to_axes(const Shape& s) {
    NSMutableArray<NSNumber*>* a = [NSMutableArray arrayWithCapacity:s.size()];
    for (std::size_t d = 0; d < s.size(); ++d)
        [a addObject:[NSNumber numberWithLongLong:(long long)d]];
    return a;
}

// ── Dynamic-batch shape helpers ──────────────────────────────────────
//
// Under ``dynamic_batch`` (lucid.compile(..., dynamic=True) with the symbolic
// opt-in), MpsBuilder marks the LEADING dim (dim 0) of every user-input
// placeholder -1 so one executable serves any batch size.  That -1 must keep
// flowing through the graph; the *view* ops (reshape / flatten / squeeze /
// contiguous / reduce-squeeze) copy a target shape verbatim from the trace,
// pinning the batch to its concrete trace-time value — which mismatches the
// symbolic input and aborts MPSGraph's MLIR pass.  ``reshape_dynamic_aware``
// re-introduces the symbolic dim, but ONLY when the view provably keeps the
// batch at dim 0.

// True if ``t``'s leading (batch) dim is the symbolic (-1) one.  Only dim 0 is
// ever symbolised (MpsBuilder), so this is the precise dynamic-batch predicate
// — checking dim 0 specifically, NOT "any dim is -1".
[[maybe_unused]] inline bool symbolic_batch_at_dim0(MPSGraphTensor* t) {
    return t != nil && t.shape != nil && t.shape.count > 0 &&
           t.shape[0].longLongValue < 0;
}

// Product of all dims EXCEPT the leading one (the "non-batch" numel).
[[maybe_unused]] inline long long trailing_numel(NSArray<NSNumber*>* s) {
    long long p = 1;
    for (NSUInteger i = 1; i < s.count; ++i)
        p *= s[i].longLongValue;
    return p;
}
[[maybe_unused]] inline long long trailing_numel(const Shape& s) {
    long long p = 1;
    for (std::size_t i = 1; i < s.size(); ++i)
        p *= s[i];
    return p;
}

// Reshape ``x_t`` to the trace's concrete ``out_shape``.
//
// When ``x_t`` has a symbolic batch (leading dim -1) AND the view keeps the
// batch at dim 0, mark dim 0 of the target -1 so MPSGraph INFERS it at runtime
// instead of pinning the trace-time batch.  "Keeps the batch at dim 0" is
// proven conservatively by ``trailing_numel(in) == trailing_numel(out)``: equal
// non-leading numel forces ``out_dim0 == in_dim0 == batch`` (the view only
// restructures the non-batch part — flatten, attention head-split / merge,
// reduce-squeeze all satisfy this).  MPSGraph allows exactly one inferred (-1)
// dim, so the remaining concrete dims determine it uniquely.
//
// Returns **nil** when ``x_t`` is symbolic-batch but the view does NOT provably
// preserve the batch at dim 0 — e.g. ``unsqueeze(0)`` (batch slides to dim 1) or
// a transpose-then-reshape, and also batch-merging views like
// ``(B,T,E)->(B*T,E)`` which we conservatively reject.  Such a view cannot be
// represented under a dim-0 symbolic batch, so the caller MUST fall back
// (``return false`` → per-shape / eager) rather than emit a mis-shaped graph
// that would silently produce wrong numerics at non-trace batch sizes.
//
// Static (non-dynamic) compiles are unaffected — ``symbolic_batch_at_dim0`` is
// false, so ``out_shape`` is used verbatim, identical to the prior behaviour.
[[maybe_unused]] inline MPSGraphTensor* reshape_dynamic_aware(
    MPSGraph* graph, MPSGraphTensor* x_t, const Shape& out_shape, NSString* name) {
    NSMutableArray<NSNumber*>* tgt = [NSMutableArray arrayWithCapacity:out_shape.size()];
    for (std::int64_t d : out_shape)
        [tgt addObject:[NSNumber numberWithLongLong:d]];
    if (symbolic_batch_at_dim0(x_t)) {
        if (tgt.count == 0 || trailing_numel(x_t.shape) != trailing_numel(out_shape))
            return nil;  // batch not provably at dim 0 → caller falls back
        tgt[0] = @(-1);
    }
    return [graph reshapeTensor:x_t withShape:tgt name:name];
}

}  // namespace lucid::compile
