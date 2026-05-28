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

}  // namespace lucid::compile
