// lucid/_C/core/fwd.h
//
// Forward declarations for the core types used across the engine.  Including
// this header in place of the full type headers reduces compilation times and
// breaks include cycles.  Translation units that need the full definitions
// (e.g. to call member functions or know the size of a type) must still
// include the corresponding .h file directly.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace lucid {

// Core enumerations — their underlying type is known so pointers/references
// to them can be formed without including the full enum definition.
enum class Device : std::uint8_t;
enum class Dtype : std::uint8_t;

// Tensor implementation and random-number generator.
class TensorImpl;
class Generator;

// Base exception type — forward-declared so error-handling headers need not
// pull in the full Error hierarchy.
class LucidError;

// Storage variants used by TensorImpl.
struct CpuStorage;
struct GpuStorage;
struct MemoryStats;

// Autograd graph nodes.
class Node;
struct Edge;
class AccumulateGrad;
class Engine;

// Shared-pointer aliases used pervasively in the autograd and op layers.
using TensorImplPtr = std::shared_ptr<TensorImpl>;
using NodePtr = std::shared_ptr<Node>;

}  // namespace lucid
