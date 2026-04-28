#pragma once

// =====================================================================
// Lucid C++ engine — forward declarations.
// =====================================================================
//
// Include this *instead of* the matching full header when you only need a
// pointer or reference to the type. Reduces compile time and breaks circular
// dependencies. Pair with the full header in the corresponding .cpp.
//
// Layering rule (enforced by code review and `docs/ARCHITECTURE.md`):
//
//     core/  ←  autograd/  ←  backend/  ←  optim/  ←  jit/
//                                    ↖                ↗
//                                     bindings/
//
// "←" means "depends on". Reverse arrows are forbidden — never #include from
// a higher layer in a lower layer's header.

#include <cstddef>
#include <cstdint>
#include <memory>

namespace lucid {

// core/
enum class Device : std::uint8_t;
enum class Dtype : std::uint8_t;

class TensorImpl;
class Generator;

class LucidError;

struct CpuStorage;
struct GpuStorage;
struct MemoryStats;

// autograd/
class Node;
struct Edge;
class AccumulateGrad;
class Engine;

// Convenience aliases — preferred parameter types over raw shared_ptr.
using TensorImplPtr = std::shared_ptr<TensorImpl>;
using NodePtr = std::shared_ptr<Node>;

}  // namespace lucid
