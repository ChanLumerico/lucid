#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace lucid {

enum class Device : std::uint8_t;
enum class Dtype : std::uint8_t;

class TensorImpl;
class Generator;

class LucidError;

struct CpuStorage;
struct GpuStorage;
struct MemoryStats;

class Node;
struct Edge;
class AccumulateGrad;
class Engine;

using TensorImplPtr = std::shared_ptr<TensorImpl>;
using NodePtr = std::shared_ptr<Node>;

}  // namespace lucid
