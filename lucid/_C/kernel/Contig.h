#pragma once

// =====================================================================
// Lucid C++ engine — kernel/Contig.h
// =====================================================================
//
// Forward-declares `contiguous_op` so kernel-layer headers can call it
// without pulling in the full `ops/utils/Contiguous.h` (which is ops/ layer).
//
// The implementation lives in `ops/utils/Contiguous.cpp` (ops/ layer).
// This header contains only the declaration, which depends on core/ only.
//
// Layer: kernel/. Depends on core/TensorImpl.h only.

#include <memory>

namespace lucid {

class TensorImpl;
using TensorImplPtr = std::shared_ptr<TensorImpl>;

/// Materialize a non-contiguous CPU tensor into a fresh contiguous buffer.
/// For contiguous tensors, returns the input unchanged (no copy).
/// GPU tensors are always assumed contiguous by the MLX contract.
TensorImplPtr contiguous_op(const TensorImplPtr& a);

}  // namespace lucid
