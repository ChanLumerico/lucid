#pragma once

// =====================================================================
// Lucid C++ engine version.
// =====================================================================
//
// Semantic versioning (semver.org):
//   MAJOR — incremented on incompatible C ABI / op-schema breaking changes
//   MINOR — incremented on backward-compatible new ops, new public APIs
//   PATCH — incremented on backward-compatible bug fixes only
//
// A separate ABI version is exposed for the Phase 7 C ABI: that one tracks
// only the binary interface and increments more rarely.

#include "api.h"

#define LUCID_VERSION_MAJOR 0
#define LUCID_VERSION_MINOR 9
#define LUCID_VERSION_PATCH 0

#define LUCID_VERSION_STRING "0.9.0-dev"

// ABI version — bump on binary layout changes (TensorImpl fields, Storage
// variant order, Node vtable). History:
//   1  Phases 0–2   TensorImpl encapsulation
//   2  Phase 3      Kernel framework (BinaryKernel, UnaryKernel, …)
//   3  Phase 4      IBackend / Dispatcher abstraction
//   4  Phase 5      SchemaGuard / AmpPolicy in OpSchema
//   5  Phase 6      OpSchema.internal flag, BindingGen
//   6  Phase 7      linalg backward nodes (NormBackward, InvBackward, …)
//   7  Phase 8      kSavesInput=false flags in SumBackward/MeanBackward
//   8  Phase 9      Node::release_saved(); OpRegistry shared_mutex;
//                   thread-local allocator pool
#define LUCID_ABI_VERSION 8

namespace lucid {

/// Version string.
LUCID_API const char* version_string();
/// Version major.
LUCID_API int version_major();
/// Version minor.
LUCID_API int version_minor();
/// Version patch.
LUCID_API int version_patch();
/// Abi version.
LUCID_API int abi_version();

}  // namespace lucid
