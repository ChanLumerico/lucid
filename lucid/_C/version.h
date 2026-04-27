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
#define LUCID_VERSION_MINOR 0
#define LUCID_VERSION_PATCH 1

#define LUCID_VERSION_STRING "0.0.1-dev"

#define LUCID_ABI_VERSION   1

namespace lucid {

LUCID_API const char* version_string();
LUCID_API int version_major();
LUCID_API int version_minor();
LUCID_API int version_patch();
LUCID_API int abi_version();

}  // namespace lucid
