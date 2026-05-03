// lucid/_C/version.h
//
// Compile-time and runtime version identifiers for the Lucid engine shared
// library.  Consumers should test LUCID_ABI_VERSION at load time to detect
// incompatible binary interfaces; the three-component semantic version is
// provided for human-readable diagnostics and Python package metadata.

#pragma once

#include "api.h"

// Semantic version components.  Increment MAJOR on ABI-breaking changes,
// MINOR for backwards-compatible feature additions, PATCH for bug fixes.
#define LUCID_VERSION_MAJOR 0
#define LUCID_VERSION_MINOR 9
#define LUCID_VERSION_PATCH 0

// Human-readable version string embedded into the shared library.
#define LUCID_VERSION_STRING "0.9.0-dev"

// Monotonically increasing ABI generation counter.  Bump this whenever the
// C++ binary interface changes in a way that makes old .so / .dylib objects
// incompatible with new headers (e.g., changed struct layout, removed
// symbols, altered calling conventions).
#define LUCID_ABI_VERSION 8

namespace lucid {

// Returns LUCID_VERSION_STRING.
LUCID_API const char* version_string();

// Returns LUCID_VERSION_MAJOR.
LUCID_API int version_major();

// Returns LUCID_VERSION_MINOR.
LUCID_API int version_minor();

// Returns LUCID_VERSION_PATCH.
LUCID_API int version_patch();

// Returns LUCID_ABI_VERSION.  Python bindings compare this against the value
// baked into the extension module to catch header/library skew at import time.
LUCID_API int abi_version();

}  // namespace lucid
