#pragma once

#include "api.h"

#define LUCID_VERSION_MAJOR 0
#define LUCID_VERSION_MINOR 9
#define LUCID_VERSION_PATCH 0

#define LUCID_VERSION_STRING "0.9.0-dev"

#define LUCID_ABI_VERSION 8

namespace lucid {

LUCID_API const char* version_string();

LUCID_API int version_major();

LUCID_API int version_minor();

LUCID_API int version_patch();

LUCID_API int abi_version();

}  // namespace lucid
