#include "version.h"

namespace lucid {

const char* version_string() { return LUCID_VERSION_STRING; }
int version_major() { return LUCID_VERSION_MAJOR; }
int version_minor() { return LUCID_VERSION_MINOR; }
int version_patch() { return LUCID_VERSION_PATCH; }
int abi_version()   { return LUCID_ABI_VERSION; }

}  // namespace lucid
