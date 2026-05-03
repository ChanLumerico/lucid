// lucid/_C/version.cpp
//
// Thin implementation file that exposes the preprocessor version constants as
// exported runtime functions.  Keeping the definitions here (rather than
// inline in the header) ensures that the values embedded in the shared library
// binary are authoritative — callers that link against the .so always obtain
// the library's own version rather than whatever was compiled into the caller.

#include "version.h"

namespace lucid {

const char* version_string() {
    return LUCID_VERSION_STRING;
}
int version_major() {
    return LUCID_VERSION_MAJOR;
}
int version_minor() {
    return LUCID_VERSION_MINOR;
}
int version_patch() {
    return LUCID_VERSION_PATCH;
}
int abi_version() {
    return LUCID_ABI_VERSION;
}

}  // namespace lucid
