// lucid/_C/api.h
//
// Portability macros that control symbol visibility and a small set of
// utility macros used throughout the engine headers.  Every public type or
// function that must be callable from outside the shared library should be
// annotated with LUCID_API.  Implementation-private symbols should use
// LUCID_INTERNAL to prevent them from leaking into the dynamic symbol table,
// which improves load-time performance and avoids accidental ABI coupling.

#pragma once

// ---------------------------------------------------------------------------
// Platform-specific import / export annotations
// ---------------------------------------------------------------------------

#if defined(_WIN32) || defined(__CYGWIN__)

#define LUCID_API_EXPORT __declspec(dllexport)
#define LUCID_API_IMPORT __declspec(dllimport)
#define LUCID_API_LOCAL
#else
// GCC/Clang: "default" visibility makes the symbol visible in the dynamic
// symbol table; "hidden" keeps it local to the shared object.
#define LUCID_API_EXPORT __attribute__((visibility("default")))
#define LUCID_API_IMPORT __attribute__((visibility("default")))
#define LUCID_API_LOCAL __attribute__((visibility("hidden")))
#endif

// When the engine shared library itself is being built (i.e. the CMake target
// sets -DLUCID_BUILDING_ENGINE), we export symbols; otherwise we import them.
#if defined(LUCID_BUILDING_ENGINE)
#define LUCID_API LUCID_API_EXPORT
#else
#define LUCID_API LUCID_API_IMPORT
#endif

// Mark a symbol as engine-internal (hidden from the dynamic symbol table).
#define LUCID_INTERNAL LUCID_API_LOCAL

// ---------------------------------------------------------------------------
// Convenience macros for deleted copy / move operations
// ---------------------------------------------------------------------------

// Deletes the copy constructor and copy-assignment operator for Type.
// Use in the public section of non-copyable classes.
#define LUCID_NOCOPY(Type)                                                                         \
    Type(const Type&) = delete;                                                                    \
    Type& operator=(const Type&) = delete

// Deletes the move constructor and move-assignment operator for Type.
// Use alongside LUCID_NOCOPY for fully immovable types (e.g., mutex holders).
#define LUCID_NOMOVE(Type)                                                                         \
    Type(Type&&) = delete;                                                                         \
    Type& operator=(Type&&) = delete
