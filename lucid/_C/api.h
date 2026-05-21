// lucid/_C/api.h
//
// Cross-platform symbol-visibility macros that gate the engine's ABI.
//
// Every type or function intended to be callable from outside the
// shared library (i.e. from the pybind11 extension module or from
// downstream C++ consumers linking against the engine) is annotated
// with ``LUCID_API``; symbols meant to remain private to the
// translation unit or the shared object carry ``LUCID_INTERNAL``.
// Restricting the exported set keeps the dynamic symbol table small,
// speeds up dynamic-linker fixups at process start, and prevents
// downstream code from accidentally taking a hard dependency on
// implementation-private entry points.
//
// Notes
// -----
// Although the export / import branches below cover the three major
// desktop toolchains, Lucid only officially ships for macOS on
// Apple Silicon (arm64).  The Windows/Cygwin branch is retained for
// future portability but is not exercised by the release pipeline,
// and the Linux/Clang branch is used purely for the engine's CI
// smoke build — neither is a supported user platform.
//
// See Also
// --------
// :file:`lucid/_C/version.h` — ABI generation counter consumed by the
//     Python bindings at import time to detect header / library skew.

#pragma once

// ---------------------------------------------------------------------------
// Platform-specific import / export annotations
// ---------------------------------------------------------------------------

#if defined(_WIN32) || defined(__CYGWIN__)

// Marks a symbol as exported from the engine DLL on Windows / Cygwin.
//
// Expands to ``__declspec(dllexport)``.  Active when the engine is
// being built as a DLL — the linker emits the symbol into the export
// table and any client linking against the import library can call it.
//
// Notes
// -----
// macOS is the only officially supported platform; this branch exists
// for portability but is not exercised by the release pipeline.
#define LUCID_API_EXPORT __declspec(dllexport)

// Marks a symbol as imported from the engine DLL on Windows / Cygwin.
//
// Expands to ``__declspec(dllimport)``.  Active in client builds —
// signals that the symbol lives in another module and instructs the
// compiler to emit an indirection through the import address table.
#define LUCID_API_IMPORT __declspec(dllimport)

// No-op on Windows — DLL symbols default to local unless exported.
//
// Provided for parity with the Unix branch so a single ``LUCID_INTERNAL``
// macro works across platforms.
#define LUCID_API_LOCAL
#else
// GCC/Clang: "default" visibility makes the symbol visible in the dynamic
// symbol table; "hidden" keeps it local to the shared object.

// Marks a symbol as exported from the engine shared object on
// macOS / Linux (Clang / GCC).
//
// Expands to ``__attribute__((visibility("default")))``, which places
// the symbol into the dynamic symbol table so it can be resolved
// across the ``dlopen``-loaded boundary between the Python extension
// module and the engine ``.dylib`` / ``.so``.
//
// Notes
// -----
// Lucid builds with ``-fvisibility=hidden`` by default, so only
// symbols explicitly annotated with ``LUCID_API`` (and therefore this
// macro on import-side) are exposed.
#define LUCID_API_EXPORT __attribute__((visibility("default")))

// Marks a symbol as imported from the engine shared object on
// macOS / Linux.
//
// Expands to the same ``visibility("default")`` attribute as
// :macro:`LUCID_API_EXPORT`; Unix link semantics treat both sides
// symmetrically, unlike Windows which distinguishes import / export.
#define LUCID_API_IMPORT __attribute__((visibility("default")))

// Marks a symbol as engine-private (hidden from the dynamic symbol
// table) on macOS / Linux.
//
// Expands to ``__attribute__((visibility("hidden")))``.  Helper
// functions, anonymous-namespace utilities, and translation-unit-
// local templates instantiated in the engine carry this attribute
// (directly or via :macro:`LUCID_INTERNAL`) so they cannot accidentally
// be picked up by downstream linkers.
#define LUCID_API_LOCAL __attribute__((visibility("hidden")))
#endif

// When the engine shared library itself is being built (i.e. the CMake target
// sets -DLUCID_BUILDING_ENGINE), we export symbols; otherwise we import them.

// Public ABI annotation: expands to export when building the engine,
// import when consumed from a client translation unit.
//
// Decided at compile time by the presence of ``LUCID_BUILDING_ENGINE``,
// which the CMake target sets exclusively while compiling the engine's
// own sources.  Every header-visible function or class that crosses
// the shared-object boundary must be annotated with this macro;
// forgetting it produces a link error in the pybind11 extension
// because the symbol stays hidden.
//
// Examples
// --------
// Annotating a public free function::
//
//     LUCID_API int abi_version();
//
// Annotating a public class::
//
//     class LUCID_API Generator { ... };
//
// See Also
// --------
// :macro:`LUCID_INTERNAL` — opposite annotation for engine-private
//     symbols.
#if defined(LUCID_BUILDING_ENGINE)
#define LUCID_API LUCID_API_EXPORT
#else
#define LUCID_API LUCID_API_IMPORT
#endif

// Mark a symbol as engine-internal (hidden from the dynamic symbol table).

// Engine-private annotation: forces the symbol to ``hidden`` visibility.
//
// Expands to :macro:`LUCID_API_LOCAL`.  Use on helpers that must be
// non-static (because they are referenced from multiple translation
// units within the engine) but that should not be part of the public
// ABI.  Hidden symbols are not callable from the pybind11 layer or
// from downstream C++ code that links against the engine.
//
// Notes
// -----
// Prefer plain ``static`` or an anonymous namespace when a helper is
// only used in a single ``.cpp`` file — ``LUCID_INTERNAL`` is reserved
// for cross-TU engine internals.
#define LUCID_INTERNAL LUCID_API_LOCAL

// ---------------------------------------------------------------------------
// Convenience macros for deleted copy / move operations
// ---------------------------------------------------------------------------

// Deletes the copy constructor and copy-assignment operator for Type.
// Use in the public section of non-copyable classes.

// Declares ``Type`` non-copyable by deleting its copy constructor and
// copy-assignment operator.
//
// Expands to two ``= delete`` declarations: ``Type(const Type&)`` and
// ``Type& operator=(const Type&)``.  Intended to be placed in the
// ``public:`` section of a class definition so the deletions are
// reported on the diagnostic side that the user is most likely to see.
//
// Parameters
// ----------
// Type : identifier
//     The name of the enclosing class.  Token-pasted directly into
//     the deleted-function declarations — must match the class name
//     exactly (no namespace prefix, no template-argument list).
//
// Examples
// --------
// Mark a resource-owning class as non-copyable::
//
//     class Tape {
//     public:
//         LUCID_NOCOPY(Tape);
//         // ... rest of API
//     };
//
// See Also
// --------
// :macro:`LUCID_NOMOVE` — companion macro that also forbids moves.
#define LUCID_NOCOPY(Type)                                                                         \
    Type(const Type&) = delete;                                                                    \
    Type& operator=(const Type&) = delete

// Deletes the move constructor and move-assignment operator for Type.
// Use alongside LUCID_NOCOPY for fully immovable types (e.g., mutex holders).

// Declares ``Type`` non-movable by deleting its move constructor and
// move-assignment operator.
//
// Expands to ``Type(Type&&) = delete`` and ``Type& operator=(Type&&)
// = delete``.  Typically used in combination with :macro:`LUCID_NOCOPY`
// for objects whose identity must be pinned to a specific memory
// address — e.g. types embedding a ``std::mutex``, types registered in
// a thread-local registry, or types pointed to by raw back-pointers.
//
// Parameters
// ----------
// Type : identifier
//     The name of the enclosing class; token-pasted into the deleted
//     declarations.
//
// Examples
// --------
// Make an object both non-copyable and non-movable::
//
//     class GraphLock {
//     public:
//         LUCID_NOCOPY(GraphLock);
//         LUCID_NOMOVE(GraphLock);
//     private:
//         std::mutex mu_;
//     };
//
// See Also
// --------
// :macro:`LUCID_NOCOPY` — companion macro that forbids copies.
#define LUCID_NOMOVE(Type)                                                                         \
    Type(Type&&) = delete;                                                                         \
    Type& operator=(Type&&) = delete
