#pragma once

// =====================================================================
// Lucid C++ engine — public API boundary macros.
// =====================================================================
//
// The engine is built with `-fvisibility=hidden` so no symbol is exported by
// default. Mark public types/functions with `LUCID_API` to expose them; mark
// helpers explicitly intended for in-engine use with `LUCID_INTERNAL` for
// documentation (the build still hides them by default).
//
// Why this matters for production:
//   - Phase 7 ships a stable C ABI. Anything tagged LUCID_API is the surface
//     other languages will bind to; anything LUCID_INTERNAL can change freely.
//   - Smaller export tables → faster dynamic loads, fewer symbol clashes.
//   - clangd/IDE tooling shows API vs internal at a glance.
//
// Usage:
//
//   class LUCID_API TensorImpl : public std::enable_shared_from_this<...> {
//     ...
//   };
//
//   class LUCID_INTERNAL Allocator { ... };
//
// Until Phase 7 freezes the ABI, these macros are documentation-grade — the
// build flags do the actual work. After Phase 7 they become load-bearing for
// `liblucid_infer.dylib`.

#if defined(_WIN32) || defined(__CYGWIN__)
// Lucid is macOS-only by design (see CLAUDE.md / README); these branches exist
// only so the macro is well-defined on every toolchain that might tooltip-parse
// the headers (clangd in CI, etc.).
#define LUCID_API_EXPORT __declspec(dllexport)
#define LUCID_API_IMPORT __declspec(dllimport)
#define LUCID_API_LOCAL
#else
#define LUCID_API_EXPORT __attribute__((visibility("default")))
#define LUCID_API_IMPORT __attribute__((visibility("default")))
#define LUCID_API_LOCAL __attribute__((visibility("hidden")))
#endif

#if defined(LUCID_BUILDING_ENGINE)
#define LUCID_API LUCID_API_EXPORT
#else
#define LUCID_API LUCID_API_IMPORT
#endif

#define LUCID_INTERNAL LUCID_API_LOCAL

// Discourage accidental copies of large objects through the API boundary.
#define LUCID_NOCOPY(Type)      \
    Type(const Type&) = delete; \
    Type& operator=(const Type&) = delete

#define LUCID_NOMOVE(Type) \
    Type(Type&&) = delete; \
    Type& operator=(Type&&) = delete
