#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)

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

#define LUCID_NOCOPY(Type)                                                                         \
    Type(const Type&) = delete;                                                                    \
    Type& operator=(const Type&) = delete

#define LUCID_NOMOVE(Type)                                                                         \
    Type(Type&&) = delete;                                                                         \
    Type& operator=(Type&&) = delete
