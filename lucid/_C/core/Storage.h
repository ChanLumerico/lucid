#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <variant>

#include "Dtype.h"

namespace mlx::core {
class array;
}

namespace lucid {

using VersionCounter = std::atomic<std::uint64_t>;

struct DataBuffer {
    std::shared_ptr<std::byte[]> ptr;
    std::size_t nbytes = 0;
    Dtype dtype = Dtype::F32;
    std::shared_ptr<VersionCounter> version = std::make_shared<VersionCounter>(0);

    void bump_version() const noexcept { version->fetch_add(1, std::memory_order_relaxed); }
    std::uint64_t get_version() const noexcept { return version->load(std::memory_order_relaxed); }
};

using StoragePtr = std::shared_ptr<DataBuffer>;

struct CpuStorage {
    std::shared_ptr<std::byte[]> ptr;
    std::size_t nbytes = 0;
    Dtype dtype = Dtype::F32;
    std::shared_ptr<VersionCounter> version = std::make_shared<VersionCounter>(0);

    CpuStorage() = default;

    CpuStorage(std::shared_ptr<std::byte[]> p, std::size_t nb, Dtype dt)
        : ptr(std::move(p)), nbytes(nb), dtype(dt) {}

    explicit CpuStorage(StoragePtr buf)
        : ptr(buf->ptr), nbytes(buf->nbytes), dtype(buf->dtype), version(buf->version) {}

    void bump_version() const noexcept { version->fetch_add(1, std::memory_order_relaxed); }
    std::uint64_t get_version() const noexcept { return version->load(std::memory_order_relaxed); }

    StoragePtr to_data_buffer() const {
        auto buf = std::make_shared<DataBuffer>();
        buf->ptr = ptr;
        buf->nbytes = nbytes;
        buf->dtype = dtype;
        buf->version = version;
        return buf;
    }
};

struct GpuStorage {
    std::shared_ptr<mlx::core::array> arr;
    std::size_t nbytes = 0;
    Dtype dtype = Dtype::F32;
    std::shared_ptr<VersionCounter> version = std::make_shared<VersionCounter>(0);

    void bump_version() const noexcept { version->fetch_add(1, std::memory_order_relaxed); }
    std::uint64_t get_version() const noexcept { return version->load(std::memory_order_relaxed); }
};

struct SharedStorage {
    void* cpu_ptr = nullptr;
    void* mtl_handle = nullptr;
    std::size_t nbytes = 0;
    Dtype dtype = Dtype::F32;
    std::shared_ptr<VersionCounter> version = std::make_shared<VersionCounter>(0);
    std::shared_ptr<void> owner;

    void bump_version() const noexcept { version->fetch_add(1, std::memory_order_relaxed); }
    std::uint64_t get_version() const noexcept { return version->load(std::memory_order_relaxed); }

    CpuStorage cpu_view() const {
        auto tok = owner;
        auto ptr = std::shared_ptr<std::byte[]>(
            static_cast<std::byte*>(cpu_ptr),
            [tok = std::move(tok)](std::byte*) mutable { tok.reset(); });

        CpuStorage cv(std::move(ptr), nbytes, dtype);
        cv.version = version;
        return cv;
    }
};

using Storage = std::variant<CpuStorage, GpuStorage, SharedStorage>;

inline bool storage_is_cpu(const Storage& s) noexcept {
    return s.index() == 0;
}
inline bool storage_is_gpu(const Storage& s) noexcept {
    return s.index() == 1;
}

inline bool storage_is_metal_shared(const Storage& s) noexcept {
    return s.index() == 2;
}
inline std::size_t storage_nbytes(const Storage& s) noexcept {
    switch (s.index()) {
    case 0:
        return std::get<0>(s).nbytes;
    case 1:
        return std::get<1>(s).nbytes;
    case 2:
        return std::get<2>(s).nbytes;
    default:
        return 0;
    }
}
inline Dtype storage_dtype(const Storage& s) noexcept {
    switch (s.index()) {
    case 0:
        return std::get<0>(s).dtype;
    case 1:
        return std::get<1>(s).dtype;
    case 2:
        return std::get<2>(s).dtype;
    default:
        return Dtype::F32;
    }
}

inline const CpuStorage& storage_cpu(const Storage& s) noexcept {
    return std::get<CpuStorage>(s);
}
inline CpuStorage& storage_cpu(Storage& s) noexcept {
    return std::get<CpuStorage>(s);
}

inline const GpuStorage& storage_gpu(const Storage& s) noexcept {
    return std::get<GpuStorage>(s);
}
inline GpuStorage& storage_gpu(Storage& s) noexcept {
    return std::get<GpuStorage>(s);
}

inline const SharedStorage& storage_metal_shared(const Storage& s) noexcept {
    return std::get<SharedStorage>(s);
}
inline SharedStorage& storage_metal_shared(Storage& s) noexcept {
    return std::get<SharedStorage>(s);
}

}  // namespace lucid
