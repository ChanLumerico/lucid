#pragma once

#include <atomic>
#include <cstddef>

#include "../api.h"
#include "Device.h"

namespace lucid {

struct LUCID_API MemoryStats {
    std::size_t current_bytes = 0;
    std::size_t peak_bytes = 0;
    std::size_t alloc_count = 0;
    std::size_t free_count = 0;
};

class LUCID_API MemoryTracker {
public:
    struct Counters {
        std::atomic<std::size_t> current_bytes{0};
        std::atomic<std::size_t> peak_bytes{0};
        std::atomic<std::size_t> alloc_count{0};
        std::atomic<std::size_t> free_count{0};
    };

    static void track_alloc(std::size_t nbytes, Device device);
    static void track_free(std::size_t nbytes, Device device);

    static MemoryStats get_stats(Device device);
    static void reset_peak(Device device);

private:
    static Counters& counters_for(Device device);
};

}  // namespace lucid
