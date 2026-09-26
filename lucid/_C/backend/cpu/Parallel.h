// lucid/_C/backend/cpu/Parallel.h
//
// Chunked parallel loops for CPU kernels that Accelerate does not thread.
//
// Accelerate's BLAS threads a large GEMM on its own, and vDSP / vForce run
// one array at a time on the calling thread.  What neither covers is a loop
// Lucid writes itself — a scalar ``std::erf`` per element, or one small
// GEMM per attention head — and those ran on a single core: the exact GELU
// at 5 ns an element and 128 heads of a small transformer one after
// another, 2.8 and 3.0 ms a call, most of a CPU training step.
//
// The loops split into contiguous chunks on libdispatch's global concurrent
// queue, the pool Accelerate itself uses, so no thread of Lucid's own is
// created.  Every chunk writes only its own range, and the work inside one
// is exactly what the serial loop did, so the result is bitwise the same at
// any core count — parallel only changes when each element is computed.

#pragma once

#include <algorithm>
#include <cstddef>
#include <thread>
#include <type_traits>

#include <dispatch/dispatch.h>

namespace lucid::backend::cpu {

// Number of chunks a parallel loop may split into on this machine.
//
// Returns
// -------
// std::size_t
//     The logical core count, at least 1.
inline std::size_t parallel_workers() {
    static const std::size_t workers =
        std::max<std::size_t>(1, static_cast<std::size_t>(std::thread::hardware_concurrency()));
    return workers;
}

// Call ``fn(begin, end)`` over ``[0, n)`` in contiguous chunks, in parallel.
//
// Parameters
// ----------
// n : std::size_t
//     Iteration count.
// grain : std::size_t
//     Smallest chunk worth handing to another core.  Below two of them the
//     loop runs inline on the caller: dispatching costs a few microseconds,
//     which a small loop never earns back.
// fn : F
//     ``void(std::size_t begin, std::size_t end)``.  Must not throw — an
//     exception cannot cross a libdispatch worker — and must write only
//     what its own range owns.
//
// Notes
// -----
// Chunks are ``n * i / chunks`` to ``n * (i + 1) / chunks``, so they differ
// in size by at most one iteration and together cover ``[0, n)`` exactly.
template <class F>
void parallel_for(std::size_t n, std::size_t grain, F&& fn) {
    if (n == 0)
        return;
    const std::size_t chunks = std::min(n / std::max<std::size_t>(grain, 1), parallel_workers());
    if (chunks <= 1) {
        fn(std::size_t{0}, n);
        return;
    }
    struct Context {
        std::remove_reference_t<F>* fn;
        std::size_t n;
        std::size_t chunks;
    };
    Context context{&fn, n, chunks};
    dispatch_apply_f(chunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &context,
                     [](void* raw, std::size_t i) {
                         const auto* c = static_cast<const Context*>(raw);
                         const std::size_t begin = c->n * i / c->chunks;
                         const std::size_t end = c->n * (i + 1) / c->chunks;
                         (*c->fn)(begin, end);
                     });
}

}  // namespace lucid::backend::cpu
