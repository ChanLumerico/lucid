// lucid/_C/backend/cpu/CtcLoss.h
//
// Connectionist Temporal Classification (CTC) forward and backward recursion
// for the CPU backend (Graves et al. 2006).
//
// Both directions run in the log domain over the extended target sequence
// ``[blank, y_0, blank, y_1, ..., blank]`` of length ``L = 2 * S + 1``.
// An empty input sequence (``T_b <= 0``) has no lattice frames to index: its
// forward loss is ``0`` when the target is also empty (``S == 0``) or under
// ``zero_infinity``, and ``+inf`` otherwise; its backward gradient over the
// ``T`` time steps is identically ``0``.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "../../core/Allocator.h"
#include "../../core/Dtype.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"

namespace lucid::backend::cpu::ctc {

inline constexpr double kNegInf = -1e30;

inline double logaddexp(double a, double b) noexcept {
    if (a <= kNegInf / 2)
        return b;
    if (b <= kNegInf / 2)
        return a;
    const double hi = std::max(a, b);
    return hi + std::log1p(std::exp(std::min(a, b) - hi));
}

inline int read_i32(const CpuStorage& s, int i) noexcept {
    return static_cast<int>(reinterpret_cast<const std::int32_t*>(s.ptr.get())[i]);
}

inline std::vector<int> extended_targets(const CpuStorage& targets, int offset, int S, int blank) {
    std::vector<int> ext(static_cast<std::size_t>(2 * S + 1), blank);
    const auto* ptr = reinterpret_cast<const std::int32_t*>(targets.ptr.get());
    for (int s = 0; s < S; ++s)
        ext[static_cast<std::size_t>(2 * s + 1)] = static_cast<int>(ptr[offset + s]);
    return ext;
}

template <class ReadFn>
struct SequenceView {
    const CpuStorage& lp;
    int N;
    int C;
    int b;
    ReadFn read;

    std::size_t index(int t, int c) const noexcept {
        return (static_cast<std::size_t>(t) * static_cast<std::size_t>(N) +
                static_cast<std::size_t>(b)) *
                   static_cast<std::size_t>(C) +
               static_cast<std::size_t>(c);
    }

    double at(int t, int c) const noexcept { return read(lp, index(t, c)); }
};

template <class ReadFn>
inline std::vector<double>
compute_alpha(const SequenceView<ReadFn>& seq, int T_b, const std::vector<int>& ext) {
    const int L = static_cast<int>(ext.size());
    std::vector<double> alpha(static_cast<std::size_t>(T_b) * static_cast<std::size_t>(L), kNegInf);
    auto al = [&](int t, int s) -> double& {
        return alpha[static_cast<std::size_t>(t) * static_cast<std::size_t>(L) +
                     static_cast<std::size_t>(s)];
    };
    al(0, 0) = seq.at(0, ext[0]);
    if (L > 1)
        al(0, 1) = seq.at(0, ext[1]);
    for (int t = 1; t < T_b; ++t) {
        for (int s = 0; s < L; ++s) {
            double a = al(t - 1, s);
            if (s > 0)
                a = logaddexp(a, al(t - 1, s - 1));
            if (s > 1 && ext[static_cast<std::size_t>(s)] != ext[static_cast<std::size_t>(s - 2)])
                a = logaddexp(a, al(t - 1, s - 2));
            al(t, s) = a <= kNegInf / 2 ? kNegInf : a + seq.at(t, ext[static_cast<std::size_t>(s)]);
        }
    }
    return alpha;
}

template <class ReadFn>
inline std::vector<double>
compute_beta(const SequenceView<ReadFn>& seq, int T_b, const std::vector<int>& ext) {
    const int L = static_cast<int>(ext.size());
    std::vector<double> beta(static_cast<std::size_t>(T_b) * static_cast<std::size_t>(L), kNegInf);
    auto be = [&](int t, int s) -> double& {
        return beta[static_cast<std::size_t>(t) * static_cast<std::size_t>(L) +
                    static_cast<std::size_t>(s)];
    };
    be(T_b - 1, L - 1) = seq.at(T_b - 1, ext[static_cast<std::size_t>(L - 1)]);
    if (L > 1)
        be(T_b - 1, L - 2) = seq.at(T_b - 1, ext[static_cast<std::size_t>(L - 2)]);
    for (int t = T_b - 2; t >= 0; --t) {
        for (int s = 0; s < L; ++s) {
            double v = be(t + 1, s);
            if (s + 1 < L)
                v = logaddexp(v, be(t + 1, s + 1));
            if (s + 2 < L &&
                ext[static_cast<std::size_t>(s + 2)] != ext[static_cast<std::size_t>(s)])
                v = logaddexp(v, be(t + 1, s + 2));
            be(t, s) = v <= kNegInf / 2 ? kNegInf : v + seq.at(t, ext[static_cast<std::size_t>(s)]);
        }
    }
    return beta;
}

inline double total_log_prob(const std::vector<double>& alpha, int T_b, int L) noexcept {
    const std::size_t last = static_cast<std::size_t>(T_b - 1) * static_cast<std::size_t>(L);
    double ll = alpha[last + static_cast<std::size_t>(L - 1)];
    if (L >= 2)
        ll = logaddexp(ll, alpha[last + static_cast<std::size_t>(L - 2)]);
    return ll;
}

template <class ReadFn, class WriteFn>
inline void backward_sequence(const SequenceView<ReadFn>& seq,
                              int T_b,
                              const std::vector<int>& ext,
                              double gr,
                              bool zero_infinity,
                              WriteFn&& write) {
    const int L = static_cast<int>(ext.size());
    const auto alpha = compute_alpha(seq, T_b, ext);
    const double ll = total_log_prob(alpha, T_b, L);
    if (ll <= kNegInf / 2) {
        // When no alignment fits and zero_infinity is false, nll is +inf while
        // every class log-occupancy is -inf, so exp(-inf + inf - l) is NaN over
        // the sequence's active frames [0, T_b), matching the reference.
        if (!zero_infinity) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            for (int t = 0; t < T_b; ++t)
                for (int c = 0; c < seq.C; ++c)
                    write(seq.index(t, c), nan);
        }
        return;
    }
    const auto beta = compute_beta(seq, T_b, ext);
    const double nll = -ll;
    std::vector<double> lcab(static_cast<std::size_t>(seq.C), kNegInf);
    for (int t = 0; t < T_b; ++t) {
        std::fill(lcab.begin(), lcab.end(), kNegInf);
        for (int s = 0; s < L; ++s) {
            const std::size_t pos = static_cast<std::size_t>(t) * static_cast<std::size_t>(L) +
                                    static_cast<std::size_t>(s);
            const double ab = alpha[pos] + beta[pos];
            if (ab > kNegInf / 2) {
                const auto cls = static_cast<std::size_t>(ext[static_cast<std::size_t>(s)]);
                lcab[cls] = logaddexp(lcab[cls], ab);
            }
        }
        for (int c = 0; c < seq.C; ++c) {
            const double l = seq.at(t, c);
            const double lc = lcab[static_cast<std::size_t>(c)];
            const double occupancy = lc <= kNegInf / 2 ? 0.0 : std::exp(lc + nll - l);
            write(seq.index(t, c), (std::exp(l) - occupancy) * gr);
        }
    }
}

template <class ReadFn, class WriteFn>
inline Storage forward(const Storage& log_probs,
                       const Storage& targets,
                       const Storage& input_lengths,
                       const Storage& target_lengths,
                       const Shape& lp_shape,
                       int blank,
                       bool zero_infinity,
                       Dtype dt,
                       const CpuStorage& out_buf,
                       ReadFn read,
                       WriteFn write) {
    const int N = static_cast<int>(lp_shape[1]);
    const int C = static_cast<int>(lp_shape[2]);
    const auto& clp = std::get<CpuStorage>(log_probs);
    const auto& ctgt = std::get<CpuStorage>(targets);
    const auto& cil = std::get<CpuStorage>(input_lengths);
    const auto& ctl = std::get<CpuStorage>(target_lengths);

    int tgt_offset = 0;
    for (int b = 0; b < N; ++b) {
        const int T_b = read_i32(cil, b);
        const int S = std::max(0, read_i32(ctl, b));
        if (T_b <= 0) {
            tgt_offset += S;
            const double v =
                (S == 0 || zero_infinity) ? 0.0 : std::numeric_limits<double>::infinity();
            write(static_cast<std::size_t>(b), v);
            continue;
        }
        const auto ext = extended_targets(ctgt, tgt_offset, S, blank);
        tgt_offset += S;
        const SequenceView<ReadFn> seq{clp, N, C, b, read};
        const auto alpha = compute_alpha(seq, T_b, ext);
        const double end = total_log_prob(alpha, T_b, static_cast<int>(ext.size()));
        double v = end <= kNegInf / 2 ? std::numeric_limits<double>::infinity() : -end;
        if (zero_infinity && !std::isfinite(v))
            v = 0.0;
        write(static_cast<std::size_t>(b), v);
    }
    return Storage{CpuStorage{out_buf.ptr, out_buf.nbytes, dt}};
}

template <class ReadFn, class WriteFn>
inline Storage backward(const Storage& log_probs,
                        const Storage& targets,
                        const Storage& input_lengths,
                        const Storage& target_lengths,
                        const Storage& grad_out,
                        const Shape& lp_shape,
                        int blank,
                        bool zero_infinity,
                        Dtype dt,
                        const CpuStorage& out_buf,
                        ReadFn read,
                        WriteFn write) {
    const int N = static_cast<int>(lp_shape[1]);
    const int C = static_cast<int>(lp_shape[2]);
    const auto& clp = std::get<CpuStorage>(log_probs);
    const auto& ctgt = std::get<CpuStorage>(targets);
    const auto& cil = std::get<CpuStorage>(input_lengths);
    const auto& ctl = std::get<CpuStorage>(target_lengths);
    const auto& cgo = std::get<CpuStorage>(grad_out);

    int tgt_offset = 0;
    for (int b = 0; b < N; ++b) {
        const int T_b = read_i32(cil, b);
        const int S = std::max(0, read_i32(ctl, b));
        if (T_b <= 0) {
            tgt_offset += S;
            continue;
        }
        const auto ext = extended_targets(ctgt, tgt_offset, S, blank);
        tgt_offset += S;
        const SequenceView<ReadFn> seq{clp, N, C, b, read};
        const double gr = read(cgo, static_cast<std::size_t>(b));
        backward_sequence(seq, T_b, ext, gr, zero_infinity, write);
    }
    return Storage{CpuStorage{out_buf.ptr, out_buf.nbytes, dt}};
}

}  // namespace lucid::backend::cpu::ctc
