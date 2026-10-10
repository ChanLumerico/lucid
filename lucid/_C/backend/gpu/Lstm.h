// lucid/_C/backend/gpu/Lstm.h
//
// Single-layer, single-direction LSTM on the MLX (Metal) stream, float32.
// :class:`GpuBackend` checks the configuration and runs the 16-bit formats
// through these in float32 (``cast`` below): the recurrence sums T steps of
// gradient into the weights, which half precision cannot hold.  The saved
// gate and cell trajectories therefore stay float32, exactly as on CPU.
//
// Each timestep:
//   raw = X_t @ W_ih.T + h_prev @ W_hh.T + (b_ih + b_hh)        (B, 4H)
//   i, f = sigmoid(raw[:, :H]), sigmoid(raw[:, H:2H])
//   g    = tanh(raw[:, 2H:3H])
//   o    = sigmoid(raw[:, 3H:4H])
//   c    = f * c_prev + i * g
//   h_raw= o * tanh(c)
//   h    = (proj > 0) ? h_raw @ W_hr.T : h_raw
// Saved for BPTT: per-step gate post-activations (i, f, g, o concatenated)
// in ``gates_all`` of shape (T, B, 4H), and the cell trajectory in
// ``cells_all`` of shape (T+1, B, H) where cells_all[0] = c0 and
// cells_all[t+1] = c_t.

#pragma once

#include <optional>
#include <vector>

#include <mlx/array.h>
#include <mlx/ops.h>

#include "../../core/Dtype.h"
#include "../../core/Storage.h"
#include "../IBackend.h"
#include "HalfAccumulation.h"
#include "MlxBridge.h"

namespace lucid::gpu::lstm {

namespace mx = ::mlx::core;

inline Storage cast(const Storage& s, Dtype to) {
    return Storage{gpu::wrap_mlx_array(
        backend::narrow_to(*std::get<GpuStorage>(s).arr, gpu::to_mlx_dtype(to)), to)};
}

inline std::vector<Storage> cast(const std::vector<Storage>& v, Dtype to) {
    std::vector<Storage> out;
    out.reserve(v.size());
    for (const auto& s : v)
        out.push_back(cast(s, to));
    return out;
}

inline const mx::array& arr(const Storage& s) {
    return *std::get<GpuStorage>(s).arr;
}

inline Storage wrap(const mx::array& a, Dtype dt) {
    return Storage{gpu::wrap_mlx_array(mx::contiguous(a), dt)};
}

// Block t of a (T, B, W) array, as (B, W).
inline mx::array step(const mx::array& a, int t, int B, int W) {
    return mx::reshape(mx::slice(a, mx::Shape{t, 0, 0}, mx::Shape{t + 1, B, W}), mx::Shape{B, W});
}

// Columns [lo, hi) of a (B, ·) array.
inline mx::array cols(const mx::array& a, int B, int lo, int hi) {
    return mx::slice(a, mx::Shape{0, lo}, mx::Shape{B, hi});
}

inline mx::array transposed(const mx::array& a) {
    return mx::transpose(a, std::vector<int>{1, 0});
}

// Returns {output, hn, cn, gates_all, cells_all}.
inline std::vector<Storage> forward_train(const Storage& input,
                                          const Storage& h0,
                                          const Storage& c0,
                                          const std::vector<Storage>& weights,
                                          const backend::IBackend::LstmOpts& opts,
                                          Dtype dt) {
    const int T = opts.seq_len, B = opts.batch_size, I = opts.input_size;
    const int H = opts.hidden_size, P = opts.proj_size, Hrec = P > 0 ? P : H;

    // h0 / c0 arrive (1, B, ·); the recurrence runs on (B, ·).
    auto h = mx::reshape(arr(h0), mx::Shape{B, Hrec});
    auto c = mx::reshape(arr(c0), mx::Shape{B, H});
    const auto Wih_T = transposed(arr(weights[0]));  // (I, 4H)
    const auto Whh_T = transposed(arr(weights[1]));  // (Hrec, 4H)
    const auto bias = mx::add(arr(weights[2]), arr(weights[3]));
    std::optional<mx::array> Whr_T;
    if (P > 0)
        Whr_T = transposed(arr(weights[4]));  // (H, P)

    std::vector<mx::array> outputs, gates_steps, cells_steps{c};
    for (int t = 0; t < T; ++t) {
        const auto raw = mx::add(
            mx::add(mx::matmul(step(arr(input), t, B, I), Wih_T), mx::matmul(h, Whh_T)), bias);
        const auto i_g = mx::sigmoid(cols(raw, B, 0, H));
        const auto f_g = mx::sigmoid(cols(raw, B, H, 2 * H));
        const auto g_g = mx::tanh(cols(raw, B, 2 * H, 3 * H));
        const auto o_g = mx::sigmoid(cols(raw, B, 3 * H, 4 * H));
        gates_steps.push_back(mx::concatenate(std::vector<mx::array>{i_g, f_g, g_g, o_g}, 1));

        c = mx::add(mx::multiply(f_g, c), mx::multiply(i_g, g_g));
        cells_steps.push_back(c);
        const auto h_raw = mx::multiply(o_g, mx::tanh(c));
        h = Whr_T ? mx::matmul(h_raw, *Whr_T) : h_raw;
        outputs.push_back(h);
    }

    return {wrap(mx::stack(outputs, 0), dt), wrap(h, dt), wrap(c, dt),
            wrap(mx::stack(gates_steps, 0), dt), wrap(mx::stack(cells_steps, 0), dt)};
}

// The recurrent input of step t: h0 at t == 0, otherwise the (possibly
// projected) o_{t-1} * tanh(c_{t-1}) rebuilt from the saved trajectory.
inline mx::array recurrent_input(const mx::array& gates,
                                 const mx::array& cells,
                                 const mx::array& h0,
                                 const std::optional<mx::array>& Whr,
                                 int t,
                                 int B,
                                 int H) {
    if (t == 0)
        return h0;
    const auto o_prev = cols(step(gates, t - 1, B, 4 * H), B, 3 * H, 4 * H);
    const auto h_raw = mx::multiply(o_prev, mx::tanh(step(cells, t, B, H)));
    return Whr ? mx::matmul(h_raw, transposed(*Whr)) : h_raw;
}

struct GateGrads {
    mx::array d_gates;  // d(pre-activation) for [i, f, g, o], (B, 4H)
    mx::array dc_prev;  // dL/dc_{t-1}, (B, H)
};

// Chain one step's hidden gradient ``dh`` (size-H basis) and the cell
// gradient carried from step t + 1 through the gates.
inline GateGrads gate_grads(const mx::array& gates_t,
                            const mx::array& c_t,
                            const mx::array& c_prev,
                            const mx::array& dh,
                            const mx::array& dc,
                            int B,
                            int H) {
    const auto i = cols(gates_t, B, 0, H), f = cols(gates_t, B, H, 2 * H);
    const auto g = cols(gates_t, B, 2 * H, 3 * H), o = cols(gates_t, B, 3 * H, 4 * H);
    const auto one = mx::array(1.0f, gates_t.dtype());
    const auto tc = mx::tanh(c_t);
    const auto dc_k =
        mx::add(mx::multiply(mx::multiply(dh, o), mx::subtract(one, mx::multiply(tc, tc))), dc);
    const auto d_i = mx::multiply(mx::multiply(dc_k, g), mx::multiply(i, mx::subtract(one, i)));
    const auto d_f =
        mx::multiply(mx::multiply(dc_k, c_prev), mx::multiply(f, mx::subtract(one, f)));
    const auto d_g = mx::multiply(mx::multiply(dc_k, i), mx::subtract(one, mx::multiply(g, g)));
    const auto d_o = mx::multiply(mx::multiply(dh, tc), mx::multiply(o, mx::subtract(one, o)));
    return {mx::concatenate(std::vector<mx::array>{d_i, d_f, d_g, d_o}, 1), mx::multiply(dc_k, f)};
}

// BPTT.  Returns {dX, dh0, dc0, dW_ih, dW_hh, db_ih, db_hh[, dW_hr]}.
inline std::vector<Storage> backward(const Storage& grad_output,
                                     const Storage& grad_hn,
                                     const Storage& grad_cn,
                                     const Storage& input,
                                     const Storage& h0,
                                     const std::vector<Storage>& weights,
                                     const Storage& gates_all,
                                     const Storage& cells_all,
                                     const backend::IBackend::LstmOpts& opts,
                                     Dtype dt) {
    const int T = opts.seq_len, B = opts.batch_size, I = opts.input_size;
    const int H = opts.hidden_size, P = opts.proj_size, Hrec = P > 0 ? P : H, fH = 4 * H;
    const auto mdt = gpu::to_mlx_dtype(dt);
    const auto& Gates = arr(gates_all);  // (T, B, 4H)
    const auto& Cells = arr(cells_all);  // (T+1, B, H)
    const auto& Wih = arr(weights[0]);
    const auto& Whh = arr(weights[1]);
    const auto H0 = mx::reshape(arr(h0), mx::Shape{B, Hrec});
    std::optional<mx::array> Whr, dWhr;
    if (P > 0) {
        Whr = arr(weights[4]);  // (P, H)
        dWhr = mx::zeros(mx::Shape{P, H}, mdt);
    }

    auto dWih = mx::zeros(mx::Shape{fH, I}, mdt);
    auto dWhh = mx::zeros(mx::Shape{fH, Hrec}, mdt);
    auto dB = mx::zeros(mx::Shape{fH}, mdt);
    auto dh_next = mx::reshape(arr(grad_hn), mx::Shape{B, Hrec});
    auto dc_next = mx::reshape(arr(grad_cn), mx::Shape{B, H});
    std::vector<mx::array> dX_steps(static_cast<std::size_t>(T), mx::zeros({1}, mdt));

    for (int t = T - 1; t >= 0; --t) {
        const auto gates_t = step(Gates, t, B, fH);
        const auto c_t = step(Cells, t + 1, B, H);
        auto dh = mx::add(step(arr(grad_output), t, B, Hrec), dh_next);
        if (Whr) {
            const auto h_raw = mx::multiply(cols(gates_t, B, 3 * H, fH), mx::tanh(c_t));
            dWhr = mx::add(*dWhr, mx::matmul(transposed(dh), h_raw));
            dh = mx::matmul(dh, *Whr);
        }
        const auto gg = gate_grads(gates_t, c_t, step(Cells, t, B, H), dh, dc_next, B, H);
        const auto dG_T = transposed(gg.d_gates);
        const auto h_prev = recurrent_input(Gates, Cells, H0, Whr, t, B, H);

        dX_steps[static_cast<std::size_t>(t)] = mx::matmul(gg.d_gates, Wih);
        dh_next = mx::matmul(gg.d_gates, Whh);
        dc_next = gg.dc_prev;
        dWih = mx::add(dWih, mx::matmul(dG_T, step(arr(input), t, B, I)));
        dWhh = mx::add(dWhh, mx::matmul(dG_T, h_prev));
        dB = mx::add(dB, mx::sum(gg.d_gates, std::vector<int>{0}));
    }

    // b_ih and b_hh enter the pre-activation as one sum, so they share dB.
    // dh0 / dc0 leave in the (1, B, ·) of the h0 / c0 they lead to: a rank-2
    // array reaching that tensor's slice or cat backward fails on its rank.
    std::vector<Storage> result{wrap(mx::stack(dX_steps, 0), dt),
                                wrap(mx::reshape(dh_next, mx::Shape{1, B, Hrec}), dt),
                                wrap(mx::reshape(dc_next, mx::Shape{1, B, H}), dt),
                                wrap(dWih, dt),
                                wrap(dWhh, dt),
                                wrap(dB, dt),
                                wrap(dB, dt)};
    if (P > 0)
        result.push_back(wrap(*dWhr, dt));
    return result;
}

}  // namespace lucid::gpu::lstm
