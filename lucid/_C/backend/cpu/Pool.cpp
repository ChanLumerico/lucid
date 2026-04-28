#include "Pool.h"

#include <algorithm>
#include <limits>

namespace lucid::backend::cpu {

namespace {

template <typename T>
void max_pool2d_forward_typed(const T* x,
                              T* y,
                              std::int32_t* argmax,
                              int B,
                              int C,
                              int H,
                              int W,
                              int KH,
                              int KW,
                              int OH,
                              int OW,
                              int sh,
                              int sw,
                              int ph,
                              int pw) {
    constexpr T NEG_INF = -std::numeric_limits<T>::infinity();
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* xb = x + (b * C + c) * H * W;
            T* yb = y + (b * C + c) * OH * OW;
            auto* ab = argmax + (b * C + c) * OH * OW;
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    T best = NEG_INF;
                    int best_idx = -1;
                    for (int kh = 0; kh < KH; ++kh) {
                        const int ih = oh * sh - ph + kh;
                        if (ih < 0 || ih >= H)
                            continue;
                        for (int kw = 0; kw < KW; ++kw) {
                            const int iw = ow * sw - pw + kw;
                            if (iw < 0 || iw >= W)
                                continue;
                            const int idx = ih * W + iw;
                            const T v = xb[idx];
                            if (v > best) {
                                best = v;
                                best_idx = idx;
                            }
                        }
                    }
                    yb[oh * OW + ow] = best;
                    ab[oh * OW + ow] = best_idx;
                }
            }
        }
    }
}

template <typename T>
void max_pool2d_backward_typed(
    const T* g, const std::int32_t* argmax, T* dx, int B, int C, int H, int W, int OH, int OW) {
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* gb = g + (b * C + c) * OH * OW;
            const auto* ab = argmax + (b * C + c) * OH * OW;
            T* dxb = dx + (b * C + c) * H * W;
            for (int o = 0; o < OH * OW; ++o) {
                const int idx = ab[o];
                if (idx >= 0)
                    dxb[idx] += gb[o];
            }
        }
    }
}

template <typename T>
void avg_pool2d_forward_typed(const T* x,
                              T* y,
                              int B,
                              int C,
                              int H,
                              int W,
                              int KH,
                              int KW,
                              int OH,
                              int OW,
                              int sh,
                              int sw,
                              int ph,
                              int pw) {
    const T inv = T{1} / static_cast<T>(KH * KW);
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* xb = x + (b * C + c) * H * W;
            T* yb = y + (b * C + c) * OH * OW;
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    T sum = T{};
                    for (int kh = 0; kh < KH; ++kh) {
                        const int ih = oh * sh - ph + kh;
                        if (ih < 0 || ih >= H)
                            continue;
                        for (int kw = 0; kw < KW; ++kw) {
                            const int iw = ow * sw - pw + kw;
                            if (iw < 0 || iw >= W)
                                continue;
                            sum += xb[ih * W + iw];
                        }
                    }
                    yb[oh * OW + ow] = sum * inv;
                }
            }
        }
    }
}

template <typename T>
void avg_pool2d_backward_typed(const T* g,
                               T* dx,
                               int B,
                               int C,
                               int H,
                               int W,
                               int KH,
                               int KW,
                               int OH,
                               int OW,
                               int sh,
                               int sw,
                               int ph,
                               int pw) {
    const T inv = T{1} / static_cast<T>(KH * KW);
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* gb = g + (b * C + c) * OH * OW;
            T* dxb = dx + (b * C + c) * H * W;
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    const T scaled = gb[oh * OW + ow] * inv;
                    for (int kh = 0; kh < KH; ++kh) {
                        const int ih = oh * sh - ph + kh;
                        if (ih < 0 || ih >= H)
                            continue;
                        for (int kw = 0; kw < KW; ++kw) {
                            const int iw = ow * sw - pw + kw;
                            if (iw < 0 || iw >= W)
                                continue;
                            dxb[ih * W + iw] += scaled;
                        }
                    }
                }
            }
        }
    }
}

// ============================ 1D ============================

template <typename T>
void max_pool1d_forward_typed(
    const T* x, T* y, std::int32_t* argmax, int B, int C, int L, int KL, int OL, int sl, int pl) {
    constexpr T NEG_INF = -std::numeric_limits<T>::infinity();
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* xb = x + (b * C + c) * L;
            T* yb = y + (b * C + c) * OL;
            auto* ab = argmax + (b * C + c) * OL;
            for (int ol = 0; ol < OL; ++ol) {
                T best = NEG_INF;
                int best_idx = -1;
                for (int kl = 0; kl < KL; ++kl) {
                    const int il = ol * sl - pl + kl;
                    if (il < 0 || il >= L)
                        continue;
                    const T v = xb[il];
                    if (v > best) {
                        best = v;
                        best_idx = il;
                    }
                }
                yb[ol] = best;
                ab[ol] = best_idx;
            }
        }
    }
}

template <typename T>
void max_pool1d_backward_typed(
    const T* g, const std::int32_t* argmax, T* dx, int B, int C, int L, int OL) {
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* gb = g + (b * C + c) * OL;
            const auto* ab = argmax + (b * C + c) * OL;
            T* dxb = dx + (b * C + c) * L;
            for (int o = 0; o < OL; ++o) {
                const int idx = ab[o];
                if (idx >= 0)
                    dxb[idx] += gb[o];
            }
        }
    }
}

template <typename T>
void avg_pool1d_forward_typed(
    const T* x, T* y, int B, int C, int L, int KL, int OL, int sl, int pl) {
    const T inv = T{1} / static_cast<T>(KL);
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* xb = x + (b * C + c) * L;
            T* yb = y + (b * C + c) * OL;
            for (int ol = 0; ol < OL; ++ol) {
                T sum = T{};
                for (int kl = 0; kl < KL; ++kl) {
                    const int il = ol * sl - pl + kl;
                    if (il < 0 || il >= L)
                        continue;
                    sum += xb[il];
                }
                yb[ol] = sum * inv;
            }
        }
    }
}

template <typename T>
void avg_pool1d_backward_typed(
    const T* g, T* dx, int B, int C, int L, int KL, int OL, int sl, int pl) {
    const T inv = T{1} / static_cast<T>(KL);
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* gb = g + (b * C + c) * OL;
            T* dxb = dx + (b * C + c) * L;
            for (int ol = 0; ol < OL; ++ol) {
                const T scaled = gb[ol] * inv;
                for (int kl = 0; kl < KL; ++kl) {
                    const int il = ol * sl - pl + kl;
                    if (il < 0 || il >= L)
                        continue;
                    dxb[il] += scaled;
                }
            }
        }
    }
}

// ============================ 3D ============================

template <typename T>
void max_pool3d_forward_typed(const T* x,
                              T* y,
                              std::int32_t* argmax,
                              int B,
                              int C,
                              int D,
                              int H,
                              int W,
                              int KD,
                              int KH,
                              int KW,
                              int OD,
                              int OH,
                              int OW,
                              int sd,
                              int sh,
                              int sw,
                              int pd,
                              int ph,
                              int pw) {
    constexpr T NEG_INF = -std::numeric_limits<T>::infinity();
    const int HW = H * W;
    const int OHW = OH * OW;
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* xb = x + (b * C + c) * D * HW;
            T* yb = y + (b * C + c) * OD * OHW;
            auto* ab = argmax + (b * C + c) * OD * OHW;
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        T best = NEG_INF;
                        int best_idx = -1;
                        for (int kd = 0; kd < KD; ++kd) {
                            const int id = od * sd - pd + kd;
                            if (id < 0 || id >= D)
                                continue;
                            for (int kh = 0; kh < KH; ++kh) {
                                const int ih = oh * sh - ph + kh;
                                if (ih < 0 || ih >= H)
                                    continue;
                                for (int kw = 0; kw < KW; ++kw) {
                                    const int iw = ow * sw - pw + kw;
                                    if (iw < 0 || iw >= W)
                                        continue;
                                    const int idx = (id * H + ih) * W + iw;
                                    const T v = xb[idx];
                                    if (v > best) {
                                        best = v;
                                        best_idx = idx;
                                    }
                                }
                            }
                        }
                        const int oidx = (od * OH + oh) * OW + ow;
                        yb[oidx] = best;
                        ab[oidx] = best_idx;
                    }
                }
            }
        }
    }
}

template <typename T>
void max_pool3d_backward_typed(const T* g,
                               const std::int32_t* argmax,
                               T* dx,
                               int B,
                               int C,
                               int D,
                               int H,
                               int W,
                               int OD,
                               int OH,
                               int OW) {
    const int OHW = OD * OH * OW;
    const int IHW = D * H * W;
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* gb = g + (b * C + c) * OHW;
            const auto* ab = argmax + (b * C + c) * OHW;
            T* dxb = dx + (b * C + c) * IHW;
            for (int o = 0; o < OHW; ++o) {
                const int idx = ab[o];
                if (idx >= 0)
                    dxb[idx] += gb[o];
            }
        }
    }
}

template <typename T>
void avg_pool3d_forward_typed(const T* x,
                              T* y,
                              int B,
                              int C,
                              int D,
                              int H,
                              int W,
                              int KD,
                              int KH,
                              int KW,
                              int OD,
                              int OH,
                              int OW,
                              int sd,
                              int sh,
                              int sw,
                              int pd,
                              int ph,
                              int pw) {
    const T inv = T{1} / static_cast<T>(KD * KH * KW);
    const int HW = H * W;
    const int OHW = OH * OW;
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* xb = x + (b * C + c) * D * HW;
            T* yb = y + (b * C + c) * OD * OHW;
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        T sum = T{};
                        for (int kd = 0; kd < KD; ++kd) {
                            const int id = od * sd - pd + kd;
                            if (id < 0 || id >= D)
                                continue;
                            for (int kh = 0; kh < KH; ++kh) {
                                const int ih = oh * sh - ph + kh;
                                if (ih < 0 || ih >= H)
                                    continue;
                                for (int kw = 0; kw < KW; ++kw) {
                                    const int iw = ow * sw - pw + kw;
                                    if (iw < 0 || iw >= W)
                                        continue;
                                    sum += xb[(id * H + ih) * W + iw];
                                }
                            }
                        }
                        yb[(od * OH + oh) * OW + ow] = sum * inv;
                    }
                }
            }
        }
    }
}

template <typename T>
void avg_pool3d_backward_typed(const T* g,
                               T* dx,
                               int B,
                               int C,
                               int D,
                               int H,
                               int W,
                               int KD,
                               int KH,
                               int KW,
                               int OD,
                               int OH,
                               int OW,
                               int sd,
                               int sh,
                               int sw,
                               int pd,
                               int ph,
                               int pw) {
    const T inv = T{1} / static_cast<T>(KD * KH * KW);
    const int HW = H * W;
    const int OHW = OH * OW;
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* gb = g + (b * C + c) * OD * OHW;
            T* dxb = dx + (b * C + c) * D * HW;
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        const T scaled = gb[(od * OH + oh) * OW + ow] * inv;
                        for (int kd = 0; kd < KD; ++kd) {
                            const int id = od * sd - pd + kd;
                            if (id < 0 || id >= D)
                                continue;
                            for (int kh = 0; kh < KH; ++kh) {
                                const int ih = oh * sh - ph + kh;
                                if (ih < 0 || ih >= H)
                                    continue;
                                for (int kw = 0; kw < KW; ++kw) {
                                    const int iw = ow * sw - pw + kw;
                                    if (iw < 0 || iw >= W)
                                        continue;
                                    dxb[(id * H + ih) * W + iw] += scaled;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // namespace

// ------------------------ 1D extern ------------------------

void max_pool1d_forward_f32(const float* x,
                            float* y,
                            std::int32_t* a,
                            int B,
                            int C,
                            int L,
                            int KL,
                            int OL,
                            int sl,
                            int pl) {
    max_pool1d_forward_typed<float>(x, y, a, B, C, L, KL, OL, sl, pl);
}
void max_pool1d_forward_f64(const double* x,
                            double* y,
                            std::int32_t* a,
                            int B,
                            int C,
                            int L,
                            int KL,
                            int OL,
                            int sl,
                            int pl) {
    max_pool1d_forward_typed<double>(x, y, a, B, C, L, KL, OL, sl, pl);
}
void max_pool1d_backward_f32(
    const float* g, const std::int32_t* a, float* dx, int B, int C, int L, int OL) {
    max_pool1d_backward_typed<float>(g, a, dx, B, C, L, OL);
}
void max_pool1d_backward_f64(
    const double* g, const std::int32_t* a, double* dx, int B, int C, int L, int OL) {
    max_pool1d_backward_typed<double>(g, a, dx, B, C, L, OL);
}
void avg_pool1d_forward_f32(
    const float* x, float* y, int B, int C, int L, int KL, int OL, int sl, int pl) {
    avg_pool1d_forward_typed<float>(x, y, B, C, L, KL, OL, sl, pl);
}
void avg_pool1d_forward_f64(
    const double* x, double* y, int B, int C, int L, int KL, int OL, int sl, int pl) {
    avg_pool1d_forward_typed<double>(x, y, B, C, L, KL, OL, sl, pl);
}
void avg_pool1d_backward_f32(
    const float* g, float* dx, int B, int C, int L, int KL, int OL, int sl, int pl) {
    avg_pool1d_backward_typed<float>(g, dx, B, C, L, KL, OL, sl, pl);
}
void avg_pool1d_backward_f64(
    const double* g, double* dx, int B, int C, int L, int KL, int OL, int sl, int pl) {
    avg_pool1d_backward_typed<double>(g, dx, B, C, L, KL, OL, sl, pl);
}

// ------------------------ 3D extern ------------------------

void max_pool3d_forward_f32(const float* x,
                            float* y,
                            std::int32_t* a,
                            int B,
                            int C,
                            int D,
                            int H,
                            int W,
                            int KD,
                            int KH,
                            int KW,
                            int OD,
                            int OH,
                            int OW,
                            int sd,
                            int sh,
                            int sw,
                            int pd,
                            int ph,
                            int pw) {
    max_pool3d_forward_typed<float>(x, y, a, B, C, D, H, W, KD, KH, KW, OD, OH, OW, sd, sh, sw, pd,
                                    ph, pw);
}
void max_pool3d_forward_f64(const double* x,
                            double* y,
                            std::int32_t* a,
                            int B,
                            int C,
                            int D,
                            int H,
                            int W,
                            int KD,
                            int KH,
                            int KW,
                            int OD,
                            int OH,
                            int OW,
                            int sd,
                            int sh,
                            int sw,
                            int pd,
                            int ph,
                            int pw) {
    max_pool3d_forward_typed<double>(x, y, a, B, C, D, H, W, KD, KH, KW, OD, OH, OW, sd, sh, sw, pd,
                                     ph, pw);
}
void max_pool3d_backward_f32(const float* g,
                             const std::int32_t* a,
                             float* dx,
                             int B,
                             int C,
                             int D,
                             int H,
                             int W,
                             int OD,
                             int OH,
                             int OW) {
    max_pool3d_backward_typed<float>(g, a, dx, B, C, D, H, W, OD, OH, OW);
}
void max_pool3d_backward_f64(const double* g,
                             const std::int32_t* a,
                             double* dx,
                             int B,
                             int C,
                             int D,
                             int H,
                             int W,
                             int OD,
                             int OH,
                             int OW) {
    max_pool3d_backward_typed<double>(g, a, dx, B, C, D, H, W, OD, OH, OW);
}
void avg_pool3d_forward_f32(const float* x,
                            float* y,
                            int B,
                            int C,
                            int D,
                            int H,
                            int W,
                            int KD,
                            int KH,
                            int KW,
                            int OD,
                            int OH,
                            int OW,
                            int sd,
                            int sh,
                            int sw,
                            int pd,
                            int ph,
                            int pw) {
    avg_pool3d_forward_typed<float>(x, y, B, C, D, H, W, KD, KH, KW, OD, OH, OW, sd, sh, sw, pd, ph,
                                    pw);
}
void avg_pool3d_forward_f64(const double* x,
                            double* y,
                            int B,
                            int C,
                            int D,
                            int H,
                            int W,
                            int KD,
                            int KH,
                            int KW,
                            int OD,
                            int OH,
                            int OW,
                            int sd,
                            int sh,
                            int sw,
                            int pd,
                            int ph,
                            int pw) {
    avg_pool3d_forward_typed<double>(x, y, B, C, D, H, W, KD, KH, KW, OD, OH, OW, sd, sh, sw, pd,
                                     ph, pw);
}
void avg_pool3d_backward_f32(const float* g,
                             float* dx,
                             int B,
                             int C,
                             int D,
                             int H,
                             int W,
                             int KD,
                             int KH,
                             int KW,
                             int OD,
                             int OH,
                             int OW,
                             int sd,
                             int sh,
                             int sw,
                             int pd,
                             int ph,
                             int pw) {
    avg_pool3d_backward_typed<float>(g, dx, B, C, D, H, W, KD, KH, KW, OD, OH, OW, sd, sh, sw, pd,
                                     ph, pw);
}
void avg_pool3d_backward_f64(const double* g,
                             double* dx,
                             int B,
                             int C,
                             int D,
                             int H,
                             int W,
                             int KD,
                             int KH,
                             int KW,
                             int OD,
                             int OH,
                             int OW,
                             int sd,
                             int sh,
                             int sw,
                             int pd,
                             int ph,
                             int pw) {
    avg_pool3d_backward_typed<double>(g, dx, B, C, D, H, W, KD, KH, KW, OD, OH, OW, sd, sh, sw, pd,
                                      ph, pw);
}

void max_pool2d_forward_f32(const float* x,
                            float* y,
                            std::int32_t* a,
                            int B,
                            int C,
                            int H,
                            int W,
                            int KH,
                            int KW,
                            int OH,
                            int OW,
                            int sh,
                            int sw,
                            int ph,
                            int pw) {
    max_pool2d_forward_typed<float>(x, y, a, B, C, H, W, KH, KW, OH, OW, sh, sw, ph, pw);
}
void max_pool2d_forward_f64(const double* x,
                            double* y,
                            std::int32_t* a,
                            int B,
                            int C,
                            int H,
                            int W,
                            int KH,
                            int KW,
                            int OH,
                            int OW,
                            int sh,
                            int sw,
                            int ph,
                            int pw) {
    max_pool2d_forward_typed<double>(x, y, a, B, C, H, W, KH, KW, OH, OW, sh, sw, ph, pw);
}
void max_pool2d_backward_f32(
    const float* g, const std::int32_t* a, float* dx, int B, int C, int H, int W, int OH, int OW) {
    max_pool2d_backward_typed<float>(g, a, dx, B, C, H, W, OH, OW);
}
void max_pool2d_backward_f64(const double* g,
                             const std::int32_t* a,
                             double* dx,
                             int B,
                             int C,
                             int H,
                             int W,
                             int OH,
                             int OW) {
    max_pool2d_backward_typed<double>(g, a, dx, B, C, H, W, OH, OW);
}
void avg_pool2d_forward_f32(const float* x,
                            float* y,
                            int B,
                            int C,
                            int H,
                            int W,
                            int KH,
                            int KW,
                            int OH,
                            int OW,
                            int sh,
                            int sw,
                            int ph,
                            int pw) {
    avg_pool2d_forward_typed<float>(x, y, B, C, H, W, KH, KW, OH, OW, sh, sw, ph, pw);
}
void avg_pool2d_forward_f64(const double* x,
                            double* y,
                            int B,
                            int C,
                            int H,
                            int W,
                            int KH,
                            int KW,
                            int OH,
                            int OW,
                            int sh,
                            int sw,
                            int ph,
                            int pw) {
    avg_pool2d_forward_typed<double>(x, y, B, C, H, W, KH, KW, OH, OW, sh, sw, ph, pw);
}
void avg_pool2d_backward_f32(const float* g,
                             float* dx,
                             int B,
                             int C,
                             int H,
                             int W,
                             int KH,
                             int KW,
                             int OH,
                             int OW,
                             int sh,
                             int sw,
                             int ph,
                             int pw) {
    avg_pool2d_backward_typed<float>(g, dx, B, C, H, W, KH, KW, OH, OW, sh, sw, ph, pw);
}
void avg_pool2d_backward_f64(const double* g,
                             double* dx,
                             int B,
                             int C,
                             int H,
                             int W,
                             int KH,
                             int KW,
                             int OH,
                             int OW,
                             int sh,
                             int sw,
                             int ph,
                             int pw) {
    avg_pool2d_backward_typed<double>(g, dx, B, C, H, W, KH, KW, OH, OW, sh, sw, ph, pw);
}

}  // namespace lucid::backend::cpu
