#include "Im2Col.h"

namespace lucid::backend::cpu {

namespace {

// ============================ 1D ============================

template <typename T>
void im2col_1d_typed(
    const T* x, T* cols, int C, int L, int KL, int OL, int stride_l, int pad_l, int dilation_l) {
    for (int c = 0; c < C; ++c) {
        for (int kl = 0; kl < KL; ++kl) {
            const int row = c * KL + kl;
            T* col_row = cols + row * OL;
            for (int ol = 0; ol < OL; ++ol) {
                const int il = ol * stride_l - pad_l + kl * dilation_l;
                col_row[ol] = (il >= 0 && il < L) ? x[c * L + il] : T{};
            }
        }
    }
}

template <typename T>
void col2im_1d_typed(
    const T* cols, T* dx, int C, int L, int KL, int OL, int stride_l, int pad_l, int dilation_l) {
    for (int c = 0; c < C; ++c) {
        for (int kl = 0; kl < KL; ++kl) {
            const int row = c * KL + kl;
            const T* col_row = cols + row * OL;
            for (int ol = 0; ol < OL; ++ol) {
                const int il = ol * stride_l - pad_l + kl * dilation_l;
                if (il < 0 || il >= L)
                    continue;
                dx[c * L + il] += col_row[ol];
            }
        }
    }
}

// ============================ 2D ============================

template <typename T>
void im2col_typed(const T* x,
                  T* cols,
                  int C,
                  int H,
                  int W,
                  int KH,
                  int KW,
                  int OH,
                  int OW,
                  int stride_h,
                  int stride_w,
                  int pad_h,
                  int pad_w,
                  int dilation_h,
                  int dilation_w) {
    const int M = OH * OW;
    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                const int row = (c * KH + kh) * KW + kw;
                T* col_row = cols + row * M;
                for (int oh = 0; oh < OH; ++oh) {
                    const int ih = oh * stride_h - pad_h + kh * dilation_h;
                    if (ih < 0 || ih >= H) {
                        for (int ow = 0; ow < OW; ++ow)
                            col_row[oh * OW + ow] = T{};
                        continue;
                    }
                    for (int ow = 0; ow < OW; ++ow) {
                        const int iw = ow * stride_w - pad_w + kw * dilation_w;
                        col_row[oh * OW + ow] =
                            (iw >= 0 && iw < W) ? x[(c * H + ih) * W + iw] : T{};
                    }
                }
            }
        }
    }
}

template <typename T>
void col2im_typed(const T* cols,
                  T* dx,
                  int C,
                  int H,
                  int W,
                  int KH,
                  int KW,
                  int OH,
                  int OW,
                  int stride_h,
                  int stride_w,
                  int pad_h,
                  int pad_w,
                  int dilation_h,
                  int dilation_w) {
    const int M = OH * OW;
    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                const int row = (c * KH + kh) * KW + kw;
                const T* col_row = cols + row * M;
                for (int oh = 0; oh < OH; ++oh) {
                    const int ih = oh * stride_h - pad_h + kh * dilation_h;
                    if (ih < 0 || ih >= H)
                        continue;
                    for (int ow = 0; ow < OW; ++ow) {
                        const int iw = ow * stride_w - pad_w + kw * dilation_w;
                        if (iw < 0 || iw >= W)
                            continue;
                        dx[(c * H + ih) * W + iw] += col_row[oh * OW + ow];
                    }
                }
            }
        }
    }
}

// ============================ 3D ============================

template <typename T>
void im2col_3d_typed(const T* x,
                     T* cols,
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
                     int pw,
                     int dd,
                     int dh,
                     int dw) {
    const int M = OD * OH * OW;
    for (int c = 0; c < C; ++c) {
        for (int kd = 0; kd < KD; ++kd) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    const int row = ((c * KD + kd) * KH + kh) * KW + kw;
                    T* col_row = cols + row * M;
                    for (int od = 0; od < OD; ++od) {
                        const int id = od * sd - pd + kd * dd;
                        const bool d_in = (id >= 0 && id < D);
                        for (int oh = 0; oh < OH; ++oh) {
                            const int ih = oh * sh - ph + kh * dh;
                            const bool h_in = d_in && (ih >= 0 && ih < H);
                            T* row_oh = col_row + (od * OH + oh) * OW;
                            if (!h_in) {
                                for (int ow = 0; ow < OW; ++ow)
                                    row_oh[ow] = T{};
                                continue;
                            }
                            for (int ow = 0; ow < OW; ++ow) {
                                const int iw = ow * sw - pw + kw * dw;
                                row_oh[ow] =
                                    (iw >= 0 && iw < W) ? x[((c * D + id) * H + ih) * W + iw] : T{};
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void col2im_3d_typed(const T* cols,
                     T* dx,
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
                     int pw,
                     int dd,
                     int dh,
                     int dw) {
    const int M = OD * OH * OW;
    for (int c = 0; c < C; ++c) {
        for (int kd = 0; kd < KD; ++kd) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    const int row = ((c * KD + kd) * KH + kh) * KW + kw;
                    const T* col_row = cols + row * M;
                    for (int od = 0; od < OD; ++od) {
                        const int id = od * sd - pd + kd * dd;
                        if (id < 0 || id >= D)
                            continue;
                        for (int oh = 0; oh < OH; ++oh) {
                            const int ih = oh * sh - ph + kh * dh;
                            if (ih < 0 || ih >= H)
                                continue;
                            const T* row_oh = col_row + (od * OH + oh) * OW;
                            for (int ow = 0; ow < OW; ++ow) {
                                const int iw = ow * sw - pw + kw * dw;
                                if (iw < 0 || iw >= W)
                                    continue;
                                dx[((c * D + id) * H + ih) * W + iw] += row_oh[ow];
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

void im2col_1d_f32(
    const float* x, float* cols, int C, int L, int KL, int OL, int sl, int pl, int dl) {
    im2col_1d_typed<float>(x, cols, C, L, KL, OL, sl, pl, dl);
}
void im2col_1d_f64(
    const double* x, double* cols, int C, int L, int KL, int OL, int sl, int pl, int dl) {
    im2col_1d_typed<double>(x, cols, C, L, KL, OL, sl, pl, dl);
}
void col2im_1d_f32(
    const float* cols, float* dx, int C, int L, int KL, int OL, int sl, int pl, int dl) {
    col2im_1d_typed<float>(cols, dx, C, L, KL, OL, sl, pl, dl);
}
void col2im_1d_f64(
    const double* cols, double* dx, int C, int L, int KL, int OL, int sl, int pl, int dl) {
    col2im_1d_typed<double>(cols, dx, C, L, KL, OL, sl, pl, dl);
}

// ------------------------ 2D extern ------------------------

void im2col_f32(const float* x,
                float* cols,
                int C,
                int H,
                int W,
                int KH,
                int KW,
                int OH,
                int OW,
                int stride_h,
                int stride_w,
                int pad_h,
                int pad_w,
                int dilation_h,
                int dilation_w) {
    im2col_typed<float>(x, cols, C, H, W, KH, KW, OH, OW, stride_h, stride_w, pad_h, pad_w,
                        dilation_h, dilation_w);
}

void im2col_f64(const double* x,
                double* cols,
                int C,
                int H,
                int W,
                int KH,
                int KW,
                int OH,
                int OW,
                int stride_h,
                int stride_w,
                int pad_h,
                int pad_w,
                int dilation_h,
                int dilation_w) {
    im2col_typed<double>(x, cols, C, H, W, KH, KW, OH, OW, stride_h, stride_w, pad_h, pad_w,
                         dilation_h, dilation_w);
}

void col2im_f32(const float* cols,
                float* dx,
                int C,
                int H,
                int W,
                int KH,
                int KW,
                int OH,
                int OW,
                int stride_h,
                int stride_w,
                int pad_h,
                int pad_w,
                int dilation_h,
                int dilation_w) {
    col2im_typed<float>(cols, dx, C, H, W, KH, KW, OH, OW, stride_h, stride_w, pad_h, pad_w,
                        dilation_h, dilation_w);
}

void col2im_f64(const double* cols,
                double* dx,
                int C,
                int H,
                int W,
                int KH,
                int KW,
                int OH,
                int OW,
                int stride_h,
                int stride_w,
                int pad_h,
                int pad_w,
                int dilation_h,
                int dilation_w) {
    col2im_typed<double>(cols, dx, C, H, W, KH, KW, OH, OW, stride_h, stride_w, pad_h, pad_w,
                         dilation_h, dilation_w);
}

// ------------------------ 3D extern ------------------------

void im2col_3d_f32(const float* x,
                   float* cols,
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
                   int pw,
                   int dd,
                   int dh,
                   int dw) {
    im2col_3d_typed<float>(x, cols, C, D, H, W, KD, KH, KW, OD, OH, OW, sd, sh, sw, pd, ph, pw, dd,
                           dh, dw);
}
void im2col_3d_f64(const double* x,
                   double* cols,
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
                   int pw,
                   int dd,
                   int dh,
                   int dw) {
    im2col_3d_typed<double>(x, cols, C, D, H, W, KD, KH, KW, OD, OH, OW, sd, sh, sw, pd, ph, pw, dd,
                            dh, dw);
}
void col2im_3d_f32(const float* cols,
                   float* dx,
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
                   int pw,
                   int dd,
                   int dh,
                   int dw) {
    col2im_3d_typed<float>(cols, dx, C, D, H, W, KD, KH, KW, OD, OH, OW, sd, sh, sw, pd, ph, pw, dd,
                           dh, dw);
}
void col2im_3d_f64(const double* cols,
                   double* dx,
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
                   int pw,
                   int dd,
                   int dh,
                   int dw) {
    col2im_3d_typed<double>(cols, dx, C, D, H, W, KD, KH, KW, OD, OH, OW, sd, sh, sw, pd, ph, pw,
                            dd, dh, dw);
}

}  // namespace lucid::backend::cpu
