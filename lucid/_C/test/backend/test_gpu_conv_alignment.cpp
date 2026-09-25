// Tests that no Metal convolution reaches MLX with channels MLX would pad.
//
// MLX pads unaligned convolution channels to a multiple of 16 inside its
// Metal kernel and fills the padding from a zero scalar it frees before the
// GPU reads it, so a buffer claimed in between turns the padding into
// garbage.  The GPU backend aligns the channels itself
// (``GpuBackend::gpu_align_conv_channels``); these tests walk the MLX graph a
// convolution builds and check every ``Convolution`` node gets aligned
// operands.  Values are checked against the CPU by the Python suite
// (``test_conv_channel_alignment.py``); what matters here is the invariant,
// which a race cannot hide.

#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>
#include <mlx/array.h>
#include <mlx/primitives.h>

#include "backend/Dispatcher.h"
#include "backend/IBackend.h"
#include "core/Storage.h"

using namespace lucid;

namespace {

// Every ``Convolution`` node reachable from ``roots``.
std::vector<mlx::core::array> convolutions(const std::vector<Storage>& roots) {
    std::vector<mlx::core::array> found;
    std::unordered_set<std::uintptr_t> seen;
    std::function<void(const mlx::core::array&)> walk = [&](const mlx::core::array& a) {
        if (!seen.insert(a.id()).second)
            return;
        if (a.has_primitive() && std::string(a.primitive().name()) == "Convolution")
            found.push_back(a);
        for (const auto& in : a.inputs())
            walk(in);
    };
    for (const auto& s : roots)
        if (const auto* g = std::get_if<GpuStorage>(&s); g && g->arr)
            walk(*g->arr);
    return found;
}

// The operand shapes MLX's own padding checks (``conv.cpp``): the input's
// channels are its last axis, the weight's output channels its first.
void expect_aligned(const std::vector<mlx::core::array>& convs, int rank) {
    ASSERT_FALSE(convs.empty()) << "no Convolution node in the graph";
    for (const auto& c : convs) {
        const auto& in = c.inputs()[0];
        const auto& wt = c.inputs()[1];
        const int C = static_cast<int>(in.shape().back());
        const int O = static_cast<int>(wt.shape()[0]);
        if (rank == 2)
            EXPECT_TRUE(C <= 4 || C % 16 == 0) << "input channels " << C;
        else
            EXPECT_EQ(C % 16, 0) << "input channels " << C;
        EXPECT_TRUE(O <= 16 || O % 16 == 0) << "output channels " << O;
    }
}

struct Case {
    int rank;
    int Cin;
    int Cout;
    int size;
};

void check(const Case& c) {
    auto& be = backend::Dispatcher::for_device(Device::GPU);
    const int B = 2;
    const int k = 3;
    Shape xs{B, c.Cin};
    Shape ws{c.Cout, c.Cin};
    Shape os{B, c.Cout};
    int S[3] = {1, 1, 1};
    int K[3] = {1, 1, 1};
    int O[3] = {1, 1, 1};
    for (int i = 0; i < c.rank; ++i) {
        xs.push_back(c.size);
        ws.push_back(k);
        os.push_back(c.size);  // stride 1, pad 1, kernel 3
        S[i] = c.size;
        K[i] = k;
        O[i] = c.size;
    }
    auto x = be.ones(xs, Dtype::F32);
    auto w = be.ones(ws, Dtype::F32);
    auto b = be.zeros(Shape{c.Cout}, Dtype::F32);
    backend::IBackend::ConvNdOpts opts{c.rank, 1, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}};

    auto y =
        be.conv_nd_forward(x, w, b, B, c.Cin, c.Cout, c.Cin, c.Cout, S, K, O, opts, os, Dtype::F32);
    expect_aligned(convolutions({y}), c.rank);

    auto g = be.ones(os, Dtype::F32);
    auto grads =
        be.conv_nd_backward(g, x, w, B, c.Cin, c.Cout, c.Cin, c.Cout, S, K, O, opts, Dtype::F32);
    expect_aligned(convolutions(grads), c.rank);
}

}  // namespace

// The ResNet-on-MNIST block that exposed it: 8 -> 8, 3x3, stride 1.
TEST(GpuConvAlignment, Conv2dEightToEight) {
    check({2, 8, 8, 16});
}

TEST(GpuConvAlignment, Conv2dUnalignedIntoAligned) {
    check({2, 12, 32, 16});
}

// Three-channel video input, unaligned output channels: MLX would pad both.
TEST(GpuConvAlignment, Conv3dRgbIntoTwentyFour) {
    check({3, 3, 24, 6});
}
