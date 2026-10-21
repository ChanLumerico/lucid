// lucid/_C/test/core/test_storage_window.cpp
// LCD-517: unit tests for CpuStorage window metadata and TensorImpl
// allocation-frame accessors (allocation_offset, allocation_nbytes,
// allocation_base, make_block_view, make_allocation_view).

#include <gtest/gtest.h>

#include "../../core/Allocator.h"
#include "../../core/Dtype.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"

using namespace lucid;

namespace {

std::shared_ptr<TensorImpl> make_cpu_f32(Shape shape) {
    const std::size_t nbytes = shape_numel(shape) * sizeof(float);
    CpuStorage cpu{allocate_aligned_bytes(nbytes, Device::CPU), nbytes, Dtype::F32};
    auto* ptr = reinterpret_cast<float*>(cpu.ptr.get());
    for (std::size_t i = 0; i < shape_numel(shape); ++i)
        ptr[i] = static_cast<float>(i);
    return std::make_shared<TensorImpl>(Storage{std::move(cpu)}, std::move(shape), Dtype::F32,
                                        Device::CPU, false);
}

}  // namespace

TEST(StorageWindowTest, BlockViewPreservesDenseHandleAndAllocationFrame) {
    auto root = make_cpu_f32({3, 6});
    const std::uintptr_t root_base = root->allocation_base();
    EXPECT_EQ(root->storage_offset(), 0u);
    EXPECT_EQ(root->allocation_offset(), 0u);
    EXPECT_EQ(root->allocation_nbytes(), 72u);
    EXPECT_TRUE(root->is_dense());

    auto row1 = TensorImpl::make_block_view(root, {6}, 6 * sizeof(float));
    EXPECT_EQ(row1->storage_offset(), 0u);
    EXPECT_EQ(row1->allocation_offset(), 24u);
    EXPECT_EQ(row1->allocation_nbytes(), 72u);
    EXPECT_EQ(row1->allocation_base(), root_base);
    EXPECT_TRUE(row1->is_dense());
    EXPECT_TRUE(row1->is_aliased());
    EXPECT_TRUE(root->is_aliased());

    auto nested = TensorImpl::make_block_view(row1, {3}, 2 * sizeof(float));
    EXPECT_EQ(nested->storage_offset(), 0u);
    EXPECT_EQ(nested->allocation_offset(), 32u);
    EXPECT_EQ(nested->allocation_nbytes(), 72u);
    EXPECT_EQ(nested->allocation_base(), root_base);
    EXPECT_TRUE(nested->is_dense());
    EXPECT_TRUE(nested->is_aliased());
    root->bump_version();
    EXPECT_EQ(row1->version(), 1);
    EXPECT_EQ(nested->version(), 1);

    auto strided = TensorImpl::make_view(row1, {3}, {2 * static_cast<std::int64_t>(sizeof(float))},
                                         sizeof(float));
    EXPECT_EQ(strided->storage_offset(), sizeof(float));
    EXPECT_EQ(strided->allocation_offset(), 28u);
    EXPECT_EQ(strided->allocation_nbytes(), 72u);
    EXPECT_EQ(strided->allocation_base(), root_base);
}

TEST(StorageWindowTest, MakeAllocationViewReachesRootFromNarrowedBlock) {
    auto root = make_cpu_f32({3, 6});
    auto row1 = TensorImpl::make_block_view(root, {6}, 6 * sizeof(float));
    auto win =
        TensorImpl::make_allocation_view(row1, {2}, {static_cast<std::int64_t>(sizeof(float))}, 0);
    EXPECT_EQ(win->allocation_offset(), 0u);
    EXPECT_EQ(win->allocation_nbytes(), 72u);
    EXPECT_EQ(win->allocation_base(), root->allocation_base());
    EXPECT_TRUE(win->is_aliased());
    win->bump_version();
    EXPECT_EQ(root->version(), 1);

    const auto& cpu = std::get<CpuStorage>(win->raw_storage());
    const auto* vals = reinterpret_cast<const float*>(cpu.ptr.get() + win->storage_offset());
    EXPECT_FLOAT_EQ(vals[0], 0.0f);
    EXPECT_FLOAT_EQ(vals[1], 1.0f);
}

TEST(StorageWindowTest, IsContiguousIgnoresSizeOneAndEmptyAxes) {
    auto root = make_cpu_f32({1, 8});
    auto transposed =
        TensorImpl::make_view(root, {8, 1}, {static_cast<std::int64_t>(sizeof(float)), 32}, 0);
    EXPECT_TRUE(transposed->is_contiguous());

    auto one_by_one = TensorImpl::make_block_view(transposed, {1, 1}, 0);
    EXPECT_TRUE(one_by_one->is_contiguous());

    auto empty = TensorImpl::make_view(root, {0, 4}, {32, 8}, 0);
    EXPECT_TRUE(empty->is_contiguous());
}
