// lucid/_C/ops/utils/InplaceGraph.cpp
//
// Re-deriving a view family's graph after a write through one member —
// the half of InplaceGraph.h that needs a backward node of its own.

#include "InplaceGraph.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../core/Allocator.h"
#include "../../core/Shape.h"
#include "../../kernel/BinaryKernel.h"
#include "../composite/Indexing.h"
#include "../gfunc/Gfunc.h"
#include "Concat.h"
#include "Select.h"

namespace lucid::inplace {

namespace {

// A CPU buffer of ``nbytes`` zero bytes.
CpuStorage zeroed_cpu(std::size_t nbytes, Dtype dtype) {
    CpuStorage s;
    s.ptr = allocate_aligned_bytes(nbytes);
    s.nbytes = nbytes;
    s.dtype = dtype;
    if (nbytes > 0)
        std::memset(s.ptr.get(), 0, nbytes);
    return s;
}

// A 1-D int64 CPU tensor holding ``values``.
TensorImplPtr index_tensor(const std::vector<std::int64_t>& values) {
    CpuStorage s = zeroed_cpu(values.size() * sizeof(std::int64_t), Dtype::I64);
    if (!values.empty())
        std::memcpy(s.ptr.get(), values.data(), s.nbytes);
    return std::make_shared<TensorImpl>(Storage{std::move(s)},
                                        Shape{static_cast<std::int64_t>(values.size())}, Dtype::I64,
                                        Device::CPU, false);
}

// ``out[src[k]] += g[dst[k]]`` for one floating element type.
template <typename T>
void accumulate(const std::byte* g,
                std::byte* out,
                const std::vector<std::int64_t>& dst,
                const std::vector<std::int64_t>& src) {
    const auto* gp = reinterpret_cast<const T*>(g);
    auto* op = reinterpret_cast<T*>(out);
    for (std::size_t k = 0; k < dst.size(); ++k)
        op[src[k]] += gp[dst[k]];
}

// The gradient of a tensor after a write into some of its elements.
//
// The forward this stands for copies elements of ``src`` over elements of
// ``base``, both counted in row-major order:
//
//     out = base
//     out.flat[dst_begin : dst_begin + len] = src.flat[src_begin : src_begin + len]   (runs)
//     out.flat[dst_idx[k]] = src.flat[src_idx[k]]                                     (positions)
//
// the reference's CopySlices.  Two contiguous members meet in a run; a
// strided one — a transpose, a column — meets another at scattered
// positions, which the second form spells out.  The gradient of a copy does
// not depend on what was copied, so nothing is saved but where the elements
// sit — no tensor, and so nothing that holds the buffer and would stop the
// next write to it.  Either input may be absent: ``base`` when the write
// covered the member, ``src`` when what was written is a constant.  Its
// edge is then empty and it is given no gradient.
class CopySlicesBackward : public Node {
public:
    CopySlicesBackward(Shape out_shape,
                       Shape src_shape,
                       std::int64_t dst_begin,
                       std::int64_t src_begin,
                       std::int64_t len,
                       Dtype dtype,
                       Device device)
        : out_shape_(std::move(out_shape)),
          src_shape_(std::move(src_shape)),
          dst_begin_(dst_begin),
          src_begin_(src_begin),
          len_(len),
          dtype_(dtype),
          device_(device) {}

    CopySlicesBackward(Shape out_shape,
                       Shape src_shape,
                       std::vector<std::int64_t> dst_idx,
                       std::vector<std::int64_t> src_idx,
                       Dtype dtype,
                       Device device)
        : out_shape_(std::move(out_shape)),
          src_shape_(std::move(src_shape)),
          dtype_(dtype),
          device_(device),
          indexed_(true),
          dst_idx_(std::move(dst_idx)),
          src_idx_(std::move(src_idx)) {
        // A member that repeats an element (an expanded view) reads one
        // source element at several positions; their gradients add.
        std::vector<std::int64_t> sorted(src_idx_);
        std::sort(sorted.begin(), sorted.end());
        injective_ = std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
    }

    // ``base`` gets the output's gradient with the written elements zeroed —
    // they no longer depend on it — and ``src`` gets theirs, placed where
    // they were read from.  A live edge always gets a buffer: the engine
    // hands an empty one to the next node as it is.
    std::vector<Storage> apply(Storage grad_out) override {
        if (indexed_)
            return apply_indexed(grad_out);
        auto& be = backend::Dispatcher::for_device(device_);
        const auto n = static_cast<std::int64_t>(shape_numel(out_shape_));
        const Shape flat{n};
        std::vector<Storage> grads(2);
        if (next_edges_[0].node) {
            std::vector<Storage> pieces;
            std::vector<Shape> shapes;
            const std::int64_t tail = n - dst_begin_ - len_;
            if (dst_begin_ > 0) {
                pieces.push_back(be.slice_axis(grad_out, flat, Shape{dst_begin_}, 0, 0, dtype_));
                shapes.push_back(Shape{dst_begin_});
            }
            pieces.push_back(make_zero_storage(Shape{len_}, dtype_, device_));
            shapes.push_back(Shape{len_});
            if (tail > 0) {
                pieces.push_back(
                    be.slice_axis(grad_out, flat, Shape{tail}, 0, dst_begin_ + len_, dtype_));
                shapes.push_back(Shape{tail});
            }
            grads[0] = pieces.size() == 1 ? std::move(pieces[0])
                                          : be.concatenate(pieces, shapes, 0, dtype_);
        }
        if (next_edges_[1].node) {
            const auto src_n = static_cast<std::int64_t>(shape_numel(src_shape_));
            Storage run = len_ == n
                              ? grad_out
                              : be.slice_axis(grad_out, flat, Shape{len_}, 0, dst_begin_, dtype_);
            grads[1] = len_ == src_n ? std::move(run)
                                     : be.insert_axis_slice(run, Shape{len_}, Shape{src_n}, 0,
                                                            src_begin_, dtype_);
        }
        return grads;
    }

    // The same two gradients, built from ops so they carry a graph of their
    // own.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        if (indexed_)
            return graph_indexed(grad_out);
        const auto n = static_cast<std::int64_t>(shape_numel(out_shape_));
        const auto flat = reshape_op(grad_out, {n});
        const auto zeros = [&](std::int64_t count) {
            return zeros_op(Shape{count}, grad_out->dtype(), grad_out->device());
        };
        const auto join = [](const std::vector<TensorImplPtr>& pieces) {
            return pieces.size() == 1 ? pieces[0] : concatenate_op(pieces, 0);
        };
        std::vector<TensorImplPtr> grads(2);
        if (next_edges_[0].node) {
            std::vector<TensorImplPtr> pieces;
            const std::int64_t tail = n - dst_begin_ - len_;
            if (dst_begin_ > 0)
                pieces.push_back(narrow_op(flat, 0, 0, dst_begin_));
            pieces.push_back(zeros(len_));
            if (tail > 0)
                pieces.push_back(narrow_op(flat, 0, dst_begin_ + len_, tail));
            grads[0] = reshape_op(join(pieces), out_shape_);
        }
        if (next_edges_[1].node) {
            std::vector<TensorImplPtr> pieces;
            const std::int64_t after =
                static_cast<std::int64_t>(shape_numel(src_shape_)) - src_begin_ - len_;
            if (src_begin_ > 0)
                pieces.push_back(zeros(src_begin_));
            pieces.push_back(narrow_op(flat, 0, dst_begin_, len_));
            if (after > 0)
                pieces.push_back(zeros(after));
            grads[1] = reshape_op(join(pieces), src_shape_);
        }
        return grads;
    }

    std::string node_name() const override { return "CopySlices"; }

private:
    // View families live on the CPU, so their gradients do too.
    std::vector<Storage> apply_indexed(const Storage& grad_out) const {
        const auto* g = std::get_if<CpuStorage>(&grad_out);
        if (!g)
            ErrorBuilder("CopySlices").not_implemented("a gradient off the CPU");
        const std::size_t item = dtype_size(dtype_);
        std::vector<Storage> grads(2);
        if (next_edges_[0].node) {
            CpuStorage base = zeroed_cpu(shape_numel(out_shape_) * item, dtype_);
            if (base.nbytes > 0)
                std::memcpy(base.ptr.get(), g->ptr.get(), base.nbytes);
            for (const auto j : dst_idx_)
                std::memset(base.ptr.get() + j * item, 0, item);
            grads[0] = Storage{std::move(base)};
        }
        if (next_edges_[1].node) {
            CpuStorage src = zeroed_cpu(shape_numel(src_shape_) * item, dtype_);
            if (injective_) {
                for (std::size_t k = 0; k < dst_idx_.size(); ++k)
                    std::memcpy(src.ptr.get() + src_idx_[k] * item,
                                g->ptr.get() + dst_idx_[k] * item, item);
            } else if (dtype_ == Dtype::F32) {
                accumulate<float>(g->ptr.get(), src.ptr.get(), dst_idx_, src_idx_);
            } else if (dtype_ == Dtype::F64) {
                accumulate<double>(g->ptr.get(), src.ptr.get(), dst_idx_, src_idx_);
            } else if (dtype_ == Dtype::C64) {
                accumulate<std::complex<float>>(g->ptr.get(), src.ptr.get(), dst_idx_, src_idx_);
            } else {
                ErrorBuilder("CopySlices")
                    .not_implemented("the gradient of a repeated element in " +
                                     std::string(dtype_name(dtype_)));
            }
            grads[1] = Storage{std::move(src)};
        }
        return grads;
    }

    std::vector<TensorImplPtr> graph_indexed(const TensorImplPtr& grad_out) const {
        const auto n = static_cast<std::int64_t>(shape_numel(out_shape_));
        const auto flat = reshape_op(grad_out, {n});
        const auto dst = index_tensor(dst_idx_);
        const auto k = static_cast<std::int64_t>(dst_idx_.size());
        std::vector<TensorImplPtr> grads(2);
        if (next_edges_[0].node)
            grads[0] = reshape_op(
                scatter_op(flat, 0, dst, zeros_op(Shape{k}, grad_out->dtype(), grad_out->device())),
                out_shape_);
        if (next_edges_[1].node) {
            const auto src_n = static_cast<std::int64_t>(shape_numel(src_shape_));
            const auto picked = gather_op(flat, dst, 0);
            const auto zeros = zeros_op(Shape{src_n}, grad_out->dtype(), grad_out->device());
            grads[1] =
                reshape_op(scatter_add_op(zeros, index_tensor(src_idx_), picked, 0), src_shape_);
        }
        return grads;
    }

    Shape out_shape_;
    Shape src_shape_;
    std::int64_t dst_begin_ = 0;
    std::int64_t src_begin_ = 0;
    std::int64_t len_ = 0;
    Dtype dtype_;
    Device device_;
    bool indexed_ = false;
    bool injective_ = true;
    std::vector<std::int64_t> dst_idx_;
    std::vector<std::int64_t> src_idx_;
};

// Where a CPU tensor's elements sit, in its own row-major order: one run
// from ``first`` when the tensor is contiguous, one address per element
// otherwise.  ``first`` is null for any other storage.
struct Footprint {
    const std::byte* first = nullptr;
    std::size_t n = 0;
    std::size_t item = 0;
    std::vector<std::uintptr_t> addr;

    bool run() const { return addr.empty(); }
    std::uintptr_t at(std::size_t i) const {
        return run() ? reinterpret_cast<std::uintptr_t>(first) + i * item : addr[i];
    }
};

Footprint footprint(const TensorImpl& t) {
    Footprint f;
    const auto* cpu = std::get_if<CpuStorage>(&t.raw_storage());
    if (!cpu || !cpu->ptr)
        return f;
    f.first = cpu->ptr.get() + t.storage_offset();
    f.n = t.numel();
    f.item = dtype_size(t.dtype());
    if (t.is_contiguous() || f.n == 0)
        return f;
    const auto& shape = t.shape();
    const auto& stride = t.stride();
    f.addr.resize(f.n);
    std::vector<std::int64_t> idx(shape.size(), 0);
    for (std::size_t i = 0; i < f.n; ++i) {
        std::int64_t off = 0;
        for (std::size_t d = 0; d < shape.size(); ++d)
            off += idx[d] * stride[d];
        f.addr[i] = reinterpret_cast<std::uintptr_t>(f.first + off);
        for (std::size_t d = shape.size(); d-- > 0;) {
            if (++idx[d] < shape[d])
                break;
            idx[d] = 0;
        }
    }
    return f;
}

// Take ``m`` off the graph: the values it reads are a constant's now.
void cut(const TensorImplPtr& m) {
    if (!m->grad_fn())
        return;
    m->set_grad_fn(nullptr);
    m->set_grad_output_nr(0);
    m->set_requires_grad(false);
}

// Give ``m`` the graph ``node`` describes: its old graph as the base (unless
// the write covered it) and ``a``'s as the source (unless ``a`` holds a
// constant).
void attach(const TensorImplPtr& m,
            const TensorImplPtr& a,
            bool covered,
            bool differentiable,
            std::shared_ptr<CopySlicesBackward> node) {
    auto base = covered ? nullptr : detail::ensure_grad_fn(m);
    auto src = differentiable ? detail::ensure_grad_fn(a) : nullptr;
    if (!base && !src) {
        cut(m);
        return;
    }
    const std::uint32_t base_nr = base ? m->grad_output_nr() : 0;
    const std::uint32_t src_nr = src ? a->grad_output_nr() : 0;
    node->set_next_edges({Edge(std::move(base), base_nr), Edge(std::move(src), src_nr)});
    m->set_grad_fn(std::move(node));
    m->set_grad_output_nr(0);
    m->set_requires_grad(true);
    m->set_leaf(false);
}

}  // namespace

void rebase_views(const TensorImplPtr& a) {
    // A write through a detached alias (``.data``) is untracked by design,
    // and a detached member takes no part in autograd either way.
    if (a->is_detached_alias())
        return;
    const Footprint fa = footprint(*a);
    if (!fa.first)
        return;
    const bool differentiable = a->requires_grad();

    // Which of ``a``'s elements sits at an address, if any.  A run answers
    // by arithmetic; a strided ``a`` builds its table once, on first need.
    std::unordered_map<std::uintptr_t, std::int64_t> where;
    const auto locate = [&](std::uintptr_t x) -> std::int64_t {
        if (fa.run()) {
            const auto lo = reinterpret_cast<std::uintptr_t>(fa.first);
            if (x < lo || x >= lo + fa.n * fa.item || (x - lo) % fa.item != 0)
                return -1;
            return static_cast<std::int64_t>((x - lo) / fa.item);
        }
        if (where.empty()) {
            where.reserve(fa.n);
            for (std::size_t i = 0; i < fa.n; ++i)
                where.emplace(fa.addr[i], static_cast<std::int64_t>(i));
        }
        const auto it = where.find(x);
        return it == where.end() ? -1 : it->second;
    };

    for (const auto& m : a->live_views()) {
        if (m->is_detached_alias())
            continue;
        // A leaf keeps its place: it is where gradient accumulates, and its
        // AccumulateGrad is a grad_fn this must not cut.
        if (m->is_leaf() && m->requires_grad())
            continue;
        const Footprint fm = footprint(*m);
        if (!fm.first || fm.n == 0)
            continue;

        bool covered = false;
        bool same = false;
        std::shared_ptr<CopySlicesBackward> node;
        if (fa.run() && fm.run()) {
            // Two runs meet in a run.
            const std::byte* lo = std::max(fa.first, fm.first);
            const std::byte* hi = std::min(fa.first + fa.n * fa.item, fm.first + fm.n * fm.item);
            if (hi <= lo)
                continue;
            const auto item = static_cast<std::ptrdiff_t>(fa.item);
            const std::int64_t len = (hi - lo) / item;
            covered = len == static_cast<std::int64_t>(fm.n);
            same = covered && len == static_cast<std::int64_t>(fa.n);
            if (!same)
                node = std::make_shared<CopySlicesBackward>(
                    m->shape(), a->shape(), (lo - fm.first) / item, (lo - fa.first) / item, len,
                    a->dtype(), a->device());
        } else {
            // A strided side meets the other at scattered positions.
            std::vector<std::int64_t> dst;
            std::vector<std::int64_t> src;
            for (std::size_t j = 0; j < fm.n; ++j)
                if (const auto i = locate(fm.at(j)); i >= 0) {
                    dst.push_back(static_cast<std::int64_t>(j));
                    src.push_back(i);
                }
            if (dst.empty())
                continue;
            covered = dst.size() == fm.n;
            same = covered && fm.n == fa.n;
            for (std::size_t k = 0; same && k < src.size(); ++k)
                same = src[k] == static_cast<std::int64_t>(k);
            if (!same)
                node =
                    std::make_shared<CopySlicesBackward>(m->shape(), a->shape(), std::move(dst),
                                                         std::move(src), a->dtype(), a->device());
        }

        if (covered && !differentiable) {
            cut(m);
            continue;
        }
        if (same) {
            // The same elements in the same order: a reshape of ``a``.
            kernel::NaryKernel<ViewBackward, 1>::wire_autograd({a}, m, false);
            continue;
        }
        attach(m, a, covered, differentiable, std::move(node));
    }
}

}  // namespace lucid::inplace
