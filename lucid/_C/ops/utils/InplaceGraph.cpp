// lucid/_C/ops/utils/InplaceGraph.cpp
//
// Re-deriving a view family's graph after a write through one member —
// the half of InplaceGraph.h that needs a backward node of its own.

#include "InplaceGraph.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../core/Shape.h"
#include "../../kernel/BinaryKernel.h"
#include "../composite/Indexing.h"
#include "../gfunc/Gfunc.h"
#include "Concat.h"

namespace lucid::inplace {

namespace {

// The gradient of a tensor after a write into a run of its elements.
//
// The forward this stands for copies a run of ``src`` over a run of
// ``base``, both taken in row-major order:
//
//     out = base
//     out.flat[dst_begin : dst_begin + len] = src.flat[src_begin : src_begin + len]
//
// the reference's CopySlices, with runs where it has strided slices: every
// member of a view family reads one run of its buffer.  The gradient of a
// copy does not depend on what was copied, so nothing is saved but where
// the runs sit — no tensor, and so nothing that holds the buffer and would
// stop the next write to it.  Either input may be absent: ``base`` when the
// write covered the member, ``src`` when what was written is a constant.
// Its edge is then empty and it is given no gradient.
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

    // ``base`` gets the output's gradient with the written run zeroed —
    // those elements no longer depend on it — and ``src`` gets that run,
    // placed where it was read from.  A live edge always gets a buffer: the
    // engine hands an empty one to the next node as it is.
    std::vector<Storage> apply(Storage grad_out) override {
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
    Shape out_shape_;
    Shape src_shape_;
    std::int64_t dst_begin_;
    std::int64_t src_begin_;
    std::int64_t len_;
    Dtype dtype_;
    Device device_;
};

// Where a CPU tensor's elements start, or null for any other storage.
const std::byte* first_byte(const TensorImpl& t) {
    const auto* cpu = std::get_if<CpuStorage>(&t.storage());
    return cpu && cpu->ptr ? cpu->ptr.get() + t.storage_offset() : nullptr;
}

// Take ``m`` off the graph: the values it reads are a constant's now.
void cut(const TensorImplPtr& m) {
    if (!m->grad_fn())
        return;
    m->set_grad_fn(nullptr);
    m->set_grad_output_nr(0);
    m->set_requires_grad(false);
}

}  // namespace

void rebase_views(const TensorImplPtr& a) {
    // A write through a detached alias (``.data``) is untracked by design,
    // and a detached member takes no part in autograd either way.
    if (a->is_detached_alias())
        return;
    const std::byte* a_first = first_byte(*a);
    if (!a_first)
        return;
    const std::byte* a_end = a_first + a->nbytes();
    const auto item = static_cast<std::ptrdiff_t>(dtype_size(a->dtype()));
    const bool differentiable = a->requires_grad();
    for (const auto& m : a->live_views()) {
        if (m->is_detached_alias())
            continue;
        // A leaf keeps its place: it is where gradient accumulates, and its
        // AccumulateGrad is a grad_fn this must not cut.
        if (m->is_leaf() && m->requires_grad())
            continue;
        const std::byte* m_first = first_byte(*m);
        if (!m_first)
            continue;
        const std::byte* lo = std::max(a_first, m_first);
        const std::byte* hi = std::min(a_end, m_first + m->nbytes());
        if (hi <= lo)
            continue;
        const std::int64_t len = (hi - lo) / item;
        const bool covered = len == static_cast<std::int64_t>(m->numel());
        if (covered && !differentiable) {
            cut(m);
            continue;
        }
        if (covered && len == static_cast<std::int64_t>(a->numel())) {
            kernel::NaryKernel<ViewBackward, 1>::wire_autograd({a}, m, false);
            continue;
        }
        auto base = covered ? nullptr : detail::ensure_grad_fn(m);
        auto src = differentiable ? detail::ensure_grad_fn(a) : nullptr;
        if (!base && !src) {
            cut(m);
            continue;
        }
        auto node = std::make_shared<CopySlicesBackward>(
            m->shape(), a->shape(), (lo - m_first) / item, (lo - a_first) / item, len, a->dtype(),
            a->device());
        const std::uint32_t base_nr = base ? m->grad_output_nr() : 0;
        const std::uint32_t src_nr = src ? a->grad_output_nr() : 0;
        node->set_next_edges({Edge(std::move(base), base_nr), Edge(std::move(src), src_nr)});
        m->set_grad_fn(std::move(node));
        m->set_grad_output_nr(0);
        m->set_requires_grad(true);
        m->set_leaf(false);
    }
}

}  // namespace lucid::inplace
