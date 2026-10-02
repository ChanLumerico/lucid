// lucid/_C/nn/Embedding.cpp
//
// Implementation of embedding lookup, sinusoidal position encoding, and
// Rotary Position Embedding (RoPE).
//
// Embedding forward: IBackend::embedding_forward gathers rows from the weight
//   matrix at positions given by indices, padding_idx included — it does not
//   mask the lookup.  Output shape: (*indices.shape, embed_dim).
// Embedding backward: IBackend::embedding_backward scatter-adds grad_out into
//   a zero-initialized weight gradient, skipping positions at padding_idx.
//
// Sinusoidal encoding: IBackend::sinusoidal_pos_embedding fills the (L, D)
//   matrix with sin/cos at geometrically spaced frequencies.  No grad node.
//
// RoPE forward: IBackend::rope_forward returns {rotated_input, cos, sin}.
//   cos and sin are saved for the backward.
// RoPE backward: IBackend::rope_backward applies the inverse rotation to
//   grad_out using the saved cos/sin tables.

#include "Embedding.h"

#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../backend/Dispatcher.h"
#include "../backend/gpu/MlxBridge.h"
#include "../compile/Tracer.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/Helpers.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/Compare.h"
#include "../ops/bfunc/Mul.h"
#include "../ops/bfunc/_BinaryOp.h"
#include "../ops/gfunc/Gfunc.h"
#include "../ops/ufunc/Astype.h"
#include "../ops/utils/Layout.h"
#include "../ops/utils/Select.h"
#include "../ops/utils/View.h"

namespace lucid {

const OpSchema EmbeddingBackward::schema_v1{"embedding", 1, AmpPolicy::Promote, true};

TensorImplPtr EmbeddingBackward::forward(const TensorImplPtr& weight,
                                         const TensorImplPtr& indices,
                                         int padding_idx) {
    if (!weight || !indices)
        ErrorBuilder("embedding").fail("null input");
    if (weight->device() != indices->device())
        throw DeviceMismatch(std::string(device_name(weight->device())),
                             std::string(device_name(indices->device())),
                             "embedding: weight/indices");
    if (weight->shape().size() != 2)
        throw ShapeMismatch(weight->shape(), Shape{},
                            "embedding: weight must be 2-D (num_embeddings, dim)");

    const std::int64_t D = weight->shape()[1];

    Shape out_shape = indices->shape();
    out_shape.push_back(D);
    OpScopeFull scope{schema_v1.name, weight->device(), weight->dtype(), out_shape};
    scope.set_attr("padding_idx", static_cast<std::int64_t>(padding_idx));

    auto& be = backend::Dispatcher::for_device(weight->device());
    Storage out_storage =
        be.embedding_forward(weight->storage(), indices->storage(), weight->shape(),
                             indices->shape(), out_shape, padding_idx, weight->dtype());

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, weight->dtype(),
                                            weight->device(), false);

    auto bwd = std::make_shared<EmbeddingBackward>();
    bwd->saved_indices_ = indices->storage();
    bwd->saved_indices_shape_ = indices->shape();
    bwd->saved_indices_dtype_ = indices->dtype();
    bwd->padding_idx_ = padding_idx;
    bwd->weight_shape_ = weight->shape();
    kernel::NaryKernel<EmbeddingBackward, 1>::wire_autograd(std::move(bwd), {weight}, out, false);

    // The autograd wiring above only registers ``weight`` as a trace
    // input (indices is an int tensor — not differentiable), but the
    // compile path needs BOTH so the ``embedding`` emitter can call
    // ``gatherWithUpdatesTensor:indicesTensor:``.  Overwrite the trace
    // entry now (Tracer::on_op_io is last-write-wins).
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({weight, indices}, out);
    }
    return out;
}

std::vector<Storage> EmbeddingBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    return {be.embedding_backward(grad_out, saved_indices_, weight_shape_, saved_indices_shape_,
                                  padding_idx_, dtype_)};
}

std::vector<TensorImplPtr> EmbeddingBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    // dW[r] = sum of g over the positions that looked row r up, except the
    // padding row, which takes none: a scatter-add of g's rows by index.
    const std::int64_t rows = weight_shape_[0];
    const std::int64_t width = weight_shape_[1];
    std::int64_t looked = 1;
    for (auto d : saved_indices_shape_)
        looked *= d;
    auto indices = std::make_shared<TensorImpl>(saved_indices_, saved_indices_shape_,
                                                saved_indices_dtype_, device_, false);
    auto index = broadcast_to_op(reshape_op(indices, {looked, 1}), Shape{looked, width});
    auto src = reshape_op(grad_out, {looked, width});
    auto base = zeros_op(weight_shape_, dtype_, device_);
    TensorImplPtr dw = scatter_add_op(base, index, src, 0);
    if (padding_idx_ >= 0 && padding_idx_ < rows) {
        auto row = arange_op(0.0, static_cast<double>(rows), 1.0, Dtype::I64, device_);
        auto padding = equal_op(row, full_like_op(row, static_cast<double>(padding_idx_)));
        auto keep = broadcast_to_op(reshape_op(padding, {rows, 1}), weight_shape_);
        dw = where_op(keep, zeros_like_op(dw), dw);
    }
    return {dw};
}

TensorImplPtr
embedding_op(const TensorImplPtr& weight, const TensorImplPtr& indices, int padding_idx) {
    return EmbeddingBackward::forward(weight, indices, padding_idx);
}
LUCID_REGISTER_OP(EmbeddingBackward)

TensorImplPtr sinusoidal_pos_embedding_op(std::int64_t seq_len,
                                          std::int64_t embed_dim,
                                          Dtype dtype,
                                          Device device) {
    if (seq_len < 0)
        ErrorBuilder("sinusoidal_pos_embedding").fail("seq_len < 0");
    if (embed_dim <= 0)
        ErrorBuilder("sinusoidal_pos_embedding").fail("embed_dim must be > 0");

    Shape out_shape{seq_len, embed_dim};
    OpScopeFull scope{"sinusoidal_pos_embedding", device, dtype, out_shape};

    auto& be = backend::Dispatcher::for_device(device);
    Storage out_s = be.sinusoidal_pos_embedding(seq_len, embed_dim, dtype);
    return std::make_shared<TensorImpl>(std::move(out_s), out_shape, dtype, device, false);
}

const OpSchema RotaryPosEmbeddingBackward::schema_v1{"rotary_pos_embedding", 1,
                                                     AmpPolicy::ForceFP32, true};

TensorImplPtr RotaryPosEmbeddingBackward::forward(const TensorImplPtr& input,
                                                  const TensorImplPtr& position_ids_or_null,
                                                  bool interleaved) {
    Validator::input(input, "rotary_pos_embedding.input").non_null();
    if (position_ids_or_null && position_ids_or_null->device() != input->device())
        throw DeviceMismatch(std::string(device_name(input->device())),
                             std::string(device_name(position_ids_or_null->device())),
                             "rotary_pos_embedding: input/position_ids");
    if (input->shape().size() < 2)
        ErrorBuilder("rotary_pos_embedding").fail("input must be at least 2-D ([..., L, D])");

    const std::size_t ndim = input->shape().size();
    const std::size_t D = static_cast<std::size_t>(input->shape()[ndim - 1]);
    if (D % 2 != 0)
        ErrorBuilder("rotary_pos_embedding").fail("embed_dim must be even");

    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(), input->shape()};
    scope.set_attr("interleaved", interleaved);
    scope.set_attr("has_pos_ids", position_ids_or_null != nullptr);

    const Storage* pos_storage = position_ids_or_null ? &position_ids_or_null->storage() : nullptr;
    const Dtype pos_dt = position_ids_or_null ? position_ids_or_null->dtype() : Dtype::I64;
    auto& be = backend::Dispatcher::for_device(input->device());
    auto rope_out = be.rope_forward(input->storage(), pos_storage, input->shape(), interleaved,
                                    pos_dt, input->dtype());

    Storage out_storage = std::move(rope_out[0]);

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), input->shape(), input->dtype(),
                                            input->device(), false);
    // wire_autograd records on_op_io internally — no explicit call.

    {
        auto bwd = std::make_shared<RotaryPosEmbeddingBackward>();
        bwd->saved_cos_ = std::move(rope_out[1]);
        bwd->saved_sin_ = std::move(rope_out[2]);
        bwd->interleaved_ = interleaved;
        bwd->orig_shape_ = input->shape();
        kernel::NaryKernel<RotaryPosEmbeddingBackward, 1>::wire_autograd(std::move(bwd), {input},
                                                                         out, false);
    }
    return out;
}

std::vector<Storage> RotaryPosEmbeddingBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    return {be.rope_backward(grad_out, saved_cos_, saved_sin_, orig_shape_, interleaved_, dtype_)};
}

TensorImplPtr rotary_pos_embedding_op(const TensorImplPtr& input,
                                      const TensorImplPtr& position_ids_or_null,
                                      bool interleaved) {
    return RotaryPosEmbeddingBackward::forward(input, position_ids_or_null, interleaved);
}
LUCID_REGISTER_OP(RotaryPosEmbeddingBackward)

// ── EmbeddingBag ─────────────────────────────────────────────────────────────

TensorImplPtr embedding_bag_op(const TensorImplPtr& weight,
                               const TensorImplPtr& indices,
                               const TensorImplPtr& offsets,
                               int mode,
                               int padding_idx,
                               bool include_last_offset) {
    Validator::input(weight, "embedding_bag.weight").non_null();
    Validator::input(indices, "embedding_bag.indices").non_null();
    Validator::input(offsets, "embedding_bag.offsets").non_null();

    // ``embedding`` has always checked this; ``embedding_bag`` did not, and
    // read ``shape()[1]`` off a rank-1 table anyway.  What came back sized
    // the output, so a 1-D weight produced an empty ``(B, 0)`` — the right
    // dtype, a plausible shape, and no embeddings in it.
    if (weight->shape().size() != 2)
        throw ShapeMismatch(weight->shape(), Shape{},
                            "embedding_bag: weight must be 2-D (num_embeddings, dim)");

    // With ``include_last_offset`` the final offset is a sentinel — where
    // the last bag ends — so there is one bag fewer than offsets.  Counting
    // it as a bag added an empty output row and, below, ran every other bag
    // to the end of the index buffer.
    const std::int64_t n_offsets = offsets->shape().empty() ? 0 : offsets->shape()[0];
    if (include_last_offset && n_offsets < 1)
        ErrorBuilder("embedding_bag")
            .fail("include_last_offset=True needs at least one offset (the end of the last bag)");
    const int B = static_cast<int>(n_offsets - (include_last_offset ? 1 : 0));
    const int D = static_cast<int>(weight->shape()[1]);
    Shape out_shape = {static_cast<std::int64_t>(B), static_cast<std::int64_t>(D)};
    OpScopeFull scope{"embedding_bag", weight->device(), weight->dtype(), out_shape};

    auto& be = backend::Dispatcher::for_device(weight->device());
    Storage out_storage = be.embedding_bag_forward(
        weight->storage(), indices->storage(), offsets->storage(), weight->shape(),
        indices->shape(), mode, padding_idx, include_last_offset, weight->dtype());

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, weight->dtype(),
                                            weight->device(), false);

    // This op used to return here, with no backward of any kind.
    //
    // ``embedding`` directly above wires one; this did not, and nothing
    // said so: the forward was right, the output simply had no grad_fn,
    // so ``weight.grad`` stayed ``None`` and an ``nn.EmbeddingBag`` layer
    // silently never trained.  No error, no NaN — the loss just never
    // moved through it.  The reference gives a gradient here.
    auto bwd = std::make_shared<EmbeddingBagBackward>();
    bwd->saved_weight_ = weight->storage();
    bwd->saved_indices_ = indices->storage();
    bwd->saved_offsets_ = offsets->storage();
    bwd->weight_shape_ = weight->shape();
    bwd->indices_shape_ = indices->shape();
    bwd->mode_ = mode;
    bwd->padding_idx_ = padding_idx;
    bwd->include_last_offset_ = include_last_offset;
    bwd->dtype_ = weight->dtype();
    bwd->device_ = weight->device();
    kernel::NaryKernel<EmbeddingBagBackward, 1>::wire_autograd(std::move(bwd), {weight}, out,
                                                               false);
    return out;
}

const OpSchema EmbeddingBagBackward::schema_v1{"embedding_bag", 1, AmpPolicy::Promote, true};

std::vector<Storage> EmbeddingBagBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    return {be.embedding_bag_backward(grad_out, saved_weight_, saved_indices_, saved_offsets_,
                                      weight_shape_, indices_shape_, mode_, padding_idx_,
                                      include_last_offset_, dtype_)};
}

namespace {

// A saved storage's bytes on the host, whichever device holds them.
CpuStorage host_copy(const Storage& s, const Shape& shape) {
    if (const auto* cpu = std::get_if<CpuStorage>(&s))
        return *cpu;
    return gpu::download_gpu_to_cpu(std::get<GpuStorage>(s), shape);
}

std::int64_t read_index(const CpuStorage& s, std::size_t k) {
    return s.dtype == Dtype::I64 ? reinterpret_cast<const std::int64_t*>(s.ptr.get())[k]
                                 : reinterpret_cast<const std::int32_t*>(s.ptr.get())[k];
}

}  // namespace

std::vector<TensorImplPtr> EmbeddingBagBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    // The same rows, bags and weights the eager backward uses, as data; the
    // scatter-add that lands the bag gradients on the rows is the op.
    const std::int64_t num_emb = weight_shape_[0];
    const std::int64_t dim = weight_shape_[1];
    std::size_t n_idx = 1;
    for (auto d : indices_shape_)
        n_idx *= static_cast<std::size_t>(d);
    const CpuStorage indices = host_copy(saved_indices_, indices_shape_);
    const std::size_t off_elem = std::holds_alternative<CpuStorage>(saved_offsets_)
                                     ? dtype_size(std::get<CpuStorage>(saved_offsets_).dtype)
                                     : dtype_size(std::get<GpuStorage>(saved_offsets_).dtype);
    const std::size_t n_offsets = storage_nbytes(saved_offsets_) / off_elem;
    const CpuStorage offsets =
        host_copy(saved_offsets_, Shape{static_cast<std::int64_t>(n_offsets)});

    // Bag b is [offsets[b], offsets[b + 1]); the last one ends at the
    // sentinel offset under ``include_last_offset``, else at the end of the
    // indices — the boundaries the forward used.
    const std::size_t n_bags = n_offsets - (include_last_offset_ ? 1 : 0);
    std::vector<std::size_t> starts(n_bags), ends(n_bags);
    for (std::size_t b = 0; b < n_bags; ++b) {
        starts[b] = static_cast<std::size_t>(read_index(offsets, b));
        ends[b] = b + 1 < n_offsets ? static_cast<std::size_t>(read_index(offsets, b + 1)) : n_idx;
    }
    const auto usable = [&](std::int64_t emb) {
        return emb != padding_idx_ && emb >= 0 && emb < num_emb;
    };

    const Shape weight_shape{num_emb, dim};
    auto base = zeros_op(weight_shape, dtype_, device_);
    const auto upload = [&](CpuStorage cpu, const Shape& shape, Dtype dt) {
        return std::make_shared<TensorImpl>(
            backend::Dispatcher::for_device(device_).from_cpu(std::move(cpu), shape), shape, dt,
            device_, false);
    };

    if (mode_ == 2) {
        // max: each (bag, column) sends its gradient to the row that won it,
        // first on ties, as the forward's strict ``>`` decided.
        const CpuStorage weight = host_copy(saved_weight_, weight_shape);
        const Shape grid{static_cast<std::int64_t>(n_bags), dim};
        CpuStorage rows = helpers::allocate_cpu(grid, Dtype::I32);
        CpuStorage keep = helpers::allocate_cpu(grid, Dtype::F64);
        auto* row = reinterpret_cast<std::int32_t*>(rows.ptr.get());
        auto* kept = reinterpret_cast<double*>(keep.ptr.get());
        const auto value = [&](std::int64_t emb, std::int64_t d) -> double {
            const std::size_t at = static_cast<std::size_t>(emb * dim + d);
            switch (weight.dtype) {
            case Dtype::F64:
                return reinterpret_cast<const double*>(weight.ptr.get())[at];
            case Dtype::F32:
                return reinterpret_cast<const float*>(weight.ptr.get())[at];
            default:
                ErrorBuilder("embedding_bag").not_implemented("create_graph=True for this dtype");
                return 0.0;
            }
        };
        for (std::size_t b = 0; b < n_bags; ++b)
            for (std::int64_t d = 0; d < dim; ++d) {
                std::int64_t best = -1;
                double best_val = 0.0;
                for (std::size_t k = starts[b]; k < ends[b]; ++k) {
                    const std::int64_t emb = read_index(indices, k);
                    if (!usable(emb))
                        continue;
                    const double v = value(emb, d);
                    if (best < 0 || v > best_val) {
                        best = emb;
                        best_val = v;
                    }
                }
                const std::size_t at = b * static_cast<std::size_t>(dim) + d;
                row[at] = static_cast<std::int32_t>(best < 0 ? 0 : best);
                kept[at] = best < 0 ? 0.0 : 1.0;
            }
        auto keep_t = astype_op(upload(std::move(keep), grid, Dtype::F64), dtype_);
        auto src = mul_op(grad_out, keep_t);
        return {scatter_add_op(base, upload(std::move(rows), grid, Dtype::I32), src, 0)};
    }

    // sum / mean: every usable index k sends its bag's gradient, scaled by
    // 1 / count for mean, to its row.
    std::vector<std::int32_t> bag_of, row_of;
    std::vector<double> scale_of;
    for (std::size_t b = 0; b < n_bags; ++b) {
        std::size_t count = 0;
        for (std::size_t k = starts[b]; k < ends[b]; ++k)
            count += usable(read_index(indices, k)) ? 1 : 0;
        for (std::size_t k = starts[b]; k < ends[b]; ++k) {
            const std::int64_t emb = read_index(indices, k);
            if (!usable(emb))
                continue;
            bag_of.push_back(static_cast<std::int32_t>(b));
            row_of.push_back(static_cast<std::int32_t>(emb));
            scale_of.push_back(mode_ == 1 ? 1.0 / static_cast<double>(count) : 1.0);
        }
    }
    const std::int64_t used = static_cast<std::int64_t>(bag_of.size());
    if (used == 0)
        return {base};
    const Shape pick{used, dim};
    CpuStorage bags = helpers::allocate_cpu(pick, Dtype::I32);
    CpuStorage rows = helpers::allocate_cpu(pick, Dtype::I32);
    CpuStorage scales = helpers::allocate_cpu(pick, Dtype::F64);
    for (std::int64_t k = 0; k < used; ++k)
        for (std::int64_t d = 0; d < dim; ++d) {
            const std::size_t at = static_cast<std::size_t>(k * dim + d);
            reinterpret_cast<std::int32_t*>(bags.ptr.get())[at] = bag_of[k];
            reinterpret_cast<std::int32_t*>(rows.ptr.get())[at] = row_of[k];
            reinterpret_cast<double*>(scales.ptr.get())[at] = scale_of[k];
        }
    auto src = gather_op(grad_out, upload(std::move(bags), pick, Dtype::I32), 0);
    src = mul_op(src, astype_op(upload(std::move(scales), pick, Dtype::F64), dtype_));
    return {scatter_add_op(base, upload(std::move(rows), pick, Dtype::I32), src, 0)};
}

LUCID_REGISTER_OP(EmbeddingBagBackward)

}  // namespace lucid
