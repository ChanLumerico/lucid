// lucid/_C/coreml/MilProgram.h
//
// In-memory MIL program, and its serialisation to a Core ML ``Model``
// protobuf.
//
// The shape of what gets written was read off a reference package rather
// than inferred from the schema, because the schema permits far more than
// Core ML accepts:
//
//   * every operation carries a ``name`` attribute (a string ``Value``),
//     including ``const``;
//   * operands are bound by *name* (``Binding.name``), never inline —
//     even a scalar like ``groups`` is its own ``const`` operation;
//   * a scalar value's ``tensorType`` carries only ``dataType``: rank 0 is
//     proto3's default and is left off;
//   * the block's single output is named in ``Block.outputs``.
//
// Field numbers come from the generated :file:`MilSchema.h`.

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "../api.h"

namespace lucid::coreml {

// MIL element types (``MIL.proto``'s ``DataType``).  Distinct from the
// blob's own dtype numbering — see BlobWriter.h.
enum class MilDataType : int {
    Bool = 1,
    String = 2,
    Float16 = 10,
    Float32 = 11,
    Int32 = 23,
};

// A tensor type.  An empty ``shape`` means a scalar (rank 0).
struct MilTensorType {
    MilDataType dtype = MilDataType::Float32;
    std::vector<std::int64_t> shape;
};

// ``param name -> value names`` for one operation's operands.
//
// A parameter can bind more than one value — ``concat`` takes a list —
// because MIL's ``Argument`` holds a repeated ``Binding``.  Single-value
// parameters are just a one-element list.
using MilInputs = std::vector<std::pair<std::string, std::vector<std::string>>>;

class LUCID_API MilProgram {
public:
    // ``opset`` names both the function's opset and the single block
    // specialisation keyed by it.
    MilProgram(std::string input_name, MilTensorType input_type, std::string opset = "CoreML7");

    // ── constants ────────────────────────────────────────────────────
    //
    // Each becomes a ``const`` operation.  Large tensors live in the blob
    // and are referenced by the offset :class:`BlobWriter` returned;
    // everything else is inline.
    void
    add_blob_const(const std::string& name, const MilTensorType& type, std::uint64_t blob_offset);
    void
    add_int_const(const std::string& name, const std::vector<std::int64_t>& values, bool scalar);
    // Integer constant of arbitrary shape.  The weight blob this writer
    // emits carries float payloads only, and MIL already has an inline
    // integer tensor, so integer buffers go inline rather than teaching
    // the blob a dtype code that would have to be guessed.
    void add_int_const_shaped(const std::string& name, const std::vector<std::int64_t>& values,
                              const std::vector<std::int64_t>& shape);
    void add_float_const(const std::string& name, const std::vector<float>& values, bool scalar);
    void add_string_const(const std::string& name, const std::string& value);
    void add_bool_const(const std::string& name, bool value);

    // Boolean constant of arbitrary shape, carried inline.  A traced mask
    // is a real tensor of booleans, not a count that happens to be 0 or 1,
    // and MIL types the two apart.
    void add_bool_const_shaped(const std::string& name, const std::vector<bool>& values,
                               const std::vector<std::int64_t>& shape);

    // ── operations ───────────────────────────────────────────────────
    void add_op(const std::string& op_type,
                const MilInputs& inputs,
                const std::string& output_name,
                const MilTensorType& output_type);

    // Multi-output form.  ``Operation.outputs`` is repeated in the schema
    // and some operations use it — ``split`` produces one value per
    // section — so a one-output-per-op writer cannot express them.
    void add_op_multi(const std::string& op_type,
                      const MilInputs& inputs,
                      const std::vector<std::pair<std::string, MilTensorType>>& outputs);

    // The value the model returns.  Must name an existing operation output.
    void set_output(const std::string& name, const MilTensorType& type);

    // Serialise to ``Model`` protobuf bytes.
    //
    // Raises
    // ------
    // std::logic_error
    //     No output was set — a program with no result would still
    //     serialise, and Core ML would reject it far from the cause.
    std::string serialize() const;

    std::size_t op_count() const { return ops_.size(); }

private:
    struct Op {
        std::string type;
        MilInputs inputs;
        std::string output_name;
        MilTensorType output_type;
        // Additional outputs beyond the first, for multi-output ops.
        std::vector<std::pair<std::string, MilTensorType>> extra_outputs;
        // ``const`` payload, unused for other op types.
        bool is_const = false;
        bool blob = false;
        std::uint64_t blob_offset = 0;
        std::vector<std::int64_t> ints;
        std::vector<float> floats;
        std::vector<std::string> strings;
        std::vector<bool> bools;
    };

    void push_const(Op op);

    std::string input_name_;
    MilTensorType input_type_;
    std::string opset_;
    std::string output_name_;
    MilTensorType output_type_;
    bool has_output_ = false;
    std::vector<Op> ops_;
};

// Blob file reference emitted into every blob-backed constant.  Core ML
// resolves ``@model_path`` against the package directory.
LUCID_API extern const char* const kWeightFileRef;

}  // namespace lucid::coreml
