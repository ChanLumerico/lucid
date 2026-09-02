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
    Int8 = 21,
    Int32 = 23,
};

// A tensor type.  An empty ``shape`` means a scalar (rank 0), and a
// dimension of ``kUnknownDim`` is one the program does not fix — what a
// flexible input needs, and what every shape downstream of one inherits.
constexpr std::int64_t kUnknownDim = -1;

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

// ``name -> type`` for a model's inputs or its outputs.  Both are plural:
// a detector returns boxes as well as scores, and a transformer takes an
// attention mask as well as token ids.  Exporting one of them and calling
// it the model is the failure this writer exists to prevent, one level up
// from the operations.
using MilNamedTypes = std::vector<std::pair<std::string, MilTensorType>>;

// An input Core ML should present as an image rather than an array.
//
// The program still receives a float32 tensor — the difference is in the
// model description, which lets a caller hand over a ``CVPixelBuffer``
// instead of converting pixels by hand.  Scaling and mean subtraction are
// ordinary ``mul`` and ``add`` operations at the head of the program, not
// description metadata, which is where a reference package puts them.
struct MilImageSpec {
    std::int64_t width = 0;
    std::int64_t height = 0;
    int color_space = 0;  // pb::ImageFeatureType_ColorSpace
};

// What a package says about itself.  Empty fields are omitted.
struct MilMetadata {
    std::string short_description;
    std::string author;
    std::string license;
    std::string version;
};

class LUCID_API MilProgram {
public:
    // ``opset`` names both the function's opset and the single block
    // specialisation keyed by it.
    MilProgram(MilNamedTypes inputs, std::string opset = "CoreML7");

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
    // Float constant of arbitrary shape, carried inline.  Image
    // preprocessing needs a per-channel bias shaped ``(1, C, 1, 1)``, which
    // neither the scalar nor the rank-1 form can express.
    void add_float_const_shaped(const std::string& name, const std::vector<float>& values,
                                const std::vector<std::int64_t>& shape);
    void add_string_const(const std::string& name, const std::string& value);
    void add_bool_const(const std::string& name, bool value);

    // Boolean constant of arbitrary shape, carried inline.  A traced mask
    // is a real tensor of booleans, not a count that happens to be 0 or 1,
    // and MIL types the two apart.
    void add_bool_const_shaped(const std::string& name, const std::vector<bool>& values,
                               const std::vector<std::int64_t>& shape);

    // A weight stored as int8 codes plus a per-channel scale, dequantized
    // on the way into whatever consumes it.
    //
    // Core ML spells this ``constexpr_affine_dequantize``, and it is not
    // shaped like other operations: the codes, the scale, the zero point
    // and the axis are *attributes*, not operands, so it cannot go through
    // ``add_op``.  ``scale_bytes`` and ``zero_point_bytes`` are the raw
    // little-endian payloads — the schema carries both as ``bytes``, which
    // is how a float16 scale and an int8 zero point fit the same field.
    void add_quantized_const(const std::string& name,
                             const MilTensorType& output_type,
                             std::uint64_t blob_offset,
                             const std::string& scale_bytes,
                             MilDataType scale_dtype,
                             const std::string& zero_point_bytes,
                             std::int64_t channels,
                             std::int64_t axis);

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

    // The shapes a flexible input accepts.  The first is the default,
    // which the description also carries as the plain ``shape``.
    void set_enumerated_shapes(const std::string& name,
                               const std::vector<std::vector<std::int64_t>>& shapes);

    // Declare a value the model carries between predictions.
    //
    // Core ML keeps it: the caller does not pass it in and does not get it
    // back, and every prediction sees what the last one wrote.  In the
    // program the state is a function input of its own kind, read with
    // ``read_state`` and written with ``write_state``.  Needs iOS 18 /
    // macOS 15, which is why a package only asks for that when it uses
    // one.
    void add_state(const std::string& name, const MilTensorType& type);

    // Read the current value of a declared state into an ordinary value.
    void read_state(const std::string& state_name, const std::string& output_name,
                    const MilTensorType& type);

    // Store a value into a declared state.  Produces nothing.
    void write_state(const std::string& state_name, const std::string& value_name);

    // The bounds a flexible input accepts, one pair per axis.  An axis
    // the program fixes gets the same number twice.  Unlike enumerated
    // shapes this admits everything in between, which is what a variable
    // sequence length or a camera's changing resolution needs.
    void set_shape_range(const std::string& name,
                         const std::vector<std::pair<std::int64_t, std::int64_t>>& bounds);

    // The concrete shape the description should state for a feature whose
    // program type leaves axes open.  A ``-1`` in a description is not
    // "flexible": Core ML reads it as a shape and rejects it.
    void set_default_shape(const std::string& name,
                           const std::vector<std::int64_t>& shape);

    // Present an input as an image in the model description.  Must name
    // one of the inputs the program was constructed with.
    void set_image_input(const std::string& name, const MilImageSpec& spec);

    // What the package says about itself, for the tooling that reads it.
    void set_metadata(const MilMetadata& metadata);

    // Turn the model into a Core ML classifier.
    //
    // Without this a package returns a score array and the app does its
    // own argmax and label lookup; Vision's ``VNCoreMLRequest``, which
    // reads ``predictedFeatureName``, returns nothing at all.  Declaring
    // it appends a ``classify`` operation and makes its two results —
    // the winning label and the label-to-probability map — the model's
    // outputs, in place of the raw scores.
    void set_classifier(const std::string& scores_value,
                        const std::vector<std::string>& labels,
                        const std::string& label_name,
                        const std::string& probabilities_name);

    // A value the model returns.  Must name an existing operation output.
    // Called once per output, in the order the caller wants them.
    void add_output(const std::string& name, const MilTensorType& type);

    // Serialise to ``Model`` protobuf bytes.
    //
    // Raises
    // ------
    // std::logic_error
    //     No output was added — a program with no result would still
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
        // ``constexpr_affine_dequantize`` payload, likewise.
        bool is_quantized = false;
        std::string scale_bytes;
        std::string zero_point_bytes;
        MilDataType scale_dtype = MilDataType::Float16;
        std::int64_t channels = 0;
        std::int64_t axis = 0;
        // ``classify`` payload: the labels travel inline and the two
        // results have different type kinds, so this op is written by hand.
        bool is_classify = false;
        // ``write_state`` has no result at all, which every other
        // operation here does.
        bool no_output = false;
        std::vector<std::string> labels;
        std::string second_output_name;
        bool blob = false;
        std::uint64_t blob_offset = 0;
        std::vector<std::int64_t> ints;
        std::vector<float> floats;
        std::vector<std::string> strings;
        std::vector<bool> bools;
    };

    void push_const(Op op);

    MilNamedTypes inputs_;
    std::vector<std::pair<std::string, MilImageSpec>> images_;
    MilNamedTypes states_;
    std::vector<std::pair<std::string, std::vector<std::vector<std::int64_t>>>> enumerated_;
    std::vector<std::pair<std::string, std::vector<std::int64_t>>> defaults_;
    std::vector<std::pair<std::string, std::vector<std::pair<std::int64_t, std::int64_t>>>>
        ranges_;
    MilMetadata metadata_;
    struct Classifier {
        bool present = false;
        std::string label_name;
        std::string probabilities_name;
    } classifier_;
    std::string opset_;
    MilNamedTypes outputs_;
    std::vector<Op> ops_;
};

// Blob file reference emitted into every blob-backed constant.  Core ML
// resolves ``@model_path`` against the package directory.
LUCID_API extern const char* const kWeightFileRef;

}  // namespace lucid::coreml
