// lucid/_C/coreml/MilProgram.cpp — see MilProgram.h.

#include "MilProgram.h"

#include <algorithm>
#include <stdexcept>

#include "MilSchema.h"
#include "ProtoWriter.h"

namespace lucid::coreml {

const char* const kWeightFileRef = "@model_path/weights/weight.bin";

namespace {

// Core ML specification version that introduced the ML Program format
// this writer emits, paired with the opset the reference package used.
// Core ML 7 (iOS 17) is what everything here needs, except state, which
// arrived in Core ML 8 (iOS 18).  A package asks for the later runtime
// only when it actually uses one.
constexpr int kSpecificationVersion = 8;
constexpr int kSpecificationVersionState = 9;
constexpr const char* kOpsetState = "CoreML8";
constexpr int kProgramVersion = 1;

ProtoWriter make_value_type(const MilTensorType& type) {
    ProtoWriter tensor;
    tensor.write_enum(pb::TensorType::kDataType, static_cast<int>(type.dtype));
    if (!type.shape.empty()) {
        // Rank 0 is proto3's default and is omitted, matching how Core ML
        // writes scalars.
        tensor.write_int(pb::TensorType::kRank, static_cast<std::int64_t>(type.shape.size()));
        for (std::int64_t dim : type.shape) {
            ProtoWriter dimension;
            if (dim == kUnknownDim) {
                // An empty ``UnknownDimension``: not variadic, just not
                // fixed by the program.  Core ML infers it per prediction.
                ProtoWriter unknown;
                dimension.write_message(pb::Dimension::kUnknown, unknown);
            } else {
                ProtoWriter constant;
                constant.write_int(pb::ConstantDimension::kSize, dim);
                dimension.write_message(pb::Dimension::kConstant, constant);
            }
            tensor.write_message(pb::TensorType::kDimensions, dimension);
        }
    }
    ProtoWriter value_type;
    value_type.write_message(pb::ValueType::kTensorType, tensor);
    return value_type;
}

ProtoWriter make_named_value_type(const std::string& name, const MilTensorType& type) {
    ProtoWriter nvt;
    nvt.write_string(pb::NamedValueType::kName, name);
    nvt.write_message(pb::NamedValueType::kType, make_value_type(type));
    return nvt;
}

// ``Value`` holding an inline string — the shape every operation's
// ``name`` attribute takes.
ProtoWriter make_string_value(const std::string& text) {
    ProtoWriter repeated;
    repeated.write_string(1, text);  // RepeatedStrings.values
    ProtoWriter tensor_value;
    tensor_value.write_message(pb::TensorValue::kStrings, repeated);
    ProtoWriter immediate;
    immediate.write_message(pb::ImmediateValue::kTensor, tensor_value);

    ProtoWriter value;
    value.write_message(pb::Value::kType, make_value_type({MilDataType::String, {}}));
    value.write_message(pb::Value::kImmediateValue, immediate);
    return value;
}

// ``Value`` holding an inline byte payload.  MIL has no float16 or int8
// immediate list; both travel as raw little-endian bytes under
// ``TensorValue.bytes``, which is what a reference quantized package does.
// ``Value`` holding a float32 tensor as a typed list.
//
// The raw-bytes encoding below exists because MIL has no immediate list
// for float16 or int8 — it does have one for float32, and the reader
// wants it: handed float32 as bytes it counts elements against the
// declared type and rejects the mismatch ("Tensor storage and type have
// different number of elements"), which is what an int8 export with a
// float32 body used to fail on.
ProtoWriter make_float_value(const std::vector<float>& values, const MilTensorType& type) {
    ProtoWriter repeated;
    repeated.write_packed_floats(1, values);  // RepeatedFloats.values
    ProtoWriter tensor_value;
    tensor_value.write_message(pb::TensorValue::kFloats, repeated);
    ProtoWriter immediate;
    immediate.write_message(pb::ImmediateValue::kTensor, tensor_value);

    ProtoWriter value;
    value.write_message(pb::Value::kType, make_value_type(type));
    value.write_message(pb::Value::kImmediateValue, immediate);
    return value;
}

ProtoWriter make_bytes_value(const std::string& payload, const MilTensorType& type) {
    ProtoWriter repeated;
    repeated.write_bytes(1, payload.data(), payload.size());  // RepeatedBytes.values
    ProtoWriter tensor_value;
    tensor_value.write_message(pb::TensorValue::kBytes, repeated);
    ProtoWriter immediate;
    immediate.write_message(pb::ImmediateValue::kTensor, tensor_value);

    ProtoWriter value;
    value.write_message(pb::Value::kType, make_value_type(type));
    value.write_message(pb::Value::kImmediateValue, immediate);
    return value;
}

// ``Value`` holding one int32 scalar.
ProtoWriter make_int_value(std::int64_t number) {
    ProtoWriter repeated;
    repeated.write_packed_ints(1, {number});  // RepeatedInts.values
    ProtoWriter tensor_value;
    tensor_value.write_message(pb::TensorValue::kInts, repeated);
    ProtoWriter immediate;
    immediate.write_message(pb::ImmediateValue::kTensor, tensor_value);

    ProtoWriter value;
    value.write_message(pb::Value::kType, make_value_type({MilDataType::Int32, {}}));
    value.write_message(pb::Value::kImmediateValue, immediate);
    return value;
}

// ``Value`` holding the dense shape a compressed weight expands to.
//
// Carried as little-endian uint32 *bytes*, not as an integer list: MIL's
// immediate lists are int32 and float32, and an unsigned type travels
// the same way float16 does.  Read off a reference package rather than
// guessed — the integer-list form parses as a different element count
// and the model is rejected before it runs.
ProtoWriter make_shape_value(const std::vector<std::int64_t>& dims) {
    std::string payload(dims.size() * sizeof(std::uint32_t), '\0');
    for (std::size_t i = 0; i < dims.size(); ++i) {
        const auto value = static_cast<std::uint32_t>(dims[i]);
        std::memcpy(&payload[i * sizeof(std::uint32_t)], &value, sizeof(value));
    }
    const std::vector<std::int64_t> extent{static_cast<std::int64_t>(dims.size())};
    return make_bytes_value(payload, {MilDataType::UInt32, extent});
}

// ``Value`` for a payload that lives in the weight blob rather than
// inline — the shared shape of every compressed operand.
ProtoWriter make_blob_value(std::uint64_t offset, const MilTensorType& type) {
    ProtoWriter value;
    value.write_message(pb::Value::kType, make_value_type(type));
    ProtoWriter blob;
    blob.write_string(pb::BlobFileValue::kFileName, kWeightFileRef);
    blob.write_int(pb::BlobFileValue::kOffset, static_cast<std::int64_t>(offset));
    value.write_message(pb::Value::kBlobFileValue, blob);
    return value;
}

// ``ValueType`` for a list of strings of a known length — how MIL types
// a classifier's label set.
ProtoWriter make_string_list_type(std::size_t length) {
    ProtoWriter element;
    ProtoWriter element_tensor;
    element_tensor.write_enum(pb::TensorType::kDataType,
                              static_cast<int>(MilDataType::String));
    element.write_message(pb::ValueType::kTensorType, element_tensor);

    ProtoWriter constant;
    constant.write_int(pb::ConstantDimension::kSize, static_cast<std::int64_t>(length));
    ProtoWriter dimension;
    dimension.write_message(pb::Dimension::kConstant, constant);

    ProtoWriter list;
    list.write_message(pb::ListType::kType, element);
    list.write_message(pb::ListType::kLength, dimension);

    ProtoWriter value_type;
    value_type.write_message(pb::ValueType::kListType, list);
    return value_type;
}

// ``ValueType`` for ``dict<string, double>`` — a classifier's second
// result, which Core ML hands back as label to probability.
ProtoWriter make_string_double_dict_type() {
    ProtoWriter key_tensor;
    key_tensor.write_enum(pb::TensorType::kDataType, static_cast<int>(MilDataType::String));
    ProtoWriter key;
    key.write_message(pb::ValueType::kTensorType, key_tensor);

    ProtoWriter value_tensor;
    value_tensor.write_enum(pb::TensorType::kDataType, pb::DataType::kFLOAT64);
    ProtoWriter value;
    value.write_message(pb::ValueType::kTensorType, value_tensor);

    ProtoWriter dictionary;
    dictionary.write_message(pb::DictionaryType::kKeyType, key);
    dictionary.write_message(pb::DictionaryType::kValueType, value);

    ProtoWriter value_type;
    value_type.write_message(pb::ValueType::kDictionaryType, dictionary);
    return value_type;
}

// An operand binding: references to other values by name.  More than one
// when the parameter is variadic (``concat``'s ``values``).
ProtoWriter make_argument(const std::vector<std::string>& value_names) {
    ProtoWriter argument;
    for (const std::string& name : value_names) {
        ProtoWriter binding;
        binding.write_string(pb::Binding::kName, name);
        argument.write_message(pb::Argument::kArguments, binding);
    }
    return argument;
}

// An argument that carries its value rather than naming one declared
// elsewhere.  The grouped palettization operation takes its payloads
// this way.
ProtoWriter make_value_argument(const ProtoWriter& value) {
    ProtoWriter binding;
    binding.write_message(pb::Binding::kValue, value);
    ProtoWriter argument;
    argument.write_message(pb::Argument::kArguments, binding);
    return argument;
}

}  // namespace

MilProgram::MilProgram(MilNamedTypes inputs, std::string opset)
    : inputs_(std::move(inputs)), opset_(std::move(opset)) {}

void MilProgram::push_const(Op op) {
    op.is_const = true;
    op.type = "const";
    ops_.push_back(std::move(op));
}

void MilProgram::add_blob_const(const std::string& name,
                                const MilTensorType& type,
                                std::uint64_t blob_offset) {
    Op op;
    op.output_name = name;
    op.output_type = type;
    op.blob = true;
    op.blob_offset = blob_offset;
    push_const(std::move(op));
}

void MilProgram::add_int_const(const std::string& name,
                               const std::vector<std::int64_t>& values,
                               bool scalar) {
    Op op;
    op.output_name = name;
    op.output_type = {MilDataType::Int32,
                      scalar ? std::vector<std::int64_t>{}
                             : std::vector<std::int64_t>{static_cast<std::int64_t>(values.size())}};
    op.ints = values;
    push_const(std::move(op));
}

void MilProgram::add_int_const_shaped(const std::string& name,
                                      const std::vector<std::int64_t>& values,
                                      const std::vector<std::int64_t>& shape) {
    Op op;
    op.output_name = name;
    op.output_type = {MilDataType::Int32, shape};
    op.ints = values;
    push_const(std::move(op));
}

void MilProgram::add_float_const(const std::string& name,
                                 const std::vector<float>& values,
                                 bool scalar) {
    Op op;
    op.output_name = name;
    op.output_type = {MilDataType::Float32,
                      scalar ? std::vector<std::int64_t>{}
                             : std::vector<std::int64_t>{static_cast<std::int64_t>(values.size())}};
    op.floats = values;
    push_const(std::move(op));
}

void MilProgram::add_float_const_shaped(const std::string& name,
                                        const std::vector<float>& values,
                                        const std::vector<std::int64_t>& shape) {
    Op op;
    op.output_name = name;
    op.output_type = {MilDataType::Float32, shape};
    op.floats = values;
    push_const(std::move(op));
}

void MilProgram::add_string_const(const std::string& name, const std::string& value) {
    Op op;
    op.output_name = name;
    op.output_type = {MilDataType::String, {}};
    op.strings.push_back(value);
    push_const(std::move(op));
}

void MilProgram::add_bool_const(const std::string& name, bool value) {
    Op op;
    op.output_name = name;
    op.output_type = {MilDataType::Bool, {}};
    op.bools.push_back(value);
    push_const(std::move(op));
}

void MilProgram::add_bool_const_shaped(const std::string& name,
                                       const std::vector<bool>& values,
                                       const std::vector<std::int64_t>& shape) {
    Op op;
    op.output_name = name;
    op.output_type = {MilDataType::Bool, shape};
    op.bools = values;
    push_const(std::move(op));
}

void MilProgram::add_grouped_lut_const(const std::string& name,
                                       const MilTensorType& output_type,
                                       std::uint64_t indices_offset,
                                       MilDataType indices_dtype,
                                       std::uint64_t lut_offset,
                                       MilDataType lut_dtype,
                                       const std::vector<std::int64_t>& lut_shape) {
    Op op;
    op.type = "constexpr_lut_to_dense";
    op.output_name = name;
    op.output_type = output_type;
    op.is_lut = true;
    op.blob = true;
    op.blob_offset = indices_offset;
    op.key_dtype = indices_dtype;
    // ``mask_offset`` doubles as the table's blob offset: the operation
    // carries exactly one auxiliary payload beside the packed keys.
    op.mask_offset = lut_offset;
    op.palette_dtype = lut_dtype;
    op.lut_shape = lut_shape;
    // The grouped spelling of this operation is iOS18 and later.
    if (opset_ == "CoreML7") opset_ = kOpsetState;
    needs_extended_opset_ = true;
    ops_.push_back(std::move(op));
}

void MilProgram::add_sparse_const(const std::string& name,
                                  const MilTensorType& output_type,
                                  std::uint64_t nonzero_offset,
                                  std::int64_t nonzero_count,
                                  std::uint64_t mask_offset,
                                  std::int64_t mask_bytes) {
    Op op;
    op.type = "constexpr_sparse_to_dense";
    op.output_name = name;
    op.output_type = output_type;
    op.is_sparse = true;
    op.blob = true;
    op.blob_offset = nonzero_offset;
    op.nonzero_count = nonzero_count;
    op.mask_offset = mask_offset;
    op.mask_bytes = mask_bytes;
    ops_.push_back(std::move(op));
}

void MilProgram::add_quantized_const(const std::string& name,
                                     const MilTensorType& output_type,
                                     std::uint64_t blob_offset,
                                     const std::string& scale_bytes,
                                     MilDataType scale_dtype,
                                     const std::string& zero_point_bytes,
                                     std::int64_t channels,
                                     std::int64_t axis) {
    Op op;
    op.type = "constexpr_affine_dequantize";
    op.output_name = name;
    op.output_type = output_type;
    op.is_quantized = true;
    op.blob = true;
    op.blob_offset = blob_offset;
    op.scale_bytes = scale_bytes;
    op.zero_point_bytes = zero_point_bytes;
    op.scale_dtype = scale_dtype;
    op.channels = channels;
    op.axis = axis;
    // Not through ``push_const``: that stamps every entry as a ``const``,
    // and this one is a ``constexpr_affine_dequantize`` whose payload
    // lives in attributes of its own.
    ops_.push_back(std::move(op));
}

void MilProgram::add_op(const std::string& op_type,
                        const MilInputs& inputs,
                        const std::string& output_name,
                        const MilTensorType& output_type) {
    Op op;
    op.type = op_type;
    op.inputs = inputs;
    op.output_name = output_name;
    op.output_type = output_type;
    ops_.push_back(std::move(op));
}

void MilProgram::add_op_multi(const std::string& op_type,
                              const MilInputs& inputs,
                              const std::vector<std::pair<std::string, MilTensorType>>& outputs) {
    if (outputs.empty())
        throw std::invalid_argument("MilProgram::add_op_multi: an operation needs an output");
    Op op;
    op.type = op_type;
    op.inputs = inputs;
    op.output_name = outputs.front().first;
    op.output_type = outputs.front().second;
    op.extra_outputs.assign(outputs.begin() + 1, outputs.end());
    ops_.push_back(std::move(op));
}

void MilProgram::set_enumerated_shapes(
    const std::string& name, const std::vector<std::vector<std::int64_t>>& shapes) {
    for (auto& [existing, held] : enumerated_) {
        if (existing == name) {
            held = shapes;
            return;
        }
    }
    enumerated_.emplace_back(name, shapes);
}

void MilProgram::add_state(const std::string& name, const MilTensorType& type) {
    states_.emplace_back(name, type);
    if (opset_ == "CoreML7")
        opset_ = kOpsetState;
}

void MilProgram::read_state(const std::string& state_name,
                            const std::string& output_name,
                            const MilTensorType& type) {
    add_op("read_state", {{"input", {state_name}}}, output_name, type);
}

void MilProgram::write_state(const std::string& state_name,
                             const std::string& value_name) {
    Op op;
    op.type = "write_state";
    op.inputs = {{"input", {state_name}}, {"data", {value_name}}};
    op.output_name = "_write_" + state_name;
    op.no_output = true;
    ops_.push_back(std::move(op));
}

void MilProgram::set_shape_range(
    const std::string& name,
    const std::vector<std::pair<std::int64_t, std::int64_t>>& bounds) {
    for (auto& [existing, held] : ranges_) {
        if (existing == name) {
            held = bounds;
            return;
        }
    }
    ranges_.emplace_back(name, bounds);
}

void MilProgram::set_default_shape(const std::string& name,
                                   const std::vector<std::int64_t>& shape) {
    for (auto& [existing, held] : defaults_) {
        if (existing == name) {
            held = shape;
            return;
        }
    }
    defaults_.emplace_back(name, shape);
}

void MilProgram::set_image_input(const std::string& name, const MilImageSpec& spec) {
    for (auto& [existing, held] : images_) {
        if (existing == name) {
            held = spec;
            return;
        }
    }
    images_.emplace_back(name, spec);
}

void MilProgram::set_metadata(const MilMetadata& metadata) { metadata_ = metadata; }

void MilProgram::set_classifier(const std::string& scores_value,
                                const std::vector<std::string>& labels,
                                const std::string& label_name,
                                const std::string& probabilities_name) {
    Op op;
    op.type = "classify";
    op.is_classify = true;
    op.inputs = {{"probabilities", {scores_value}}};
    op.labels = labels;
    op.output_name = label_name;
    op.second_output_name = probabilities_name;
    ops_.push_back(std::move(op));

    classifier_.present = true;
    classifier_.label_name = label_name;
    classifier_.probabilities_name = probabilities_name;
}

void MilProgram::add_output(const std::string& name, const MilTensorType& type) {
    outputs_.emplace_back(name, type);
}

void MilProgram::set_opset(const std::string& opset) { opset_ = opset; }

ProtoWriter MilProgram::build_function() const {
    if (outputs_.empty() && !classifier_.present)
        throw std::logic_error(
            "MilProgram::serialize: no output was added — Core ML would reject the "
            "package with an error far from this cause");

    // ── operations ───────────────────────────────────────────────────
    ProtoWriter block;
    if (classifier_.present) {
        block.write_string(pb::Block::kOutputs, classifier_.label_name);
        block.write_string(pb::Block::kOutputs, classifier_.probabilities_name);
    } else {
        for (const auto& [name, type] : outputs_)
            block.write_string(pb::Block::kOutputs, name);
    }
    for (const Op& op : ops_) {
        ProtoWriter operation;
        operation.write_string(pb::Operation::kType, op.type);
        if (op.is_classify) {
            // ``classes`` is bound to an inline list rather than to a
            // const: MIL types it ``list<string>``, which a rank-1 string
            // tensor is not.
            ProtoWriter list;
            for (const std::string& label : op.labels) {
                ProtoWriter repeated;
                repeated.write_string(1, label);  // RepeatedStrings.values
                ProtoWriter tensor_value;
                tensor_value.write_message(pb::TensorValue::kStrings, repeated);
                ProtoWriter immediate;
                immediate.write_message(pb::ImmediateValue::kTensor, tensor_value);
                ProtoWriter entry;
                entry.write_message(pb::Value::kType,
                                    make_value_type({MilDataType::String, {}}));
                entry.write_message(pb::Value::kImmediateValue, immediate);
                list.write_message(pb::ListValue::kValues, entry);
            }
            ProtoWriter list_immediate;
            list_immediate.write_message(pb::ImmediateValue::kList, list);
            ProtoWriter classes;
            classes.write_message(pb::Value::kType, make_string_list_type(op.labels.size()));
            classes.write_message(pb::Value::kImmediateValue, list_immediate);

            ProtoWriter binding;
            binding.write_message(pb::Binding::kValue, classes);
            ProtoWriter argument;
            argument.write_message(pb::Argument::kArguments, binding);
            operation.write_map_entry(pb::Operation::kInputs, "classes", argument);

            for (const auto& [param, value_names] : op.inputs)
                operation.write_map_entry(pb::Operation::kInputs, param,
                                          make_argument(value_names));

            ProtoWriter label;
            label.write_string(pb::NamedValueType::kName, op.output_name);
            ProtoWriter label_type;
            ProtoWriter label_tensor;
            label_tensor.write_enum(pb::TensorType::kDataType,
                                    static_cast<int>(MilDataType::String));
            label_type.write_message(pb::ValueType::kTensorType, label_tensor);
            label.write_message(pb::NamedValueType::kType, label_type);
            operation.write_message(pb::Operation::kOutputs, label);

            ProtoWriter probabilities;
            probabilities.write_string(pb::NamedValueType::kName, op.second_output_name);
            probabilities.write_message(pb::NamedValueType::kType,
                                        make_string_double_dict_type());
            operation.write_message(pb::Operation::kOutputs, probabilities);

            operation.write_map_entry(pb::Operation::kAttributes, "name",
                                      make_string_value(op.output_name));
            block.write_message(pb::Block::kOperations, operation);
            continue;
        }
        for (const auto& [param, value_names] : op.inputs)
            operation.write_map_entry(pb::Operation::kInputs, param, make_argument(value_names));
        if (!op.no_output)
            operation.write_message(pb::Operation::kOutputs,
                                    make_named_value_type(op.output_name, op.output_type));
        for (const auto& [extra_name, extra_type] : op.extra_outputs)
            operation.write_message(pb::Operation::kOutputs,
                                    make_named_value_type(extra_name, extra_type));

        if (op.is_quantized) {
            // (see below for the palettized and sparse forms, which
            // carry their dense shape as an attribute of its own)
            // The codes carry the weight's own shape; the scale and zero
            // point carry one entry per channel along ``axis``.
            ProtoWriter codes;
            codes.write_message(pb::Value::kType,
                                make_value_type({MilDataType::Int8, op.output_type.shape}));
            ProtoWriter blob;
            blob.write_string(pb::BlobFileValue::kFileName, kWeightFileRef);
            blob.write_int(pb::BlobFileValue::kOffset, static_cast<std::int64_t>(op.blob_offset));
            codes.write_message(pb::Value::kBlobFileValue, blob);
            operation.write_map_entry(pb::Operation::kAttributes, "quantized_data", codes);

            const std::vector<std::int64_t> per_channel{op.channels};
            if (op.scale_dtype == MilDataType::Float32) {
                // The bytes arrive little-endian float32; hand them over
                // as the typed list the reader expects for this dtype.
                std::vector<float> scales(op.scale_bytes.size() / sizeof(float));
                std::memcpy(scales.data(), op.scale_bytes.data(), scales.size() * sizeof(float));
                operation.write_map_entry(pb::Operation::kAttributes, "scale",
                                          make_float_value(scales, {op.scale_dtype, per_channel}));
            } else {
                operation.write_map_entry(
                    pb::Operation::kAttributes, "scale",
                    make_bytes_value(op.scale_bytes, {op.scale_dtype, per_channel}));
            }
            operation.write_map_entry(
                pb::Operation::kAttributes, "zero_point",
                make_bytes_value(op.zero_point_bytes, {MilDataType::Int8, per_channel}));
            operation.write_map_entry(pb::Operation::kAttributes, "axis",
                                      make_int_value(op.axis));
        } else if (op.is_lut) {
            // Unlike every other compressed form here, this one names
            // its payloads through *inputs* carrying inline values, not
            // through attributes, and the keys are typed by the weight's
            // own shape at a sub-byte width rather than by a byte count.
            operation.write_map_entry(
                pb::Operation::kInputs, "indices",
                make_value_argument(
                    make_blob_value(op.blob_offset,
                                    {op.key_dtype, op.output_type.shape})));
            operation.write_map_entry(
                pb::Operation::kInputs, "lut",
                make_value_argument(
                    make_blob_value(op.mask_offset, {op.palette_dtype, op.lut_shape})));
        } else if (op.is_sparse) {
            operation.write_map_entry(
                pb::Operation::kAttributes, "nonzero_data",
                make_blob_value(op.blob_offset,
                                {op.output_type.dtype, {op.nonzero_count}}));
            operation.write_map_entry(
                pb::Operation::kAttributes, "mask",
                make_blob_value(op.mask_offset, {MilDataType::UInt8, {op.mask_bytes}}));
            operation.write_map_entry(pb::Operation::kAttributes, "shape",
                                      make_shape_value(op.output_type.shape));
        } else if (op.is_const) {
            ProtoWriter value;
            value.write_message(pb::Value::kType, make_value_type(op.output_type));
            if (op.blob) {
                ProtoWriter blob;
                blob.write_string(pb::BlobFileValue::kFileName, kWeightFileRef);
                blob.write_int(pb::BlobFileValue::kOffset,
                               static_cast<std::int64_t>(op.blob_offset));
                value.write_message(pb::Value::kBlobFileValue, blob);
            } else {
                ProtoWriter tensor_value;
                if (!op.ints.empty()) {
                    ProtoWriter repeated;
                    repeated.write_packed_ints(1, op.ints);
                    tensor_value.write_message(pb::TensorValue::kInts, repeated);
                } else if (!op.floats.empty()) {
                    ProtoWriter repeated;
                    repeated.write_packed_floats(1, op.floats);
                    tensor_value.write_message(pb::TensorValue::kFloats, repeated);
                } else if (!op.strings.empty()) {
                    ProtoWriter repeated;
                    for (const std::string& s : op.strings)
                        repeated.write_string(1, s);
                    tensor_value.write_message(pb::TensorValue::kStrings, repeated);
                } else if (!op.bools.empty()) {
                    ProtoWriter repeated;
                    std::vector<std::int64_t> as_ints;
                    as_ints.reserve(op.bools.size());
                    for (bool b : op.bools)
                        as_ints.push_back(b ? 1 : 0);
                    repeated.write_packed_ints(1, as_ints);
                    tensor_value.write_message(pb::TensorValue::kBools, repeated);
                }
                ProtoWriter immediate;
                immediate.write_message(pb::ImmediateValue::kTensor, tensor_value);
                value.write_message(pb::Value::kImmediateValue, immediate);
            }
            operation.write_map_entry(pb::Operation::kAttributes, "val", value);
        }
        // Core ML expects every operation to name itself, ``const``
        // included.
        operation.write_map_entry(pb::Operation::kAttributes, "name",
                                  make_string_value(op.output_name));
        block.write_message(pb::Block::kOperations, operation);
    }

    // ── function / program ───────────────────────────────────────────
    ProtoWriter function;
    for (const auto& [name, type] : inputs_)
        function.write_message(pb::Function::kInputs, make_named_value_type(name, type));
    for (const auto& [name, type] : states_) {
        ProtoWriter state;
        state.write_message(pb::StateType::kWrappedType, make_value_type(type));
        ProtoWriter value_type;
        value_type.write_message(pb::ValueType::kStateType, state);
        ProtoWriter named;
        named.write_string(pb::NamedValueType::kName, name);
        named.write_message(pb::NamedValueType::kType, value_type);
        function.write_message(pb::Function::kInputs, named);
    }
    function.write_string(pb::Function::kOpset, opset_);
    function.write_map_entry(pb::Function::kBlockSpecializations, opset_, block);

    return function;
}

ProtoWriter MilProgram::build_description(const MilDescriptionFields& fields) const {
    // ── model description ────────────────────────────────────────────
    auto feature = [this](const std::string& name, const MilTensorType& type) {
        for (const auto& [image_name, spec] : images_) {
            if (image_name != name)
                continue;
            ProtoWriter image;
            image.write_int(pb::ImageFeatureType::kWidth, spec.width);
            image.write_int(pb::ImageFeatureType::kHeight, spec.height);
            image.write_enum(pb::ImageFeatureType::kColorSpace, spec.color_space);
            ProtoWriter image_feature_type;
            image_feature_type.write_message(pb::FeatureType::kImageType, image);
            ProtoWriter image_description;
            image_description.write_string(pb::FeatureDescription::kName, name);
            image_description.write_message(pb::FeatureDescription::kType,
                                            image_feature_type);
            return image_description;
        }
        ProtoWriter array;
        // What the description states about the shape depends on whether
        // the program fixed it.  An input with alternatives states the
        // default one; an output whose shape follows the input states no
        // shape at all, which is what a reference flexible package does.
        // A ``-1`` is never right here: Core ML reads it as a size.
        const std::vector<std::int64_t>* stated_shape = &type.shape;
        for (const auto& [stated_name, stated] : defaults_) {
            if (stated_name == name)
                stated_shape = &stated;
        }
        const bool open =
            std::find(stated_shape->begin(), stated_shape->end(), kUnknownDim) !=
            stated_shape->end();
        if (!open)
            array.write_packed_ints(pb::ArrayFeatureType::kShape, *stated_shape);
        int array_dtype = pb::ArrayFeatureType_ArrayDataType::kFLOAT32;
        if (type.dtype == MilDataType::Float16)
            array_dtype = pb::ArrayFeatureType_ArrayDataType::kFLOAT16;
        else if (type.dtype == MilDataType::Int32)
            // Token ids and masks arrive as integers; Core ML's multi-array
            // has an int32 type but no int64, which is why the Python layer
            // narrows on the way in.
            array_dtype = pb::ArrayFeatureType_ArrayDataType::kINT32;
        array.write_enum(pb::ArrayFeatureType::kDataType, array_dtype);
        for (const auto& [ranged_name, bounds] : ranges_) {
            if (ranged_name != name)
                continue;
            ProtoWriter range;
            for (const auto& [low, high] : bounds) {
                ProtoWriter size;
                size.write_int(pb::SizeRange::kLowerBound, low);
                size.write_int(pb::SizeRange::kUpperBound, high);
                range.write_message(pb::ShapeRange::kSizeRanges, size);
            }
            array.write_message(pb::ArrayFeatureType::kShapeRange, range);
        }
        for (const auto& [flexible_name, shapes] : enumerated_) {
            if (flexible_name != name)
                continue;
            ProtoWriter enumerated;
            for (const std::vector<std::int64_t>& alternative : shapes) {
                ProtoWriter one;
                one.write_packed_ints(pb::Shape::kShape, alternative);
                enumerated.write_message(pb::EnumeratedShapes::kShapes, one);
            }
            array.write_message(pb::ArrayFeatureType::kEnumeratedShapes, enumerated);
        }
        ProtoWriter feature_type;
        feature_type.write_message(pb::FeatureType::kMultiArrayType, array);
        ProtoWriter description;
        description.write_string(pb::FeatureDescription::kName, name);
        description.write_message(pb::FeatureDescription::kType, feature_type);
        return description;
    };

    ProtoWriter description;
    for (const auto& [name, type] : inputs_)
        description.write_message(fields.input, feature(name, type));
    if (classifier_.present) {
        ProtoWriter string_type;
        ProtoWriter label_feature_type;
        label_feature_type.write_message(pb::FeatureType::kStringType, string_type);
        ProtoWriter label_description;
        label_description.write_string(pb::FeatureDescription::kName,
                                       classifier_.label_name);
        label_description.write_message(pb::FeatureDescription::kType, label_feature_type);
        description.write_message(fields.output, label_description);

        ProtoWriter string_key;
        ProtoWriter dictionary;
        dictionary.write_message(pb::DictionaryFeatureType::kStringKeyType, string_key);
        ProtoWriter probabilities_feature_type;
        probabilities_feature_type.write_message(pb::FeatureType::kDictionaryType,
                                                 dictionary);
        ProtoWriter probabilities_description;
        probabilities_description.write_string(pb::FeatureDescription::kName,
                                               classifier_.probabilities_name);
        probabilities_description.write_message(pb::FeatureDescription::kType,
                                                probabilities_feature_type);
        description.write_message(fields.output, probabilities_description);

        description.write_string(fields.predicted_feature,
                                 classifier_.label_name);
        description.write_string(fields.predicted_probabilities,
                                 classifier_.probabilities_name);
    } else {
        for (const auto& [name, type] : outputs_)
            description.write_message(fields.output, feature(name, type));
    }


    for (const auto& [name, type] : states_) {
        ProtoWriter array;
        array.write_packed_ints(pb::ArrayFeatureType::kShape, type.shape);
        int array_dtype = pb::ArrayFeatureType_ArrayDataType::kFLOAT32;
        if (type.dtype == MilDataType::Float16)
            array_dtype = pb::ArrayFeatureType_ArrayDataType::kFLOAT16;
        array.write_enum(pb::ArrayFeatureType::kDataType, array_dtype);
        ProtoWriter state_feature;
        state_feature.write_message(pb::StateFeatureType::kArrayType, array);
        ProtoWriter feature_type;
        feature_type.write_message(pb::FeatureType::kStateType, state_feature);
        ProtoWriter state_description;
        state_description.write_string(pb::FeatureDescription::kName, name);
        state_description.write_message(pb::FeatureDescription::kType, feature_type);
        description.write_message(fields.state, state_description);
    }

    return description;
}

void MilProgram::write_metadata(ProtoWriter& description) const {
    if (!metadata_.short_description.empty() || !metadata_.author.empty() ||
        !metadata_.license.empty() || !metadata_.version.empty()) {
        ProtoWriter metadata;
        if (!metadata_.short_description.empty())
            metadata.write_string(pb::Metadata::kShortDescription,
                                  metadata_.short_description);
        if (!metadata_.version.empty())
            metadata.write_string(pb::Metadata::kVersionString, metadata_.version);
        if (!metadata_.author.empty())
            metadata.write_string(pb::Metadata::kAuthor, metadata_.author);
        if (!metadata_.license.empty())
            metadata.write_string(pb::Metadata::kLicense, metadata_.license);
        description.write_message(pb::ModelDescription::kMetadata, metadata);
    }
}

std::string MilProgram::serialize() const {
    ProtoWriter function = build_function();

    ProtoWriter program;
    program.write_int(pb::Program::kVersion, kProgramVersion);
    program.write_map_entry(pb::Program::kFunctions, "main", function);

    ProtoWriter description = build_description(
        {pb::ModelDescription::kInput, pb::ModelDescription::kOutput,
         pb::ModelDescription::kState, pb::ModelDescription::kPredictedFeatureName,
         pb::ModelDescription::kPredictedProbabilitiesName});

    write_metadata(description);

    ProtoWriter model;
    model.write_int(pb::Model::kSpecificationVersion,
                    (states_.empty() && !needs_extended_opset_)
                        ? kSpecificationVersion
                        : kSpecificationVersionState);
    model.write_message(pb::Model::kDescription, description);
    model.write_message(pb::Model::kMlProgram, program);
    return model.bytes();
}

std::string
serialize_functions(const std::vector<std::pair<std::string, MilProgram*>>& functions,
                    const std::string& default_name) {
    if (functions.empty())
        throw std::logic_error("serialize_functions: a package needs at least one function");

    bool named = false;
    for (const auto& [name, program] : functions) {
        if (program == nullptr)
            throw std::logic_error("serialize_functions: null program for " + name);
        // Several entry points is a Core ML 8 feature whatever the
        // operations inside need.
        program->set_opset(kOpsetState);
        named = named || name == default_name;
    }
    if (!named)
        throw std::logic_error("serialize_functions: the default function " +
                               default_name + " is not one of the functions");

    ProtoWriter program_message;
    program_message.write_int(pb::Program::kVersion, kProgramVersion);
    ProtoWriter description;
    for (const auto& [name, program] : functions) {
        program_message.write_map_entry(pb::Program::kFunctions, name,
                                        program->build_function());
        ProtoWriter one = program->build_description(
            {pb::FunctionDescription::kInput, pb::FunctionDescription::kOutput,
             pb::FunctionDescription::kState,
             pb::FunctionDescription::kPredictedFeatureName,
             pb::FunctionDescription::kPredictedProbabilitiesName});
        ProtoWriter entry;
        entry.write_string(pb::FunctionDescription::kName, name);
        entry.append_raw(one);
        description.write_message(pb::ModelDescription::kFunctions, entry);
    }
    description.write_string(pb::ModelDescription::kDefaultFunctionName, default_name);
    // One package, one set of metadata: the first function's.
    functions.front().second->write_metadata(description);

    ProtoWriter model;
    model.write_int(pb::Model::kSpecificationVersion, kSpecificationVersionState);
    model.write_message(pb::Model::kDescription, description);
    model.write_message(pb::Model::kMlProgram, program_message);
    return model.bytes();
}

}  // namespace lucid::coreml
