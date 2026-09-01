// lucid/_C/coreml/MilProgram.cpp — see MilProgram.h.

#include "MilProgram.h"

#include <stdexcept>

#include "MilSchema.h"
#include "ProtoWriter.h"

namespace lucid::coreml {

const char* const kWeightFileRef = "@model_path/weights/weight.bin";

namespace {

// Core ML specification version that introduced the ML Program format
// this writer emits, paired with the opset the reference package used.
constexpr int kSpecificationVersion = 8;
constexpr int kProgramVersion = 1;

ProtoWriter make_value_type(const MilTensorType& type) {
    ProtoWriter tensor;
    tensor.write_enum(pb::TensorType::kDataType, static_cast<int>(type.dtype));
    if (!type.shape.empty()) {
        // Rank 0 is proto3's default and is omitted, matching how Core ML
        // writes scalars.
        tensor.write_int(pb::TensorType::kRank, static_cast<std::int64_t>(type.shape.size()));
        for (std::int64_t dim : type.shape) {
            ProtoWriter constant;
            constant.write_int(pb::ConstantDimension::kSize, dim);
            ProtoWriter dimension;
            dimension.write_message(pb::Dimension::kConstant, constant);
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

}  // namespace

MilProgram::MilProgram(std::string input_name, MilTensorType input_type, std::string opset)
    : input_name_(std::move(input_name)),
      input_type_(std::move(input_type)),
      opset_(std::move(opset)) {}

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

void MilProgram::set_output(const std::string& name, const MilTensorType& type) {
    output_name_ = name;
    output_type_ = type;
    has_output_ = true;
}

std::string MilProgram::serialize() const {
    if (!has_output_)
        throw std::logic_error(
            "MilProgram::serialize: no output was set — Core ML would reject the "
            "package with an error far from this cause");

    // ── operations ───────────────────────────────────────────────────
    ProtoWriter block;
    block.write_string(pb::Block::kOutputs, output_name_);
    for (const Op& op : ops_) {
        ProtoWriter operation;
        operation.write_string(pb::Operation::kType, op.type);
        for (const auto& [param, value_names] : op.inputs)
            operation.write_map_entry(pb::Operation::kInputs, param, make_argument(value_names));
        operation.write_message(pb::Operation::kOutputs,
                                make_named_value_type(op.output_name, op.output_type));

        if (op.is_const) {
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
    function.write_message(pb::Function::kInputs, make_named_value_type(input_name_, input_type_));
    function.write_string(pb::Function::kOpset, opset_);
    function.write_map_entry(pb::Function::kBlockSpecializations, opset_, block);

    ProtoWriter program;
    program.write_int(pb::Program::kVersion, kProgramVersion);
    program.write_map_entry(pb::Program::kFunctions, "main", function);

    // ── model description ────────────────────────────────────────────
    auto feature = [](const std::string& name, const MilTensorType& type) {
        ProtoWriter array;
        array.write_packed_ints(pb::ArrayFeatureType::kShape, type.shape);
        const int array_dtype = type.dtype == MilDataType::Float16
                                    ? pb::ArrayFeatureType_ArrayDataType::kFLOAT16
                                    : pb::ArrayFeatureType_ArrayDataType::kFLOAT32;
        array.write_enum(pb::ArrayFeatureType::kDataType, array_dtype);
        ProtoWriter feature_type;
        feature_type.write_message(pb::FeatureType::kMultiArrayType, array);
        ProtoWriter description;
        description.write_string(pb::FeatureDescription::kName, name);
        description.write_message(pb::FeatureDescription::kType, feature_type);
        return description;
    };

    ProtoWriter description;
    description.write_message(pb::ModelDescription::kInput, feature(input_name_, input_type_));
    description.write_message(pb::ModelDescription::kOutput, feature(output_name_, output_type_));

    ProtoWriter model;
    model.write_int(pb::Model::kSpecificationVersion, kSpecificationVersion);
    model.write_message(pb::Model::kDescription, description);
    model.write_message(pb::Model::kMlProgram, program);
    return model.bytes();
}

}  // namespace lucid::coreml
