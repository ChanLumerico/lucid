// lucid/_C/bindings/bind_coreml.cpp
//
// Python surface for the Core ML writer (``lucid._C.engine.coreml``).
//
// Deliberately thin: the engine owns the *format* (protobuf bytes, blob
// layout, package skeleton) and nothing else.  Which Lucid op becomes
// which MIL op, and what a model's inputs are, is decided in
// ``lucid/coreml/`` where it can be read and changed without a rebuild.
//
// Tensor types cross the boundary as ``(dtype, shape)`` pairs rather than
// a bound struct — one less class to keep in sync, and it reads the same
// on both sides.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../core/Storage.h"
#include "../core/TensorImpl.h"
#include "../coreml/BlobWriter.h"
#include "../coreml/CoreMLRuntime.h"
#include "../coreml/MilProgram.h"
#include "../coreml/ModelPackage.h"

namespace py = pybind11;

namespace lucid::bindings {

namespace {

using TypeSpec = std::pair<int, std::vector<std::int64_t>>;

lucid::coreml::MilTensorType to_type(const TypeSpec& spec) {
    return {static_cast<lucid::coreml::MilDataType>(spec.first), spec.second};
}

// ``CoreMLModel`` is only complete inside CoreMLRuntime.mm (it holds
// Objective-C objects), so pybind11 cannot bind it directly.  This owns
// the handle and is complete here — the same shape as
// ``PyCompiledExecutable`` in bind_compile.cpp.
class PyCoreMLModel {
public:
    explicit PyCoreMLModel(lucid::coreml::CoreMLModel* handle) : handle_(handle) {}
    ~PyCoreMLModel() { close(); }

    PyCoreMLModel(const PyCoreMLModel&) = delete;
    PyCoreMLModel& operator=(const PyCoreMLModel&) = delete;

    void close() {
        if (handle_ != nullptr) {
            lucid::coreml::destroy_model(handle_);
            handle_ = nullptr;
        }
    }

    lucid::coreml::CoreMLModel* raw() const {
        if (handle_ == nullptr)
            throw std::runtime_error("lucid.coreml: the model handle is closed");
        return handle_;
    }

private:
    lucid::coreml::CoreMLModel* handle_;
};

}  // namespace

void register_coreml(py::module_& m) {
    py::module_ cm = m.def_submodule(
        "coreml", "Core ML model-package writer: MIL protobuf, weight blob, bundle.");

    py::class_<lucid::coreml::MilProgram>(cm, "MilProgram")
        .def(py::init([](const std::vector<std::pair<std::string, TypeSpec>>& inputs,
                         const std::string& opset) {
                 lucid::coreml::MilNamedTypes converted;
                 converted.reserve(inputs.size());
                 for (const auto& [name, spec] : inputs)
                     converted.emplace_back(name, to_type(spec));
                 return std::make_unique<lucid::coreml::MilProgram>(std::move(converted), opset);
             }),
             py::arg("inputs"), py::arg("opset") = "CoreML7",
             "``inputs`` is a list of (feature name, (dtype, shape)) pairs.")
        .def(
            "add_blob_const",
            [](lucid::coreml::MilProgram& self, const std::string& name, const TypeSpec& type,
               std::uint64_t offset) { self.add_blob_const(name, to_type(type), offset); },
            py::arg("name"), py::arg("type"), py::arg("offset"),
            "Constant whose payload lives in weight.bin at the given entry offset.")
        .def(
            "add_int_const",
            [](lucid::coreml::MilProgram& self, const std::string& name,
               const std::vector<std::int64_t>& values,
               bool scalar) { self.add_int_const(name, values, scalar); },
            py::arg("name"), py::arg("values"), py::arg("scalar") = false)
        .def(
            "add_float_const",
            [](lucid::coreml::MilProgram& self, const std::string& name,
               const std::vector<float>& values,
               bool scalar) { self.add_float_const(name, values, scalar); },
            py::arg("name"), py::arg("values"), py::arg("scalar") = false)
        .def(
            "add_int_const_shaped",
            [](lucid::coreml::MilProgram& self, const std::string& name,
               const std::vector<std::int64_t>& values,
               const std::vector<std::int64_t>& shape) {
                self.add_int_const_shaped(name, values, shape);
            },
            py::arg("name"), py::arg("values"), py::arg("shape"),
            "Integer constant of arbitrary shape, carried inline.")
        .def("add_string_const", &lucid::coreml::MilProgram::add_string_const, py::arg("name"),
             py::arg("value"))
        .def("add_bool_const", &lucid::coreml::MilProgram::add_bool_const, py::arg("name"),
             py::arg("value"))
        .def(
            "add_quantized_const",
            [](lucid::coreml::MilProgram& self, const std::string& name,
               const TypeSpec& output_type, std::uint64_t offset, const py::bytes& scale,
               int scale_dtype, const py::bytes& zero_point, std::int64_t channels,
               std::int64_t axis) {
                self.add_quantized_const(name, to_type(output_type), offset, scale,
                                         static_cast<lucid::coreml::MilDataType>(scale_dtype),
                                         zero_point, channels, axis);
            },
            py::arg("name"), py::arg("output_type"), py::arg("offset"), py::arg("scale"),
            py::arg("scale_dtype"), py::arg("zero_point"), py::arg("channels"), py::arg("axis"),
            "An int8 weight plus its per-channel scale, dequantized by Core ML on "
            "the way into whatever consumes it.")
        .def("add_bool_const_shaped", &lucid::coreml::MilProgram::add_bool_const_shaped,
             py::arg("name"), py::arg("values"), py::arg("shape"),
             "Boolean constant of arbitrary shape, carried inline.")
        .def(
            "add_op",
            [](lucid::coreml::MilProgram& self, const std::string& op_type,
               const lucid::coreml::MilInputs& inputs, const std::string& output_name,
               const TypeSpec& output_type) {
                self.add_op(op_type, inputs, output_name, to_type(output_type));
            },
            py::arg("op_type"), py::arg("inputs"), py::arg("output_name"), py::arg("output_type"),
            "Append an operation. ``inputs`` is a list of (parameter, value-name) pairs.")
        .def(
            "add_op_multi",
            [](lucid::coreml::MilProgram& self, const std::string& op_type,
               const lucid::coreml::MilInputs& inputs,
               const std::vector<std::pair<std::string, TypeSpec>>& outputs) {
                std::vector<std::pair<std::string, lucid::coreml::MilTensorType>> converted;
                converted.reserve(outputs.size());
                for (const auto& [name, spec] : outputs)
                    converted.emplace_back(name, to_type(spec));
                self.add_op_multi(op_type, inputs, converted);
            },
            py::arg("op_type"), py::arg("inputs"), py::arg("outputs"),
            "Append an operation with more than one output, e.g. ``split``.")
        .def(
            "add_output",
            [](lucid::coreml::MilProgram& self, const std::string& name, const TypeSpec& type) {
                self.add_output(name, to_type(type));
            },
            py::arg("name"), py::arg("type"),
            "Append one of the model's outputs, in the order the caller wants them.")
        .def(
            "serialize",
            [](const lucid::coreml::MilProgram& self) { return py::bytes(self.serialize()); },
            "Serialised Core ML ``Model`` protobuf.")
        .def_property_readonly("op_count", &lucid::coreml::MilProgram::op_count);

    py::class_<lucid::coreml::BlobWriter>(cm, "BlobWriter")
        .def(py::init([](const std::string& path) {
                 return std::make_unique<lucid::coreml::BlobWriter>(path);
             }),
             py::arg("path"))
        .def(
            "append_tensor",
            [](lucid::coreml::BlobWriter& self, const TensorImplPtr& tensor, int dtype) {
                if (!tensor)
                    throw std::invalid_argument("BlobWriter.append_tensor: null tensor");
                if (tensor->device() != Device::CPU)
                    throw std::invalid_argument(
                        "BlobWriter.append_tensor: weights must be on the CPU to be "
                        "written into the blob");
                const auto& storage = std::get<CpuStorage>(tensor->storage());
                if (!storage.ptr)
                    throw std::runtime_error(
                        "BlobWriter.append_tensor: the tensor has no host storage");
                return self.append(storage.ptr.get(), storage.nbytes,
                                   static_cast<lucid::coreml::BlobDataType>(dtype));
            },
            py::arg("tensor"), py::arg("dtype"),
            "Append a tensor's bytes; returns the entry offset for the protobuf. "
            "Reading the storage directly keeps numpy out of lucid/coreml/ (H4).")
        .def("finalize", &lucid::coreml::BlobWriter::finalize)
        .def_property_readonly("count", &lucid::coreml::BlobWriter::count);

    py::class_<lucid::coreml::PackagePaths>(cm, "PackagePaths")
        .def_readonly("root", &lucid::coreml::PackagePaths::root)
        .def_readonly("mlmodel", &lucid::coreml::PackagePaths::mlmodel)
        .def_readonly("weights_dir", &lucid::coreml::PackagePaths::weights_dir)
        .def_readonly("weight_bin", &lucid::coreml::PackagePaths::weight_bin);

    cm.def("prepare_package", &lucid::coreml::prepare_package, py::arg("root"),
           "Create the .mlpackage directory skeleton, replacing any package already there.");
    cm.def(
        "finish_package",
        [](const lucid::coreml::PackagePaths& paths, const std::string& mlmodel_bytes) {
            lucid::coreml::finish_package(paths, mlmodel_bytes);
        },
        py::arg("paths"), py::arg("mlmodel_bytes"),
        "Write model.mlmodel and Manifest.json, completing the package.");

    py::enum_<lucid::coreml::ComputeUnits>(cm, "ComputeUnits")
        .value("ALL", lucid::coreml::ComputeUnits::All)
        .value("CPU_ONLY", lucid::coreml::ComputeUnits::CpuOnly)
        .value("CPU_AND_GPU", lucid::coreml::ComputeUnits::CpuAndGpu)
        .value("CPU_AND_NE", lucid::coreml::ComputeUnits::CpuAndNeuralEngine);

    py::class_<PyCoreMLModel, std::shared_ptr<PyCoreMLModel>>(cm, "CoreMLModel")
        .def_property_readonly("input_names",
                               [](const PyCoreMLModel& self) {
                                   return lucid::coreml::input_feature_names(self.raw());
                               })
        .def_property_readonly("output_names",
                               [](const PyCoreMLModel& self) {
                                   return lucid::coreml::output_feature_names(self.raw());
                               })
        .def(
            "predict",
            [](const PyCoreMLModel& self,
               const std::vector<std::pair<std::string, TensorImplPtr>>& inputs,
               const std::vector<std::string>& output_names) {
                return lucid::coreml::predict(self.raw(), inputs, output_names);
            },
            py::arg("inputs"), py::arg("output_names"),
            "Run one prediction. ``inputs`` pairs each feature name with a "
            "contiguous CPU float32 or int32 tensor; the results come back in "
            "the order ``output_names`` asks for.")
        .def("close", &PyCoreMLModel::close,
             "Release the compiled model and its cached artifacts.");

    cm.def(
        "compute_plan",
        [](const std::string& path, lucid::coreml::ComputeUnits units) {
            std::vector<std::pair<std::string, std::string>> out;
            for (const auto& placement : lucid::coreml::compute_plan(path, units))
                out.emplace_back(placement.op_type, placement.device);
            return out;
        },
        py::arg("path"), py::arg("units") = lucid::coreml::ComputeUnits::All,
        "Per-operation device assignment as (op, device) pairs. Empty on macOS < 14.4, "
        "which means unknown rather than unaccelerated.");

    cm.def(
        "load_model",
        [](const std::string& path, lucid::coreml::ComputeUnits units) {
            return std::make_shared<PyCoreMLModel>(lucid::coreml::load_model(path, units));
        },
        py::arg("path"), py::arg("units") = lucid::coreml::ComputeUnits::All,
        "Compile and load a .mlpackage. Compilation is the expensive step and is "
        "done once per handle.");

    cm.attr("BLOB_INT8") = static_cast<int>(lucid::coreml::BlobDataType::Int8);
    cm.attr("BLOB_FLOAT16") = static_cast<int>(lucid::coreml::BlobDataType::Float16);
    cm.attr("BLOB_FLOAT32") = static_cast<int>(lucid::coreml::BlobDataType::Float32);
    cm.attr("DTYPE_BOOL") = static_cast<int>(lucid::coreml::MilDataType::Bool);
    cm.attr("DTYPE_STRING") = static_cast<int>(lucid::coreml::MilDataType::String);
    cm.attr("DTYPE_FLOAT16") = static_cast<int>(lucid::coreml::MilDataType::Float16);
    cm.attr("DTYPE_FLOAT32") = static_cast<int>(lucid::coreml::MilDataType::Float32);
    cm.attr("DTYPE_INT8") = static_cast<int>(lucid::coreml::MilDataType::Int8);
    cm.attr("DTYPE_INT32") = static_cast<int>(lucid::coreml::MilDataType::Int32);
}

}  // namespace lucid::bindings
