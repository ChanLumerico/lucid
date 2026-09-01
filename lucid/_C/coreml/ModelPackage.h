// lucid/_C/coreml/ModelPackage.h
//
// The ``.mlpackage`` directory bundle.
//
//     X.mlpackage/
//       Manifest.json                             item table, UUID-keyed
//       Data/com.apple.CoreML/model.mlmodel       the protobuf
//       Data/com.apple.CoreML/weights/weight.bin  the blob
//
// Core ML locates the model through ``Manifest.json``: each item gets a
// UUID key, and ``rootModelIdentifier`` names which one is the model.  The
// weights are a *second* item whose path is the ``weights`` directory, not
// the file inside it — the protobuf refers to the file itself as
// ``@model_path/weights/weight.bin``.
//
// Writing order matters: the blob has to exist and be finalized before the
// protobuf that carries offsets into it is written, or a reader can find
// offsets pointing past the end of the file.

#pragma once

#include <string>

#include "../api.h"

namespace lucid::coreml {

// Absolute paths inside a package that is being written.
struct PackagePaths {
    std::string root;         // X.mlpackage
    std::string mlmodel;      // .../Data/com.apple.CoreML/model.mlmodel
    std::string weights_dir;  // .../Data/com.apple.CoreML/weights
    std::string weight_bin;   // .../weights/weight.bin
};

// Create the directory skeleton, replacing any package already at ``root``.
//
// Replacing rather than merging is deliberate: a stale ``weight.bin`` left
// beside a fresh ``model.mlmodel`` is a model that loads and reads the
// wrong tensors.
LUCID_API PackagePaths prepare_package(const std::string& root);

// Write ``model.mlmodel`` and ``Manifest.json``, completing the package.
//
// Parameters
// ----------
// paths : const PackagePaths&
//     From :func:`prepare_package`.
// mlmodel_bytes : const std::string&
//     Serialised ``Model`` protobuf.
//
// Raises
// ------
// std::runtime_error
//     The blob is missing — that means the caller wrote the protobuf's
//     offsets against a file that does not exist.
LUCID_API void finish_package(const PackagePaths& paths, const std::string& mlmodel_bytes);

}  // namespace lucid::coreml
