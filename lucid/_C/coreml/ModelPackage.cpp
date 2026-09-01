// lucid/_C/coreml/ModelPackage.cpp — see ModelPackage.h.

#include "ModelPackage.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <uuid/uuid.h>

namespace lucid::coreml {

namespace {

namespace fs = std::filesystem;

constexpr const char* kDataDir = "Data";
constexpr const char* kAuthor = "com.apple.CoreML";

std::string make_uuid() {
    uuid_t raw;
    uuid_generate_random(raw);
    std::array<char, 37> text{};
    uuid_unparse_upper(raw, text.data());
    return std::string(text.data());
}

void write_file(const fs::path& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out)
        throw std::runtime_error("ModelPackage: cannot write " + path.string());
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out)
        throw std::runtime_error("ModelPackage: short write to " + path.string());
}

// The manifest is small and fixed-shape, so it is rendered directly
// rather than through a JSON library the engine would otherwise not need.
// Every value here is either a literal or a generated UUID — none of it is
// user text, so there is nothing to escape.
std::string render_manifest(const std::string& model_uuid, const std::string& weights_uuid) {
    std::string out;
    out += "{\n";
    out += "    \"fileFormatVersion\": \"1.0.0\",\n";
    out += "    \"itemInfoEntries\": {\n";
    out += "        \"" + model_uuid + "\": {\n";
    out += std::string("            \"author\": \"") + kAuthor + "\",\n";
    out += "            \"description\": \"CoreML Model Specification\",\n";
    out += "            \"name\": \"model.mlmodel\",\n";
    out += std::string("            \"path\": \"") + kAuthor + "/model.mlmodel\"\n";
    out += "        },\n";
    out += "        \"" + weights_uuid + "\": {\n";
    out += std::string("            \"author\": \"") + kAuthor + "\",\n";
    out += "            \"description\": \"CoreML Model Weights\",\n";
    out += "            \"name\": \"weights\",\n";
    out += std::string("            \"path\": \"") + kAuthor + "/weights\"\n";
    out += "        }\n";
    out += "    },\n";
    out += "    \"rootModelIdentifier\": \"" + model_uuid + "\"\n";
    out += "}\n";
    return out;
}

}  // namespace

PackagePaths prepare_package(const std::string& root) {
    const fs::path base(root);
    std::error_code ec;
    if (fs::exists(base, ec))
        fs::remove_all(base, ec);
    if (ec)
        throw std::runtime_error("ModelPackage: cannot replace " + root + ": " + ec.message());

    const fs::path data = base / kDataDir / kAuthor;
    const fs::path weights = data / "weights";
    fs::create_directories(weights, ec);
    if (ec)
        throw std::runtime_error("ModelPackage: cannot create " + weights.string() + ": " +
                                 ec.message());

    PackagePaths paths;
    paths.root = base.string();
    paths.mlmodel = (data / "model.mlmodel").string();
    paths.weights_dir = weights.string();
    paths.weight_bin = (weights / "weight.bin").string();
    return paths;
}

void finish_package(const PackagePaths& paths, const std::string& mlmodel_bytes) {
    std::error_code ec;
    if (!fs::exists(fs::path(paths.weight_bin), ec)) {
        throw std::runtime_error(
            "ModelPackage: the weight blob is missing at " + paths.weight_bin +
            " — the protobuf's blob offsets would point into a file that does not exist");
    }
    write_file(fs::path(paths.mlmodel), mlmodel_bytes);
    write_file(fs::path(paths.root) / "Manifest.json", render_manifest(make_uuid(), make_uuid()));
}

}  // namespace lucid::coreml
