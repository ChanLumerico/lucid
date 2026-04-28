#include "Error.h"

#include <sstream>

namespace lucid {

namespace {

std::string format_shape(const std::vector<std::int64_t>& s) {
    std::ostringstream os;
    os << '[';
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (i)
            os << ',';
        os << s[i];
    }
    os << ']';
    return os.str();
}

}  // namespace

OutOfMemory::OutOfMemory(std::size_t requested_bytes,
                         std::size_t current_bytes,
                         std::size_t peak_bytes,
                         std::string device)
    : LucidError(""),
      requested_bytes_(requested_bytes),
      current_bytes_(current_bytes),
      peak_bytes_(peak_bytes),
      device_(std::move(device)) {
    std::ostringstream os;
    os << "OutOfMemory: requested " << requested_bytes << " bytes on " << device_ << ", current "
       << current_bytes << " bytes, peak " << peak_bytes << " bytes";
    msg_ = os.str();
}

ShapeMismatch::ShapeMismatch(std::vector<std::int64_t> expected,
                             std::vector<std::int64_t> got,
                             std::string context)
    : LucidError(""), expected_(std::move(expected)), got_(std::move(got)) {
    std::ostringstream os;
    os << "ShapeMismatch (" << context << "): expected " << format_shape(expected_) << ", got "
       << format_shape(got_);
    msg_ = os.str();
}

DtypeMismatch::DtypeMismatch(std::string expected, std::string got, std::string context)
    : LucidError(""), expected_(std::move(expected)), got_(std::move(got)) {
    std::ostringstream os;
    os << "DtypeMismatch (" << context << "): expected " << expected_ << ", got " << got_;
    msg_ = os.str();
}

DeviceMismatch::DeviceMismatch(std::string expected, std::string got, std::string context)
    : LucidError(""), expected_(std::move(expected)), got_(std::move(got)) {
    std::ostringstream os;
    os << "DeviceMismatch (" << context << "): expected " << expected_ << ", got " << got_;
    msg_ = os.str();
}

VersionMismatch::VersionMismatch(std::int64_t expected, std::int64_t got, std::string context)
    : LucidError(""), expected_(expected), got_(got) {
    std::ostringstream os;
    os << "VersionMismatch (" << context << "): saved version " << expected
       << " but tensor is now at version " << got
       << " (in-place mutation between forward and backward?)";
    msg_ = os.str();
}

GpuNotAvailable::GpuNotAvailable(std::string reason) : LucidError("GpuNotAvailable: " + reason) {}

}  // namespace lucid
