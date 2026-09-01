// lucid/_C/coreml/BlobWriter.cpp — see BlobWriter.h for the layout.

#include "BlobWriter.h"

#include <array>
#include <cstring>
#include <stdexcept>

namespace lucid::coreml {

namespace {

constexpr std::size_t kAlignment = 64;
constexpr std::uint32_t kEntrySentinel = 0xDEADBEEFu;
constexpr std::uint32_t kBlobVersion = 2u;

// Every integer in this format is little-endian, which is also the host
// order on every machine Lucid supports.  Written byte by byte anyway so
// the file does not silently depend on that.
template <typename T>
void put_le(std::ofstream& out, T value) {
    std::array<char, sizeof(T)> bytes{};
    for (std::size_t i = 0; i < sizeof(T); ++i)
        bytes[i] = static_cast<char>((static_cast<std::uint64_t>(value) >> (8 * i)) & 0xFF);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void put_zeros(std::ofstream& out, std::size_t n) {
    constexpr std::size_t kChunk = 64;
    const std::array<char, kChunk> zeros{};
    while (n > 0) {
        const std::size_t take = n < kChunk ? n : kChunk;
        out.write(zeros.data(), static_cast<std::streamsize>(take));
        n -= take;
    }
}

}  // namespace

BlobWriter::BlobWriter(const std::string& path) : path_(path) {
    out_.open(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out_)
        throw std::runtime_error("BlobWriter: cannot open " + path + " for writing");
    // Reserve the header; ``finalize`` seeks back to fill in the count.
    put_zeros(out_, kAlignment);
    if (!out_)
        throw std::runtime_error("BlobWriter: failed to write the file header");
}

void BlobWriter::pad_to_alignment() {
    const auto pos = static_cast<std::uint64_t>(out_.tellp());
    const std::uint64_t rem = pos % kAlignment;
    if (rem != 0)
        put_zeros(out_, static_cast<std::size_t>(kAlignment - rem));
}

std::uint64_t BlobWriter::append(const void* data, std::size_t size_bytes, BlobDataType dtype) {
    if (finalized_)
        throw std::logic_error("BlobWriter::append: the blob is already finalized");
    if (data == nullptr && size_bytes != 0)
        throw std::invalid_argument("BlobWriter::append: null data with non-zero size");

    pad_to_alignment();
    const auto entry_offset = static_cast<std::uint64_t>(out_.tellp());
    const std::uint64_t data_offset = entry_offset + kAlignment;

    put_le<std::uint32_t>(out_, kEntrySentinel);
    put_le<std::uint32_t>(out_, static_cast<std::uint32_t>(dtype));
    put_le<std::uint64_t>(out_, static_cast<std::uint64_t>(size_bytes));
    put_le<std::uint64_t>(out_, data_offset);
    put_le<std::uint64_t>(out_, 0u);  // reserved
    // Header occupies a full alignment unit: 4 + 4 + 8 + 8 + 8 = 32 written.
    put_zeros(out_, kAlignment - 32);

    if (size_bytes != 0)
        out_.write(static_cast<const char*>(data), static_cast<std::streamsize>(size_bytes));
    if (!out_)
        throw std::runtime_error("BlobWriter::append: write failed for " + path_);

    ++count_;
    return entry_offset;
}

void BlobWriter::finalize() {
    if (finalized_)
        return;
    pad_to_alignment();
    out_.seekp(0, std::ios::beg);
    put_le<std::uint32_t>(out_, count_);
    put_le<std::uint32_t>(out_, kBlobVersion);
    out_.flush();
    if (!out_)
        throw std::runtime_error("BlobWriter::finalize: failed to backfill the header");
    out_.close();
    finalized_ = true;
}

}  // namespace lucid::coreml
