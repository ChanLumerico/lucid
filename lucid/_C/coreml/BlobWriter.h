// lucid/_C/coreml/BlobWriter.h
//
// Writer for a Core ML model package's ``weights/weight.bin``.
//
// The layout below was read out of a package rather than assumed, because
// getting it wrong does not fail — the model loads and computes with
// whatever bytes the offsets happen to land on:
//
//     [0, 64)          file header   uint32 count, uint32 version = 2, zero pad
//     [O, O + 64)      entry header  uint32 0xDEADBEEF, uint32 dtype,
//                                    uint64 size_bytes, uint64 data_offset,
//                                    zero pad
//     [data_offset, +size_bytes)     payload, then zero pad
//
// The next entry header begins at ``align64(data_offset + size_bytes)``,
// verified against three consecutive entries of a reference package.
//
// The offset recorded in the protobuf (``Value.blobFileValue.offset``) is
// the **entry header** offset, not the payload's — the first entry sits at
// 64 while its payload starts at 128.  Mistaking one for the other is the
// silent-corruption case this comment exists to prevent.

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>

#include "../api.h"

namespace lucid::coreml {

// Element types the blob format tags payloads with.  These are *not* the
// MIL ``DataType`` numbers — the two enums disagree (MIL FLOAT32 is 11,
// the blob calls it 2), which is why both appear explicitly.
enum class BlobDataType : std::uint32_t {
    Float16 = 1,  // MIL FLOAT16 (10)
    Float32 = 2,  // MIL FLOAT32 (11)
    Int8 = 4,     // MIL INT8 (21) — read off a reference quantized package
};

class LUCID_API BlobWriter {
public:
    // Creates (or truncates) ``path`` and reserves the file header.
    explicit BlobWriter(const std::string& path);

    BlobWriter(const BlobWriter&) = delete;
    BlobWriter& operator=(const BlobWriter&) = delete;

    // Append one tensor payload.
    //
    // Returns
    // -------
    // std::uint64_t
    //     Offset of the entry header, which is what the protobuf's
    //     ``blobFileValue.offset`` must carry.
    std::uint64_t append(const void* data, std::size_t size_bytes, BlobDataType dtype);

    // Backfill the entry count into the file header and close.  Must be
    // called before the package is read; an unfinalised blob reports zero
    // entries and Core ML rejects the model.
    void finalize();

    std::uint32_t count() const { return count_; }

private:
    void pad_to_alignment();

    std::ofstream out_;
    std::string path_;
    std::uint32_t count_ = 0;
    bool finalized_ = false;
};

}  // namespace lucid::coreml
