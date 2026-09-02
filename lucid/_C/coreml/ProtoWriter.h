// lucid/_C/coreml/ProtoWriter.h
//
// Minimal protobuf *writer*.  Lucid emits ``.mlpackage`` files without a
// protobuf library so nothing under ``lucid/`` gains an external
// dependency (H4); only the encoding side is needed, and the encoding
// side is small — tags, varints, and length-delimited bytes.
//
// Deliberately not a parser.  Reading protobuf means unknown fields,
// groups, and validation; writing means appending well-defined bytes.
// Keeping the scope to writing is what makes hand-rolling defensible.
//
// Field *numbers* are never written by hand — they come from the
// generated :file:`MilSchema.h`.  See tools/gen_mil_schema.py for why.
//
// Nested messages are built into their own buffers and then embedded,
// because a length-delimited field needs its length before its bytes.
// The nesting depth of a Core ML program is small and the buffers are
// short-lived, so the extra copies are not worth designing around.

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace lucid::coreml {

// Protobuf wire types (spec §"Message Structure").
enum class WireType : std::uint32_t {
    Varint = 0,
    Fixed64 = 1,
    LengthDelimited = 2,
    Fixed32 = 5,
};

class ProtoWriter {
public:
    ProtoWriter() = default;

    // ── scalars ──────────────────────────────────────────────────────
    void write_varint(int field, std::uint64_t value) {
        write_tag(field, WireType::Varint);
        append_varint(value);
    }

    // Signed ints go out as two's-complement varints (protobuf's ``int32``
    // / ``int64``, not ``sint*``): a negative value therefore always
    // occupies ten bytes, which is the encoding readers expect.
    void write_int(int field, std::int64_t value) {
        write_varint(field, static_cast<std::uint64_t>(value));
    }

    void write_bool(int field, bool value) { write_varint(field, value ? 1u : 0u); }

    void write_enum(int field, int value) {
        write_varint(field, static_cast<std::uint64_t>(static_cast<std::int64_t>(value)));
    }

    void write_float(int field, float value) {
        write_tag(field, WireType::Fixed32);
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        for (int i = 0; i < 4; ++i)
            buf_.push_back(static_cast<char>((bits >> (8 * i)) & 0xFF));
    }

    // ── length-delimited ─────────────────────────────────────────────
    void write_string(int field, std::string_view value) {
        write_tag(field, WireType::LengthDelimited);
        append_varint(value.size());
        buf_.append(value.data(), value.size());
    }

    void write_bytes(int field, const void* data, std::size_t size) {
        write_tag(field, WireType::LengthDelimited);
        append_varint(size);
        buf_.append(static_cast<const char*>(data), size);
    }

    // Concatenate another writer's fields into this one.
    //
    // Protobuf has no field order, so two writers describing the same
    // message can simply be joined — which is how a message assembled in
    // pieces (a function's name here, its inputs and outputs there) comes
    // back together without re-encoding either half.
    void append_raw(const ProtoWriter& other) {
        buf_.append(other.buf_.data(), other.buf_.size());
    }

    // Embed an already-built submessage.
    void write_message(int field, const ProtoWriter& sub) {
        write_bytes(field, sub.buf_.data(), sub.buf_.size());
    }

    // ``repeated`` scalars in proto3 default to packed: one
    // length-delimited field holding the concatenated varints.
    void write_packed_ints(int field, const std::vector<std::int64_t>& values) {
        ProtoWriter packed;
        for (std::int64_t v : values)
            packed.append_varint(static_cast<std::uint64_t>(v));
        write_bytes(field, packed.buf_.data(), packed.buf_.size());
    }

    void write_packed_floats(int field, const std::vector<float>& values) {
        ProtoWriter packed;
        for (float v : values) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &v, sizeof(bits));
            for (int i = 0; i < 4; ++i)
                packed.buf_.push_back(static_cast<char>((bits >> (8 * i)) & 0xFF));
        }
        write_bytes(field, packed.buf_.data(), packed.buf_.size());
    }

    // ── map<K, V> ────────────────────────────────────────────────────
    //
    // A protobuf map is sugar for ``repeated Entry { key = 1; value = 2; }``,
    // so each entry is one occurrence of the map's own field number.
    void write_map_entry(int field, std::string_view key, const ProtoWriter& value) {
        ProtoWriter entry;
        entry.write_string(1, key);
        entry.write_message(2, value);
        write_message(field, entry);
    }

    // ── result ───────────────────────────────────────────────────────
    const std::string& bytes() const { return buf_; }
    std::size_t size() const { return buf_.size(); }
    bool empty() const { return buf_.empty(); }
    void clear() { buf_.clear(); }

private:
    void write_tag(int field, WireType type) {
        append_varint((static_cast<std::uint64_t>(field) << 3) | static_cast<std::uint64_t>(type));
    }

    void append_varint(std::uint64_t value) {
        // Seven bits per byte, high bit set while more bytes follow.
        while (value >= 0x80) {
            buf_.push_back(static_cast<char>((value & 0x7F) | 0x80));
            value >>= 7;
        }
        buf_.push_back(static_cast<char>(value));
    }

    std::string buf_;
};

}  // namespace lucid::coreml
