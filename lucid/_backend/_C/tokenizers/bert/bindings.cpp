#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "bert.hpp"

#include <optional>
#include <string>
#include <vector>

namespace py = pybind11;

using lucid::tokenizers::fast::BERTEncodePairResult;
using lucid::tokenizers::fast::BERTTokenizer;

PYBIND11_MODULE(core, m) {
    m.doc() = "Lucid BERT tokenizer C++ bindings";

    py::class_<BERTEncodePairResult>(m, "_C_BERTEncodePairResult")
        .def(py::init<>())
        .def_readwrite("input_ids", &BERTEncodePairResult::input_ids)
        .def_readwrite("token_type_ids", &BERTEncodePairResult::token_type_ids)
        .def_readwrite("attention_mask", &BERTEncodePairResult::attention_mask);

    py::class_<BERTTokenizer>(m, "_C_BERTTokenizer")
        .def(
            py::init<
                std::optional<BERTTokenizer::Vocab>,
                std::optional<std::filesystem::path>,
                std::string,
                std::string,
                std::string,
                std::string,
                std::string,
                std::size_t,
                bool,
                bool>(),
            py::arg("vocab") = std::nullopt,
            py::arg("vocab_file") = std::nullopt,
            py::arg("unk_token") = "[UNK]",
            py::arg("pad_token") = "[PAD]",
            py::arg("cls_token") = "[CLS]",
            py::arg("mask_token") = "[MASK]",
            py::arg("wordpieces_prefix") = "##",
            py::arg("max_input_chars_per_word") = 100,
            py::arg("lowercase") = true,
            py::arg("clean_text") = true)
        .def("vocab_size", &BERTTokenizer::vocab_size)
        .def("tokenize", &BERTTokenizer::tokenize, py::arg("text"))
        .def("convert_token_to_id", &BERTTokenizer::convert_token_to_id, py::arg("token"))
        .def("convert_tokens_to_ids", &BERTTokenizer::convert_tokens_to_ids, py::arg("tokens"))
        .def("convert_id_to_token", &BERTTokenizer::convert_id_to_token, py::arg("id"))
        .def("convert_ids_to_tokens", &BERTTokenizer::convert_ids_to_tokens, py::arg("ids"))
        .def("convert_tokens_to_string", &BERTTokenizer::convert_tokens_to_string, py::arg("tokens"))
        .def(
            "build_inputs_with_special_tokens",
            &BERTTokenizer::build_inputs_with_special_tokens,
            py::arg("tokens"))
        .def(
            "build_inputs_with_special_tokens_pair",
            &BERTTokenizer::build_inputs_with_special_tokens_pair,
            py::arg("tokens_a"),
            py::arg("tokens_b"))
        .def(
            "create_token_type_ids_from_sequences",
            &BERTTokenizer::create_token_type_ids_from_sequences,
            py::arg("tokens_a"),
            py::arg("tokens_b") = std::nullopt)
        .def(
            "encode_plus",
            [](const BERTTokenizer& self,
               const std::string& text_a,
               std::optional<std::string> text_b) {
                if (text_b.has_value()) {
                    return self.encode_plus(
                        std::string_view(text_a),
                        std::optional<std::string_view>(*text_b));
                }
                return self.encode_plus(std::string_view(text_a), std::nullopt);
            },
            py::arg("text_a"),
            py::arg("text_b") = std::nullopt)
        .def("cls_token", &BERTTokenizer::cls_token)
        .def("sep_token", &BERTTokenizer::sep_token)
        .def("mask_token", &BERTTokenizer::mask_token);
}
