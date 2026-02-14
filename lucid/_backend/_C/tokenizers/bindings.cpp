#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "tokenizers.hpp"

namespace py = pybind11;

using lucid::tokenizers::fast::WordPieceTokenizer;

PYBIND11_MODULE(core, m) {
    m.doc() = "Lucid tokenizers C++ bindings";

    py::class_<WordPieceTokenizer>(m, "_C_WordPieceTokenizer")
        .def(
            py::init<
                std::optional<WordPieceTokenizer::Vocab>,
                std::optional<std::filesystem::path>,
                std::string,
                std::string,
                std::optional<std::string>,
                std::optional<std::string>,
                std::string,
                std::size_t,
                bool,
                bool
            >(),
            py::arg("vocab") = std::nullopt,
            py::arg("vocab_file") = std::nullopt,
            py::arg("unk_token") = "[UNK]",
            py::arg("pad_token") = "[PAD]",
            py::arg("bos_token") = std::nullopt,
            py::arg("eos_token") = std::nullopt,
            py::arg("wordpieces_prefix") = "##",
            py::arg("max_input_chars_per_word") = 100,
            py::arg("lowercase") = true,
            py::arg("clean_text") = true
        )
        .def("vocab_size", &WordPieceTokenizer::vocab_size)
        .def("tokenize", &WordPieceTokenizer::tokenize, py::arg("text"))
        .def("convert_token_to_id", &WordPieceTokenizer::convert_token_to_id, py::arg("token"))
        .def("convert_tokens_to_ids", &WordPieceTokenizer::convert_tokens_to_ids, py::arg("tokens"))
        .def("convert_id_to_token", &WordPieceTokenizer::convert_id_to_token, py::arg("id"))
        .def("convert_ids_to_tokens", &WordPieceTokenizer::convert_ids_to_tokens, py::arg("ids"))
        .def("convert_tokens_to_string", &WordPieceTokenizer::convert_tokens_to_string, py::arg("tokens"))
        .def(
            "fit",
            [](WordPieceTokenizer& self,
               const std::vector<std::string>& texts,
               std::size_t vocab_size,
               std::size_t min_frequency) -> WordPieceTokenizer& {
                return self.fit(texts, vocab_size, min_frequency);
            },
            py::arg("texts"),
            py::arg("vocab_size"),
            py::arg("min_frequency") = 2,
            py::return_value_policy::reference_internal
        );
}
