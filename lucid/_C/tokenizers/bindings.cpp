#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "tokenizers.hpp"

namespace py = pybind11;

using lucid::tokenizers::fast::WordPieceTokenizer;
using lucid::tokenizers::fast::BPETokenizer;
using lucid::tokenizers::fast::ByteBPETokenizer;

PYBIND11_MODULE(core, m) {
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
            [](
                WordPieceTokenizer& self,
                const std::vector<std::string>& texts,
                std::size_t vocab_size,
                std::size_t min_frequency
            ) -> WordPieceTokenizer& {
                return self.fit(texts, vocab_size, min_frequency);
            },
            py::arg("texts"),
            py::arg("vocab_size"),
            py::arg("min_frequency") = 2,
            py::return_value_policy::reference_internal
        );

    py::class_<BPETokenizer>(m, "_C_BPETokenizer")
        .def(
            py::init<
                std::optional<BPETokenizer::Vocab>,
                std::optional<std::vector<BPETokenizer::Merge>>,
                std::optional<std::filesystem::path>,
                std::optional<std::filesystem::path>,
                std::string,
                std::string,
                std::optional<std::string>,
                std::optional<std::string>,
                bool,
                bool,
                std::string
            >(),
            py::arg("vocab") = std::nullopt,
            py::arg("merges") = std::nullopt,
            py::arg("vocab_file") = std::nullopt,
            py::arg("merges_file") = std::nullopt,
            py::arg("unk_token") = "[UNK]",
            py::arg("pad_token") = "[PAD]",
            py::arg("bos_token") = std::nullopt,
            py::arg("eos_token") = std::nullopt,
            py::arg("lowercase") = true,
            py::arg("clean_text") = true,
            py::arg("end_of_word_suffix") = "</w>"
        )
        .def("vocab_size", &BPETokenizer::vocab_size)
        .def("tokenize", &BPETokenizer::tokenize, py::arg("text"))
        .def(
            "encode_ids",
            &BPETokenizer::encode_ids,
            py::arg("text"),
            py::arg("add_special_tokens") = true
        )
        .def("convert_token_to_id", &BPETokenizer::convert_token_to_id, py::arg("token"))
        .def("convert_tokens_to_ids", &BPETokenizer::convert_tokens_to_ids, py::arg("tokens"))
        .def("convert_id_to_token", &BPETokenizer::convert_id_to_token, py::arg("id"))
        .def("convert_ids_to_tokens", &BPETokenizer::convert_ids_to_tokens, py::arg("ids"))
        .def("convert_tokens_to_string", &BPETokenizer::convert_tokens_to_string, py::arg("tokens"))
        .def("get_vocab", &BPETokenizer::vocab, py::return_value_policy::copy)
        .def("get_merges", &BPETokenizer::merges, py::return_value_policy::copy)
        .def(
            "fit",
            [](
                BPETokenizer& self,
                const std::vector<std::string>& texts,
                std::size_t vocab_size,
                std::size_t min_frequency
            ) -> BPETokenizer& {
                return self.fit(texts, vocab_size, min_frequency);
            },
            py::arg("texts"),
            py::arg("vocab_size"),
            py::arg("min_frequency") = 2,
            py::return_value_policy::reference_internal
        );

    py::class_<ByteBPETokenizer, BPETokenizer>(m, "_C_ByteBPETokenizer")
        .def(
            py::init<
                std::optional<ByteBPETokenizer::Vocab>,
                std::optional<std::vector<ByteBPETokenizer::Merge>>,
                std::optional<std::filesystem::path>,
                std::optional<std::filesystem::path>,
                std::string,
                std::string,
                std::optional<std::string>,
                std::optional<std::string>,
                bool,
                bool,
                bool,
                std::string
            >(),
            py::arg("vocab") = std::nullopt,
            py::arg("merges") = std::nullopt,
            py::arg("vocab_file") = std::nullopt,
            py::arg("merges_file") = std::nullopt,
            py::arg("unk_token") = "[UNK]",
            py::arg("pad_token") = "[PAD]",
            py::arg("bos_token") = std::nullopt,
            py::arg("eos_token") = std::nullopt,
            py::arg("lowercase") = false,
            py::arg("clean_text") = true,
            py::arg("add_prefix_space") = false,
            py::arg("end_of_word_suffix") = ""
        )
        .def("vocab_size", &ByteBPETokenizer::vocab_size)
        .def("tokenize", &ByteBPETokenizer::tokenize, py::arg("text"))
        .def("encode_ids", &ByteBPETokenizer::encode_ids, py::arg("text"), py::arg("add_special_tokens") = true)
        .def("convert_token_to_id", &ByteBPETokenizer::convert_token_to_id, py::arg("token"))
        .def("convert_tokens_to_ids", &ByteBPETokenizer::convert_tokens_to_ids, py::arg("tokens"))
        .def("convert_id_to_token", &ByteBPETokenizer::convert_id_to_token, py::arg("id"))
        .def("convert_ids_to_tokens", &ByteBPETokenizer::convert_ids_to_tokens, py::arg("ids"))
        .def("convert_tokens_to_string", &ByteBPETokenizer::convert_tokens_to_string, py::arg("tokens"))
        .def("get_vocab", &ByteBPETokenizer::vocab, py::return_value_policy::copy)
        .def("get_merges", &ByteBPETokenizer::merges, py::return_value_policy::copy)
        .def("add_prefix_space", &ByteBPETokenizer::add_prefix_space)
        .def(
            "fit", 
            [](
                ByteBPETokenizer& self, 
                const std::vector<std::string>& texts, 
                std::size_t vocab_size, 
                std::size_t min_frequency
            ) -> ByteBPETokenizer& { 
                return self.fit(texts, vocab_size, min_frequency); 
            }, 
            py::arg("texts"), 
            py::arg("vocab_size"), 
            py::arg("min_frequency") = 2, 
            py::return_value_policy::reference_internal
        );
}
