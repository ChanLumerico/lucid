// lucid/_C/bindings/bind_tokenizer.cpp
//
// pybind11 surface for ``lucid._C.engine.utils.tokenizer``.  Mirrors
// the Python :mod:`lucid.utils.tokenizer` sub-package; every C++
// tokenizer is exposed as ``_C.engine.utils.tokenizer.<Name>`` and
// wrapped by the matching ``<Name>TokenizerFast`` Python class.
//
// Exported types
// --------------
// * ``SpecialTokens`` — value type holding the canonical 7 slots
//   plus a string-keyed extras map.
// * ``BPE`` — Byte-Pair Encoding (Sennrich et al. 2016).
//
// Future tokenizers (WordPiece / Unigram / ByteLevelBPE) plug in by
// adding a ``py::class_<XXX, Tokenizer>`` block here.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../utils/tokenizer/BPE.h"
#include "../utils/tokenizer/Basic.h"
#include "../utils/tokenizer/Tokenizer.h"
#include "../utils/tokenizer/Unigram.h"
#include "../utils/tokenizer/WordPiece.h"

namespace py = pybind11;

namespace lucid::bindings {

namespace tk = lucid::utils::tokenizer;

void register_tokenizer(py::module_& parent) {
    // ``parent`` is the ``utils`` sub-module created in bind.cpp; create
    // ``tokenizer`` as a child so the dotted path mirrors the Python
    // package (``lucid._C.engine.utils.tokenizer``).
    auto m = parent.def_submodule(
        "tokenizer", "Tokenization primitives (BPE / WordPiece / Unigram / ByteLevelBPE).  "
                     "Wrapped by lucid.utils.tokenizer.*TokenizerFast.");

    // ── SpecialTokens ──────────────────────────────────────────────
    py::class_<tk::SpecialTokens>(m, "SpecialTokens")
        .def(py::init<>())
        .def_readwrite("pad", &tk::SpecialTokens::pad)
        .def_readwrite("unk", &tk::SpecialTokens::unk)
        .def_readwrite("bos", &tk::SpecialTokens::bos)
        .def_readwrite("eos", &tk::SpecialTokens::eos)
        .def_readwrite("mask", &tk::SpecialTokens::mask)
        .def_readwrite("sep", &tk::SpecialTokens::sep)
        .def_readwrite("cls", &tk::SpecialTokens::cls)
        .def_readwrite("extra", &tk::SpecialTokens::extra);

    // ── Tokenizer base ─────────────────────────────────────────────
    // Expose as a non-instantiable base so the Python ``isinstance``
    // checks work across all concrete subclasses.
    py::class_<tk::Tokenizer>(m, "Tokenizer")
        .def("encode", &tk::Tokenizer::encode, py::arg("text"))
        .def("decode", &tk::Tokenizer::decode, py::arg("ids"))
        .def("vocab_size", &tk::Tokenizer::vocab_size)
        .def("algo", &tk::Tokenizer::algo)
        .def("encode_batch", &tk::Tokenizer::encode_batch, py::arg("texts"))
        .def("decode_batch", &tk::Tokenizer::decode_batch, py::arg("batch"))
        .def("get_vocab", &tk::Tokenizer::get_vocab)
        .def("id_to_token", &tk::Tokenizer::id_to_token, py::arg("id"))
        .def("train", &tk::Tokenizer::train, py::arg("corpus"), py::arg("target_vocab_size"))
        .def_property("special_tokens", &tk::Tokenizer::special_tokens,
                      &tk::Tokenizer::set_special_tokens);

    // ── BPE ────────────────────────────────────────────────────────
    py::class_<tk::BPE, tk::Tokenizer>(m, "BPE")
        .def(py::init<>())
        .def(py::init<std::unordered_map<std::string, tk::TokenId>,
                      std::vector<std::pair<std::string, std::string>>>(),
             py::arg("vocab"), py::arg("merges"))
        .def("merges", &tk::BPE::merges,
             "Ordered list of (left, right) merge pairs — rank ascending.");

    // ── Basic lookup family ────────────────────────────────────────
    //
    // All 5 derive from ``LookupTokenizer`` (which itself derives
    // from ``Tokenizer``).  We expose ``LookupTokenizer`` as the
    // pybind base so the ``isinstance(_, LookupTokenizer)`` test
    // also works on the Python side.
    py::class_<tk::LookupTokenizer, tk::Tokenizer>(m, "LookupTokenizer");

    py::class_<tk::ByteTokenizer, tk::LookupTokenizer>(m, "ByteTokenizer").def(py::init<>());

    py::class_<tk::CharTokenizer, tk::LookupTokenizer>(m, "CharTokenizer")
        .def(py::init<>())
        .def(py::init<std::unordered_map<std::string, tk::TokenId>>(), py::arg("vocab"));

    py::class_<tk::WhitespaceTokenizer, tk::LookupTokenizer>(m, "WhitespaceTokenizer")
        .def(py::init<>())
        .def(py::init<std::unordered_map<std::string, tk::TokenId>>(), py::arg("vocab"));

    py::class_<tk::WordTokenizer, tk::LookupTokenizer>(m, "WordTokenizer")
        .def(py::init<>())
        .def(py::init<std::unordered_map<std::string, tk::TokenId>>(), py::arg("vocab"));

    py::class_<tk::RegexTokenizer, tk::LookupTokenizer>(m, "RegexTokenizer")
        .def(py::init<std::string, std::unordered_map<std::string, tk::TokenId>>(),
             py::arg("pattern"), py::arg("vocab") = std::unordered_map<std::string, tk::TokenId>{})
        .def("pattern", &tk::RegexTokenizer::pattern, py::return_value_policy::reference_internal);

    // ── WordPiece ──────────────────────────────────────────────────
    py::class_<tk::WordPiece, tk::Tokenizer>(m, "WordPiece")
        .def(py::init<>())
        .def(py::init<std::unordered_map<std::string, tk::TokenId>, std::string, std::string,
                      std::size_t>(),
             py::arg("vocab"), py::arg("unk_token") = "[UNK]", py::arg("continuing_prefix") = "##",
             py::arg("max_chars_per_word") = 100)
        .def("unk_token", &tk::WordPiece::unk_token, py::return_value_policy::reference_internal)
        .def("continuing_prefix", &tk::WordPiece::continuing_prefix,
             py::return_value_policy::reference_internal);

    // ── Unigram ────────────────────────────────────────────────────
    py::class_<tk::Unigram, tk::Tokenizer>(m, "Unigram")
        .def(py::init<>())
        .def(py::init<std::vector<std::pair<std::string, double>>, std::string, double>(),
             py::arg("pieces"), py::arg("unk_token") = "<unk>", py::arg("unk_log_prob") = -100.0)
        .def("pieces", &tk::Unigram::pieces, py::return_value_policy::reference_internal,
             "Ordered list of (piece_str, log_prob) — index = id.")
        .def("unk_token", &tk::Unigram::unk_token, py::return_value_policy::reference_internal)
        .def("unk_log_prob", &tk::Unigram::unk_log_prob)
        .def("train_with_options", &tk::Unigram::train_with_options, py::arg("corpus"),
             py::arg("target_vocab_size"), py::arg("num_iterations") = 2,
             py::arg("shrink_factor") = 0.75, py::arg("max_piece_length") = 16,
             py::arg("initial_vocab_multiplier") = 10);
}

}  // namespace lucid::bindings
