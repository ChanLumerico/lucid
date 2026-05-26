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
#include "../utils/tokenizer/Tokenizer.h"

namespace py = pybind11;

namespace lucid::bindings {

namespace tk = lucid::utils::tokenizer;

void register_tokenizer(py::module_& parent) {
    // ``parent`` is the ``utils`` sub-module created in bind.cpp; create
    // ``tokenizer`` as a child so the dotted path mirrors the Python
    // package (``lucid._C.engine.utils.tokenizer``).
    auto m = parent.def_submodule(
        "tokenizer",
        "Tokenization primitives (BPE / WordPiece / Unigram / ByteLevelBPE).  "
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
        .def("train",
             &tk::Tokenizer::train,
             py::arg("corpus"),
             py::arg("target_vocab_size"))
        .def_property("special_tokens",
                      &tk::Tokenizer::special_tokens,
                      &tk::Tokenizer::set_special_tokens);

    // ── BPE ────────────────────────────────────────────────────────
    py::class_<tk::BPE, tk::Tokenizer>(m, "BPE")
        .def(py::init<>())
        .def(py::init<std::unordered_map<std::string, tk::TokenId>,
                      std::vector<std::pair<std::string, std::string>>>(),
             py::arg("vocab"),
             py::arg("merges"))
        .def("merges", &tk::BPE::merges,
             "Ordered list of (left, right) merge pairs — rank ascending.");
}

}  // namespace lucid::bindings
