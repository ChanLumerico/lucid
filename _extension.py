from setuptools import Extension

try:
    import pybind11
except ImportError as e:
    raise RuntimeError(
        "pybind11 is required to build C++ extensions. "
        "Install it first: pip install pybind11"
    ) from e


__all__ = ["tokenizers_cpp"]


tokenizers_cpp = Extension(
    "lucid.data.tokenizers._C",
    sources=[
        "lucid/_backend/_C/tokenizers/base.cpp",
        "lucid/_backend/_C/tokenizers/wordpiece.cpp",
        "lucid/_backend/_C/tokenizers/bindings.cpp",
    ],
    include_dirs=[
        pybind11.get_include(),
    ],
    language="c++",
    extra_compile_args=["-std=c++20"],
)
