import setuptools
from setuptools import Extension

try:
    import pybind11
except ImportError as e:
    raise RuntimeError(
        "pybind11 is required to build C++ extensions. "
        "Install it first: pip install pybind11"
    ) from e


with open("README.md", "r") as fh:
    long_description = fh.read()

_raw_version = "{{VERSION_PLACEHOLDER}}"
PACKAGE_VERSION = "0.0.0" if _raw_version.startswith("{{") else _raw_version

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

setuptools.setup(
    name="lucid-dl",
    version=PACKAGE_VERSION,
    author="ChanLumerico",
    author_email="greensox284@gmail.com",
    description="Lumerico's Comprehensive Interface for Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChanLumerico/lucid",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Operating System :: MacOS :: MacOS X",
    ],
    python_requires=">=3.12",
    install_requires=[
        "numpy",
        "pandas",
        "openml",
        "mlx; platform_system == 'Darwin' and platform_machine == 'arm64'",
    ],
    ext_modules=[tokenizers_cpp],
    include_package_data=True,
)
