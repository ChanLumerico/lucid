import json
import setuptools
from setuptools import Extension
from pathlib import Path

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

BUILD_CFG_PATH = Path(".extensions/cpp_extensions.json")
with BUILD_CFG_PATH.open("r", encoding="utf-8") as f:
    build_cfg = json.load(f)

common = build_cfg.get("common", {})
extensions_cfg = build_cfg.get("extensions", [])
if not isinstance(extensions_cfg, list) or not extensions_cfg:
    raise RuntimeError(f"Invalid extension config: {BUILD_CFG_PATH}")

ext_modules: list[Extension] = []
for ext_cfg in extensions_cfg:
    ext_modules.append(
        Extension(
            ext_cfg["name"],
            sources=ext_cfg["sources"],
            include_dirs=[pybind11.get_include()],
            language=common.get("language", "c++"),
            extra_compile_args=common.get("extra_compile_args", ["-std=c++20"]),
        )
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
    packages=setuptools.find_namespace_packages(include=["lucid*"]),
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
    ext_modules=ext_modules,
    include_package_data=True,
)
