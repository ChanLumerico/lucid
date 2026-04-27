import json
import os
import setuptools
import shutil
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


# ----------------------------------------------------------------------
# Build acceleration: ccache (recompile cache) + parallel object compile.
# ----------------------------------------------------------------------
# Both controls can be opted out with env vars. Defaults: enabled when ccache
# is on PATH; parallel compile uses every core.
#
#   LUCID_USE_CCACHE=0   — disable ccache wrapping (forces recompiles)
#   LUCID_BUILD_JOBS=N   — override the worker-process count
#                          (default: os.cpu_count(); set to 1 for serial)

def _maybe_enable_ccache() -> None:
    if os.environ.get("LUCID_USE_CCACHE", "1") == "0":
        return
    ccache = shutil.which("ccache")
    if not ccache:
        return
    # If user already set CC/CXX with ccache, don't double-wrap.
    cc = os.environ.get("CC", "clang")
    cxx = os.environ.get("CXX", "clang++")
    if "ccache" not in cc:
        os.environ["CC"] = f"{ccache} {cc}"
    if "ccache" not in cxx:
        os.environ["CXX"] = f"{ccache} {cxx}"

def _enable_parallel_compile() -> None:
    """Monkeypatch CCompiler.compile to dispatch sources across a process pool.

    setuptools/distutils still build object files serially even when many TUs
    are independent. We replace the inner per-source loop with a
    multiprocessing pool that runs `_compile()` for each source in parallel.
    The default worker count is `os.cpu_count()`; override via
    LUCID_BUILD_JOBS."""
    try:
        jobs_env = os.environ.get("LUCID_BUILD_JOBS")
        n_jobs = int(jobs_env) if jobs_env else (os.cpu_count() or 1)
    except ValueError:
        n_jobs = os.cpu_count() or 1
    if n_jobs <= 1:
        return

    try:
        import distutils.ccompiler as _cc
    except ImportError:
        return

    def parallel_compile(self, sources, output_dir=None, macros=None,
                         include_dirs=None, debug=0, extra_preargs=None,
                         extra_postargs=None, depends=None):
        macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
            output_dir, macros, include_dirs, sources, depends, extra_postargs
        )
        cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

        def _single(obj):
            try:
                src, ext = build[obj]
            except KeyError:
                return
            self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        from concurrent.futures import ThreadPoolExecutor
        # Use threads — `_compile` ultimately spawns a clang subprocess per
        # call, so the GIL doesn't bottleneck us; we just need many concurrent
        # subprocesses. (Process pool would also work but adds pickling cost.)
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            list(pool.map(_single, objects))
        return objects

    _cc.CCompiler.compile = parallel_compile


_maybe_enable_ccache()
_enable_parallel_compile()

_raw_version = "{{VERSION_PLACEHOLDER}}"
PACKAGE_VERSION = "0.0.0" if _raw_version.startswith("{{") else _raw_version

BUILD_CFG_PATH = Path(".extensions/cpp_extensions.json")
with BUILD_CFG_PATH.open("r", encoding="utf-8") as f:
    build_cfg = json.load(f)

common = build_cfg.get("common", {})
extensions_cfg = build_cfg.get("extensions", [])
if not isinstance(extensions_cfg, list) or not extensions_cfg:
    raise RuntimeError(f"Invalid extension config: {BUILD_CFG_PATH}")

import os as _os

# LUCID_BUILD_MODE controls debug/sanitizer flavors. Defaults to release.
# Recognized values:
#   release       — production; -O3 -DNDEBUG (default)
#   debug         — -O0 -g
#   debug-asan    — -O1 -g -fsanitize=address,undefined  (clang must enable both)
#   debug-tsan    — -O1 -g -fsanitize=thread
#   debug-ubsan   — -O1 -g -fsanitize=undefined
_LUCID_BUILD_MODE = _os.environ.get("LUCID_BUILD_MODE", "release").lower()

def _sanitizer_flags(mode: str) -> tuple[list[str], list[str]]:
    """Returns (extra_compile_args, extra_link_args) for the given build mode."""
    if mode == "release":
        return ([], [])
    if mode == "debug":
        return (["-O0", "-g", "-fno-omit-frame-pointer"], ["-g"])
    if mode == "debug-asan":
        flags = ["-O1", "-g", "-fno-omit-frame-pointer",
                 "-fsanitize=address,undefined"]
        return (flags, ["-fsanitize=address,undefined"])
    if mode == "debug-tsan":
        flags = ["-O1", "-g", "-fno-omit-frame-pointer", "-fsanitize=thread"]
        return (flags, ["-fsanitize=thread"])
    if mode == "debug-ubsan":
        flags = ["-O1", "-g", "-fno-omit-frame-pointer",
                 "-fsanitize=undefined"]
        return (flags, ["-fsanitize=undefined"])
    raise RuntimeError(f"Unknown LUCID_BUILD_MODE={mode!r}")

_SAN_COMPILE, _SAN_LINK = _sanitizer_flags(_LUCID_BUILD_MODE)


def _resolve_per_ext(ext_cfg: dict) -> dict:
    include_dirs = [pybind11.get_include()]
    library_dirs: list[str] = []
    libraries: list[str] = []
    extra_link_args: list[str] = []

    if ext_cfg.get("link_accelerate"):
        # Accelerate ships with macOS — no extra include/library paths needed,
        # just `-framework Accelerate` (already inlined into extra_link_args).
        pass

    if ext_cfg.get("link_mlx"):
        # `mlx` is a namespace package (mlx.__file__ is None on newer wheels);
        # locate the install directory via `mlx.core` and walk one level up.
        try:
            import mlx.core as _mlx_core
            core_path = Path(_mlx_core.__file__).resolve()
            mlx_pkg = core_path.parent
        except ImportError as e:
            raise RuntimeError(
                f"Extension {ext_cfg['name']} requested link_mlx but mlx is not "
                f"installed. Install it first: pip install mlx"
            ) from e
        include_dirs.append(str(mlx_pkg / "include"))
        library_dirs.append(str(mlx_pkg / "lib"))
        libraries.append("mlx")
        # Embed an rpath so the runtime linker finds libmlx.dylib without
        # DYLD_LIBRARY_PATH gymnastics.
        extra_link_args.extend([f"-Wl,-rpath,{mlx_pkg / 'lib'}"])

    include_dirs.extend(ext_cfg.get("include_dirs", []))
    library_dirs.extend(ext_cfg.get("library_dirs", []))
    libraries.extend(ext_cfg.get("libraries", []))
    extra_link_args.extend(ext_cfg.get("extra_link_args", []))

    extra_compile_args = list(ext_cfg.get(
        "extra_compile_args",
        common.get("extra_compile_args", ["-std=c++20"]),
    ))

    # Apply sanitizer flags only to extensions opted-in via "sanitizable": true.
    # Tokenizer extension keeps release flags even under LUCID_BUILD_MODE=debug-asan
    # to avoid pulling all of pybind11/STL into ASan instrumentation noise.
    if ext_cfg.get("sanitizable") and _LUCID_BUILD_MODE != "release":
        # When sanitizers are on, drop -O3 in favor of the sanitizer's preferred -O1.
        extra_compile_args = [a for a in extra_compile_args
                              if not a.startswith("-O")]
        extra_compile_args.extend(_SAN_COMPILE)
        extra_link_args.extend(_SAN_LINK)

    return {
        "include_dirs": include_dirs,
        "library_dirs": library_dirs,
        "libraries": libraries,
        "extra_compile_args": extra_compile_args,
        "extra_link_args": extra_link_args,
    }


ext_modules: list[Extension] = []
for ext_cfg in extensions_cfg:
    resolved = _resolve_per_ext(ext_cfg)
    ext_modules.append(
        Extension(
            ext_cfg["name"],
            sources=ext_cfg["sources"],
            language=common.get("language", "c++"),
            **resolved,
        )
    )

setuptools.setup(
    name="lucid-dl",
    version=PACKAGE_VERSION,
    license="MIT",
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
    extras_require={
        "test": [
            "pytest>=7.0",
            "torch>=2.3,<2.7",
            "einops>=0.8",
            "tqdm",
        ],
    },
    package_data={
        "lucid.weights": ["registry.json", "__init__.pyi"],
        "lucid.models": ["registry.json"],
    },
    ext_modules=ext_modules,
    include_package_data=True,
)
