import argparse
import re
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--python-bin", default="python3")
    p.add_argument("--mode", choices=("inplace", "dist"), default="inplace")
    p.add_argument("cpp_files", nargs="*")
    return p.parse_args()


def _make_pbar(total: int):
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm(
            total=total,
            desc="cpp compile",
            unit="file",
            dynamic_ncols=True,
            leave=True,
        )
    except Exception:
        return None


def _print_plain_progress(current: int, total: int, name: str) -> None:
    width = 32
    filled = int(width * current / max(total, 1))
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r[cpp {bar}] {current}/{total} {name}")
    sys.stdout.flush()


def main() -> int:
    args = _parse_args()
    total_cpp = len(args.cpp_files)

    cmd = [args.python_bin, "setup.py"]
    if args.mode == "inplace":
        cmd += ["build_ext", "--inplace"]
    else:
        cmd += ["sdist", "bdist_wheel"]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    compile_re = re.compile(r" -c (\S+\.cpp)\b")
    link_re = re.compile(r"\s-o\s+\S+\.(?:so|dylib|a)\b")
    current = 0

    pbar = _make_pbar(total_cpp) if total_cpp > 0 else None
    using_plain = pbar is None and total_cpp > 0

    for raw in proc.stdout:
        line = raw.rstrip("\n")
        match = compile_re.search(line)
        if match:
            current += 1
            cpp_name = Path(match.group(1)).name
            if pbar is not None:
                pbar.set_postfix_str(cpp_name)
                pbar.update(1)
            elif using_plain:
                _print_plain_progress(current, total_cpp, cpp_name)
            continue

        if line.startswith(("clang++ ", "g++ ", "c++ ")) or link_re.search(line):
            continue

        if using_plain:
            sys.stdout.write("\n")
            sys.stdout.flush()
            using_plain = False
        print(line)

    rc = proc.wait()
    if pbar is not None:
        pbar.close()
    elif total_cpp > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
