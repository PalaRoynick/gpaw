"""Checks that Python.h is included before system headers:
https://docs.python.org/3/c-api/intro.html#include-files.
This is done by reading actual build commands from compile_commands.json and
appending -E -H (preprocess only + header trace). Should work for gcc/clang
based compilers, including nvcc. You can generate a suitable
compile_commands.json by wrapping your build with the Bear program (needed also
for clang-tidy).

More specifically, this checks that if Python.h is used in a TU, it must be
the first processed header EXCLUDING:
    - Our own headers. These are safe and many of our files anyway include some
      convenience header like python_utils.h, which #includes Python.h.
    - Pybind11 headers. These include Python.h anyway (well, pybind11.h does).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_INCLUDE_ROOT = (REPO_ROOT / "c").resolve()


# Dependency files (-MMD et al.) would be generated in preprocess-only mode,
# so strip those args manually if present.
NO_ARG_FLAGS_TO_DROP = {"-MMD", "-MD", "-MP", "-M", "-MM"}
FLAGS_WITH_ARG_TO_DROP = {"-MF", "-MT", "-MQ"}


def preprocess_command(entry: dict) -> list[str]:
    """Rewrites a compile_commands.json entry's command into a
    preprocess-only header-trace invocation."""
    args = list(entry["arguments"])
    cleaned = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in NO_ARG_FLAGS_TO_DROP:
            continue
        if arg in FLAGS_WITH_ARG_TO_DROP:
            skip_next = True
            continue
        cleaned.append(arg)
    return cleaned[:1] + ["-E", "-H"] + cleaned[1:]


def header_trace(entry: dict) -> list[str]:
    """Returns the paths of every header opened during preprocessing, in
    the order the compiler actually opened them."""
    result = subprocess.run(
        preprocess_command(entry),
        cwd=entry["directory"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    headers = []
    for line in result.stderr.splitlines():
        stripped = line.lstrip(".")
        if stripped != line and stripped.startswith(" "):
            headers.append(stripped.strip())
    return headers


def is_whitelisted(resolved: Path) -> bool:
    """Pybind11 is safe as first header: it includes Python.h anyway."""
    return "pybind11" in resolved.parts


def first_violation(entry: dict) -> str | None:
    headers = header_trace(entry)
    python_h_index = next(
        (i for i, h in enumerate(headers) if h.endswith("Python.h")), None
    )
    if python_h_index is None:
        return None  # this TU doesn't touch Python at all
    directory = Path(entry["directory"])
    for h in headers[:python_h_index]:
        path = Path(h)
        resolved = path if path.is_absolute() else (directory / path)
        try:
            resolved = resolved.resolve()
        except OSError:
            continue
        if (PROJECT_INCLUDE_ROOT not in resolved.parents
            and not is_whitelisted(resolved)):
            return h
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "compile_commands",
        type=Path,
        help="Path to compile_commands.json",
    )
    args = parser.parse_args()

    entries = json.loads(args.compile_commands.read_text())
    bad = []
    for entry in entries:
        violation = first_violation(entry)
        if violation is not None:
            bad.append((entry["file"], violation))

    if bad:
        print("Include order check failed for:")
        for file, violation in bad:
            print(f"  {file}: {violation!r} is preprocessed before Python.h")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
