#!/usr/bin/env python3
"""Prints GPAW siteconfig.py lines for the flags `mpicxx -show` (OpenMPI)
reports it would inject. Useful when a build uses the real compiler
directly instead of the mpicxx wrapper (so that compile_commands.json
ends up with explicit, correct flags instead of ones only the wrapper
would have known to add -- see tools/clang_tidy_extra_args.py for the
same underlying problem on the clang-tidy side).

Usage: append its output straight into siteconfig.py, e.g.:
    python3 tools/print_mpi_siteconfig.py >> siteconfig.py
"""
import shlex
import subprocess
import sys


def mpi_flags() -> dict[str, list[str]]:
    result = subprocess.run(
        ["mpicxx", "-show"], capture_output=True, text=True, check=True
    )
    flags: dict[str, list[str]] = {
        "include_dirs": [],
        "library_dirs": [],
        "libraries": [],
    }
    for token in shlex.split(result.stdout):
        if token.startswith("-I"):
            flags["include_dirs"].append(token[2:])
        elif token.startswith("-L"):
            flags["library_dirs"].append(token[2:])
        elif token.startswith("-l"):
            flags["libraries"].append(token[2:])
    return flags


def main() -> int:
    flags = mpi_flags()
    for key in ("include_dirs", "library_dirs", "libraries"):
        if flags[key]:
            print(f"{key} += {flags[key]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
