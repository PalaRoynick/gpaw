"""Prints GPAW siteconfig.py lines for corresponding to flags that the MPI
wrapper 'mpicxx' would inject. Useful for builds that use a 'raw' compiler
executable like g++ or clang++ instead of the MPI wrapper. Works at least for
OpenMPI (uses mpicxx -show). MPICH may require a different flag.

Usage: append its output straight into siteconfig.py, eg.:
    python tools/print_mpi_siteconfig.py >> siteconfig.py
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
