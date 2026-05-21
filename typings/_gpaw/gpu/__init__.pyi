"""
GPU specific C++ bindings for GPAW. (EXPERIMENTAL)
"""
from __future__ import annotations
from . import arraytests
__all__: list[str] = ['arraytests', 'inspect']
def inspect(arg0: "NDArray[]") -> None:
    """
    NDArray tester: gets an array from Python side, wraps it with NDArray and prints its metadata
    """
