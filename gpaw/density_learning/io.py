"""Unit-explicit cube storage with metadata and bounded-memory output."""

from contextlib import AbstractContextManager
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from ase.io.cube import read_cube
from ase import Atoms
from ase.units import Bohr


def external_directory(path):
    path = Path(path).resolve()
    checkout = Path(__file__).resolve().parents[2]
    if path.is_relative_to(checkout):
        raise ValueError(
            "Store datasets, checkpoints and outputs outside the source checkout"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometry_hash(atoms):
    digest = hashlib.sha256()
    for array in [
        np.asarray(atoms.numbers, dtype="<i8"),
        np.asarray(atoms.positions, dtype="<f8"),
        np.asarray(atoms.cell, dtype="<f8"),
        np.asarray(atoms.pbc, dtype=np.uint8),
    ]:
        digest.update(array.tobytes())
    return digest.hexdigest()


class CubeWriter(AbstractContextManager):
    """Write scalar electron density in e/angstrom^3 as standard e/Bohr^3.

    The cube header stores the ionic coordinates and nuclear atomic numbers.
    Values are supplied in C order (z fastest). Commit output only on success.
    """

    def __init__(self, path, atoms, shape, origin=(0.0, 0.0, 0.0)):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.shape = tuple(int(n) for n in shape)
        if len(self.shape) != 3 or min(self.shape) <= 0:
            raise ValueError("Expected three positive grid dimensions")
        self.count = 0
        self.file = tempfile.NamedTemporaryFile(
            mode="w", dir=self.path.parent, delete=False
        )
        self.temporary = Path(self.file.name)
        f = self.file
        f.write("Electron density; e/Bohr^3; atomic coordinates in Bohr\n")
        f.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
        f.write(
            f"{len(atoms):5d}"
            + "".join(f"{x:18.10f}" for x in np.asarray(origin) / Bohr)
            + "\n"
        )
        for n, vector in zip(self.shape, atoms.cell):
            f.write(
                f"{n:5d}" + "".join(f"{x:18.10f}" for x in vector / n / Bohr) + "\n"
            )
        for number, xyz in zip(atoms.numbers, atoms.positions / Bohr):
            f.write(
                f"{number:5d} {0.0:18.10f}" + "".join(f"{x:18.10f}" for x in xyz) + "\n"
            )

    def write(self, values):
        values = np.asarray(values).ravel()
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite cube density")
        self.count += values.size
        if self.count > np.prod(self.shape):
            raise ValueError("Too many cube values")
        np.savetxt(self.file, values * Bohr**3, fmt="%.10e")

    def __exit__(self, kind, value, traceback):
        self.file.close()
        if kind is not None or self.count != np.prod(self.shape):
            self.temporary.unlink(missing_ok=True)
            if kind is None:
                raise ValueError("Incomplete cube grid")
        else:
            # Hard-link publication refuses a concurrent writer's destination.
            try:
                os.link(self.temporary, self.path)
            finally:
                self.temporary.unlink(missing_ok=True)
        return False


def load_reference(cube_path, cache_directory=None):
    """Read a reference cube and its mandatory representation/PBC sidecar."""
    cube_path = Path(cube_path)
    metadata = json.loads(cube_path.with_suffix(".json").read_text())
    if metadata.get("schema") != "gpaw-density-v1":
        raise ValueError("Unsupported density metadata schema")
    if (
        metadata["density_kind"] == "pseudo"
        and metadata.get("pseudo_compensation_charges") is not False
    ):
        raise ValueError(
            "Smooth pseudo references must explicitly exclude compensation charges"
        )
    if file_hash(cube_path) != metadata["cube_sha256"]:
        raise ValueError("Cube checksum mismatch")
    cache = None
    if cache_directory is not None:
        cache = external_directory(cache_directory) / (metadata["cube_sha256"] + ".npy")
    if cache is not None and cache.exists() and "positions_angstrom" in metadata:
        atoms = Atoms(
            metadata["numbers"],
            positions=metadata["positions_angstrom"],
            cell=metadata["cell_angstrom"],
            pbc=metadata["pbc"],
        )
        density = np.load(cache, mmap_mode="r")
        origin = np.array(metadata["origin_angstrom"])
    else:
        with cube_path.open() as stream:
            cube = read_cube(stream)
        density = cube["data"] / Bohr**3
        atoms = cube["atoms"]
        atoms.pbc = metadata["pbc"]
        origin = cube["origin"]
        np.testing.assert_allclose(
            atoms.cell, metadata["cell_angstrom"], atol=1e-7, rtol=0
        )
    if cache_directory is not None:
        if not cache.exists():
            with tempfile.NamedTemporaryFile(dir=cache.parent, delete=False) as stream:
                temporary = Path(stream.name)
                np.save(stream, density.astype(np.float32))
            temporary.replace(cache)
        density = np.load(cache, mmap_mode="r")
    if tuple(density.shape) != tuple(metadata["shape"]):
        raise ValueError("Cached/reference grid shape mismatch")
    return {"atoms": atoms, "density": density, "origin": origin, "metadata": metadata}
