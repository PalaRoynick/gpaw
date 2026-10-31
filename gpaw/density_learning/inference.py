"""Finite-receptive-field tiling with periodic halos and disk-backed cubes."""

import itertools
from pathlib import Path
import shutil
import tempfile

import numpy as np
from ase import Atoms
from scipy.spatial import cKDTree

from gpaw.density_learning.io import (
    CubeWriter,
    external_directory,
    file_hash,
    save_json,
)
from gpaw.density_learning.painn import (
    ProbeGraph,
    load_checkpoint,
    optional_no_grad,
    torch,
)
from gpaw.density_learning.training import device_check


class HaloIndex:
    """Global atom index; construct only the image atoms needed by one tile."""

    def __init__(self, atoms):
        self.atoms = atoms
        self.lengths = np.diag(atoms.cell).copy()
        self.periodic = atoms.pbc.copy()
        if atoms.pbc.any() and (
            not np.allclose(atoms.cell, np.diag(self.lengths)) or min(self.lengths) <= 0
        ):
            raise ValueError("Periodic tiled inference requires an orthorhombic cell")
        positions = atoms.positions.copy()
        for axis in np.flatnonzero(self.periodic):
            positions[:, axis] %= self.lengths[axis]
        self.positions = positions
        self.tree = cKDTree(
            positions, boxsize=np.where(self.periodic, self.lengths, 0.0)
        )

    def local(self, center, radius, cutoff, max_atoms):
        indices = np.array(self.tree.query_ball_point(center, radius), dtype=int)
        if not len(indices):
            return None, None
        relative = self.positions[indices] - center
        for axis in np.flatnonzero(self.periodic):
            relative[:, axis] -= (
                np.rint(relative[:, axis] / self.lengths[axis]) * self.lengths[axis]
            )
        ranges = [
            range(
                -int(np.ceil(radius / length + 0.5)),
                int(np.ceil(radius / length + 0.5)) + 1,
            )
            if periodic
            else range(1)
            for length, periodic in zip(self.lengths, self.periodic)
        ]
        positions, numbers = [], []
        count = 0
        for shift in itertools.product(*ranges):
            image = relative + np.array(shift) * self.lengths
            mask = np.linalg.norm(image, axis=1) <= radius + 1e-9
            count += int(mask.sum())
            if count > max_atoms:
                raise MemoryError(
                    "Tile halo atom budget exceeded; reduce tile size or inspect model range"
                )
            positions.append(image[mask])
            numbers.append(self.atoms.numbers[indices][mask])
        side = 2.0 * (radius + cutoff + 1.0)
        local = Atoms(
            numbers=np.concatenate(numbers),
            positions=np.concatenate(positions) + side / 2.0,
            cell=[side] * 3,
        )
        return local, side / 2.0


def prediction_plan(atoms, spacing, tile, receptive_radius):
    if spacing <= 0 or tile < 1 or len(atoms) < 1 or atoms.cell.volume <= 0:
        raise ValueError("Invalid prediction geometry/grid")
    shape = np.maximum(1, np.ceil(atoms.cell.lengths() / spacing).astype(int))
    voxels = int(np.prod(shape))
    return {
        "natoms": len(atoms),
        "shape": shape.tolist(),
        "voxels": voxels,
        "estimated_disk_bytes": 26 * voxels + 100 * len(atoms),
        "tile_edge_points": tile,
        "receptive_radius_angstrom": receptive_radius,
        "full_atom_graph_required": False,
        "warning": "Execution estimates do not validate extrapolation accuracy",
    }


@optional_no_grad
def tile_values(model, index, points, device, batch_size, max_halo_atoms):
    center = points.mean(axis=0)
    radius = (
        float(np.linalg.norm(points - center, axis=1).max())
        + model.receptive_radius
        + 1e-6
    )
    local, offset = index.local(center, radius, model.config["cutoff"], max_halo_atoms)
    if local is None:
        return np.zeros(len(points), np.float32)
    graph = ProbeGraph(local, model.config["cutoff"])
    encoded = model.encode({key: value.to(device) for key, value in graph.base.items()})
    result = np.empty(len(points), np.float32)
    for start in range(0, len(points), batch_size):
        local_points = points[start : start + batch_size] - center + offset
        result[start : start + batch_size] = (
            model(graph.batch(local_points, device), encoded).cpu().numpy()[0]
        )
    return result


def predict(
    atoms,
    checkpoint_path,
    output,
    *,
    name="prediction",
    spacing=0.2,
    tile=24,
    probe_batch=2048,
    device="auto",
    threads=4,
    max_voxels=100_000_000,
    max_halo_atoms=50_000,
    plan_only=False,
):
    device = device_check(device)
    torch.set_num_threads(threads)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if set(map(int, atoms.numbers)) - set(
        checkpoint.get("species_numbers", atoms.numbers.tolist())
    ):
        raise ValueError("Prediction contains species absent from the training data")
    plan = prediction_plan(atoms, spacing, tile, model.receptive_radius)
    if plan_only:
        return plan
    if probe_batch < 1 or plan["voxels"] > max_voxels:
        raise MemoryError(
            "Grid budget exceeded; inspect the plan and raise max_voxels explicitly"
        )
    if any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for c in name
    ):
        raise ValueError("Use a simple ASCII output name")
    output = external_directory(output)
    cube = output / (name + ".cube")
    if cube.exists() or cube.with_suffix(".json").exists():
        raise FileExistsError(cube)
    if shutil.disk_usage(output).free < plan["estimated_disk_bytes"]:
        raise OSError("Insufficient free disk for the cube and temporary grid")
    index = HaloIndex(atoms)
    shape = np.array(plan["shape"])
    with tempfile.TemporaryDirectory(prefix="density-grid-", dir=output) as temporary:
        grid = np.lib.format.open_memmap(
            Path(temporary) / "grid.npy",
            mode="w+",
            dtype=np.float32,
            shape=tuple(shape),
        )
        for start in itertools.product(*(range(0, int(n), tile) for n in shape)):
            end = np.minimum(np.array(start) + tile, shape)
            dimensions = end - start
            indices = np.indices(dimensions).reshape(3, -1).T + start
            points = indices / shape @ atoms.cell
            values = tile_values(
                model, index, points, device, probe_batch, max_halo_atoms
            )
            if not np.isfinite(values).all() or np.min(values) < 0:
                raise RuntimeError("Invalid predicted electron density")
            slices = tuple(slice(a, b) for a, b in zip(start, end))
            grid[slices] = values.reshape(dimensions)
        grid.flush()
        count = 0.0
        with CubeWriter(cube, atoms, shape) as writer:
            flat = grid.reshape(-1)
            for start in range(0, grid.size, 1_000_000):
                block = flat[start : start + 1_000_000]
                count += float(np.sum(block, dtype=np.float64))
                writer.write(block)
        del grid
    metadata = {
        "schema": "gpaw-density-prediction-v1",
        **plan,
        "density_kind": json_density_kind(checkpoint),
        "pbc": atoms.pbc.tolist(),
        "cell_angstrom": atoms.cell.tolist(),
        "units": "electron/angstrom^3",
        "cube_units": "electron/Bohr^3",
        "checkpoint_sha256": file_hash(checkpoint_path),
        "predicted_grid_electrons": count * atoms.get_volume() / plan["voxels"],
        "charge_normalized": False,
        "halo_exact_for_local_model": True,
    }
    save_json(cube.with_suffix(".json"), metadata)
    return metadata


def json_density_kind(checkpoint):
    import json

    return json.loads(checkpoint["reference_signature"])["density_kind"]
