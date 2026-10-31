"""Group-safe dataset splits and reproducible real-space probe sampling."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from ase.neighborlist import neighbor_list

from gpaw.density_learning.io import external_directory, load_reference, save_json


def records(directory):
    paths = sorted(Path(directory).glob("*.cube"))
    if not paths:
        raise ValueError("No reference cubes found")
    return [
        {"cube": str(p.resolve()), **json.loads(p.with_suffix(".json").read_text())}
        for p in paths
    ]


def signature(record):
    values = dict(record["settings"])
    values["pseudo_compensation_charges"] = record["pseudo_compensation_charges"]
    if not record.get("setups") or not record.get("gpaw_version"):
        raise ValueError("Reference is missing PAW setup or GPAW provenance")
    values["gpaw_version"] = record["gpaw_version"]
    values["setup_fingerprints"] = sorted(
        {(s["symbol"], s["valence"], s["fingerprint"]) for s in record["setups"]}
    )
    return json.dumps(values, sort_keys=True)


def split_dataset(directory, output, seed=0):
    samples = records(directory)
    if len({signature(s) for s in samples}) != 1:
        raise ValueError(
            "Reference protocols differ; do not train a multi-valued target silently"
        )
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["group_id"]].append(sample)
    if len(groups) < 3:
        raise ValueError("At least three independent parent groups are required")
    by_family = defaultdict(list)
    for group, members in groups.items():
        families = {m["family"] for m in members}
        if len(families) != 1:
            raise ValueError("A parent group has inconsistent families")
        by_family[next(iter(families))].append(group)
    rng = np.random.default_rng(seed)
    assigned = {"train": [], "validation": [], "test": []}
    for family in sorted(by_family):
        names = np.array(sorted(by_family[family]))
        rng.shuffle(names)
        if len(names) >= 3:
            n = max(1, int(round(len(names) * 0.15)))
            destinations = (
                ["validation"] * n + ["test"] * n + ["train"] * (len(names) - 2 * n)
            )
        else:
            for name in names:
                destination = min(assigned, key=lambda key: len(assigned[key]))
                assigned[destination].extend(groups[str(name)])
            continue
        for name, destination in zip(names, destinations):
            assigned[destination].extend(groups[str(name)])
    if any(not values for values in assigned.values()):
        raise ValueError("Cannot form nonempty group-disjoint partitions")
    result = {
        "schema": "gpaw-density-splits-v1",
        "seed": seed,
        "reference_signature": signature(samples[0]),
        "partitions": assigned,
    }
    output = external_directory(output)
    if (output / "splits.json").exists():
        raise FileExistsError(output / "splits.json")
    save_json(output / "splits.json", result)
    return result


def grid_points(indices, shape, cell, origin):
    return (
        np.column_stack(np.unravel_index(indices, shape)) / np.asarray(shape) @ cell
        + origin
    )


def surface_atoms(atoms):
    if len(atoms) < 2:
        return np.ones(len(atoms), bool)
    # A geometric sampling heuristic, not a thermodynamic surface definition.
    nearest = atoms.get_all_distances(mic=True)
    np.fill_diagonal(nearest, np.inf)
    cutoff = float(np.min(nearest)) * 1.2
    i = neighbor_list("i", atoms, cutoff, self_interaction=False)
    coordination = np.bincount(i, minlength=len(atoms))
    return coordination < max(12, int(coordination.max()))


def sample_indices(sample, count, rng, surface_fraction=0.5):
    density, atoms = sample["density"], sample["atoms"]
    indices = rng.integers(density.size, size=count)
    surface = np.flatnonzero(surface_atoms(atoms))
    focused = int(count * surface_fraction) if len(surface) else 0
    if focused:
        sites = atoms.positions[rng.choice(surface, focused)]
        points = sites + rng.normal(0.0, 1.5, size=(focused, 3))
        fractional = (points - sample["origin"]) @ np.linalg.inv(atoms.cell)
        grid = np.rint(fractional * density.shape).astype(int)
        for axis in range(3):
            grid[:, axis] = (
                grid[:, axis] % density.shape[axis]
                if atoms.pbc[axis]
                else np.clip(grid[:, axis], 0, density.shape[axis] - 1)
            )
        indices[:focused] = np.ravel_multi_index(grid.T, density.shape)
    return indices


def read_sample(record, cache=None):
    result = load_reference(record["cube"], cache)
    if result["metadata"]["cube_sha256"] != record["cube_sha256"]:
        raise ValueError(
            f"Reference changed after the split was defined: "
            f"sample={record.get('sample_id', 'unknown')}, cube={record['cube']}, "
            f"split_sha256={record['cube_sha256']}, "
            f"current_sha256={result['metadata']['cube_sha256']}. "
            "Audit the split; regenerate it only for intentional reference updates "
            "and start a new training run."
        )
    return result
