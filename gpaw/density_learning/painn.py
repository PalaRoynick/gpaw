"""PaiNN adapter, periodic probe graphs and a nonnegative finite-range readout."""

from functools import lru_cache
import importlib
import itertools
from pathlib import Path
import sys

import numpy as np
from gpaw.density_learning.dependency import source_directory
from ase.neighborlist import neighbor_list
from scipy.spatial import cKDTree

try:
    import torch
    from torch import nn
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = nn = None


def implementation_fingerprints():
    from gpaw.density_learning.io import file_hash

    directory = source_directory()
    return {
        name: file_hash(directory / name) for name in ["densitymodel.py", "layer.py"]
    }


def require_torch():
    if torch is None:
        raise ImportError(
            f"The optional PyTorch dependency is missing from {sys.executable}. Activate the rootless ML "
            "environment, or run tools/deepdft/enable_training.py with this Python."
        )


def optional_no_grad(function):
    return torch.no_grad()(function) if torch is not None else function


@lru_cache(maxsize=1)
def external_model():
    require_torch()
    path = source_directory()
    if not (path / "densitymodel.py").is_file():
        raise RuntimeError("Initialize the pinned DeepDFT submodule")
    for name in ["layer", "densitymodel"]:
        previous = sys.modules.get(name)
        if previous is not None and Path(previous.__file__).resolve().parent != path:
            raise RuntimeError(f"Conflicting imported module: {name}")
    sys.path.insert(0, str(path))
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return importlib.import_module("densitymodel").PainnDensityModel
    finally:
        sys.path.pop(0)
        sys.dont_write_bytecode = previous_bytecode


def tensor(value, integer=False):
    require_torch()
    return torch.as_tensor(
        np.asarray(value).copy(), dtype=torch.long if integer else torch.float32
    ).unsqueeze(0)


class ProbeGraph:
    """Cache atom-only edges and an image KD-tree; never add probe-probe edges."""

    def __init__(self, atoms, cutoff, max_images=5_000_000):
        if len(atoms) < 1 or atoms.cell.volume <= 0 or cutoff <= 0:
            raise ValueError(
                "A nonempty structure, full-rank cell and positive cutoff are required"
            )
        self.atoms = atoms
        self.cutoff = cutoff
        i, j, shifts = neighbor_list("ijS", atoms, cutoff, self_interaction=False)
        self.base = {
            "nodes": tensor(atoms.numbers, True),
            "atom_xyz": tensor(atoms.positions),
            "cell": tensor(atoms.cell.array),
            "atom_edges": tensor(np.column_stack((j, i)), True),
            "atom_edges_displacement": tensor(shifts),
            "num_nodes": torch.tensor([len(atoms)]),
            "num_atom_edges": torch.tensor([len(i)]),
        }
        # ASE shifts point from receiver i to the periodic image of sender j.
        heights = 1.0 / np.linalg.norm(np.linalg.inv(atoms.cell.array), axis=0)
        ranges = [
            range(-int(np.ceil(cutoff / h)) - 1, int(np.ceil(cutoff / h)) + 2)
            if periodic
            else range(1)
            for h, periodic in zip(heights, atoms.pbc)
        ]
        image_count = np.prod([len(r) for r in ranges]) * len(atoms)
        if image_count > max_images:
            raise MemoryError(
                "Probe image budget exceeded; use spatially tiled inference"
            )
        image_shifts = np.array(list(itertools.product(*ranges)), dtype=int)
        positions = (
            atoms.positions[None, :, :] + (image_shifts @ atoms.cell)[:, None, :]
        )
        self.positions = positions.reshape(-1, 3)
        self.source = np.tile(np.arange(len(atoms)), len(image_shifts))
        self.shifts = np.repeat(image_shifts, len(atoms), axis=0)
        self.tree = cKDTree(self.positions)

    def batch(self, probes, device="cpu"):
        probes = np.asarray(probes, dtype=float).reshape(-1, 3)
        neighbours = self.tree.query_ball_point(probes, self.cutoff)
        lengths = np.fromiter(
            (len(n) for n in neighbours), dtype=int, count=len(probes)
        )
        images = (
            np.concatenate(neighbours).astype(int)
            if lengths.sum()
            else np.empty(0, int)
        )
        destinations = np.repeat(np.arange(len(probes)), lengths)
        edges = np.column_stack((self.source[images], destinations))
        batch = dict(
            self.base,
            probe_xyz=tensor(probes),
            probe_edges=tensor(edges, True),
            probe_edges_displacement=tensor(self.shifts[images]),
            num_probes=torch.tensor([len(probes)]),
            num_probe_edges=torch.tensor([len(images)]),
        )
        return {key: value.to(device) for key, value in batch.items()}


class PaiNNDensity(nn.Module if nn is not None else object):
    """Local PaiNN with a positive, smoothly vanishing vacuum readout.

    One structure per microbatch. The cosine gate forces zero beyond cutoff
    from every atom; validate omitted tails explicitly. No charge normalization
    is applied, especially not to PAW pseudo-density integrals.
    """

    def __init__(self, cutoff=5.0, interactions=3, hidden=64, scale=0.1):
        require_torch()
        super().__init__()
        if cutoff <= 0 or interactions < 1 or hidden < 1 or scale <= 0:
            raise ValueError("Invalid PaiNN architecture")
        self.config = dict(
            cutoff=cutoff, interactions=interactions, hidden=hidden, scale=scale
        )
        self.network = external_model()(interactions, hidden, cutoff)

    @property
    def receptive_radius(self):
        return (self.config["interactions"] + 1) * self.config["cutoff"]

    def encode(self, batch):
        return self.network.atom_model(batch)

    def forward(self, batch, encoded=None):
        if batch["nodes"].shape[0] != 1:
            raise ValueError("This adapter expects one structure per microbatch")
        if encoded is None:
            encoded = self.encode(batch)
        raw = self.network.probe_model(batch, *encoded)
        edges = batch["probe_edges"][0]
        n = batch["probe_xyz"].shape[1]
        minimum = torch.full((n,), self.config["cutoff"], device=raw.device)
        if len(edges):
            displacement = batch["probe_edges_displacement"][0] @ batch["cell"][0]
            delta = (
                batch["atom_xyz"][0][edges[:, 0]]
                + displacement
                - batch["probe_xyz"][0][edges[:, 1]]
            )
            minimum.scatter_reduce_(
                0,
                edges[:, 1],
                torch.linalg.vector_norm(delta, dim=1),
                reduce="amin",
                include_self=True,
            )
        gate = 0.5 * (1.0 + torch.cos(torch.pi * minimum / self.config["cutoff"]))
        return self.config["scale"] * nn.functional.softplus(raw) * gate[None, :]


def load_checkpoint(path, device="cpu"):
    require_torch()
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("schema") != "gpaw-painn-density-v1":
        raise ValueError("Unsupported density checkpoint")
    if (
        "deepdft_source_sha256" in checkpoint
        and checkpoint["deepdft_source_sha256"] != implementation_fingerprints()
    ):
        raise ValueError(
            "DeepDFT implementation differs from the checkpoint provenance"
        )
    model = PaiNNDensity(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint
