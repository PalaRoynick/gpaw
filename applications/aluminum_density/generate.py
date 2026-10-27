"""Generate labeled-by-family aluminum geometries; no DFT is run here."""

import argparse
import numpy as np
from ase.build import bulk, fcc100, fcc110, fcc111
from ase.cluster import Decahedron, FaceCenteredCubic, Icosahedron
from ase.io import write

from gpaw.density_learning.io import external_directory, save_json


def sphere(radius, lattice=4.05, vacuum=12.0):
    if radius <= 0 or vacuum <= 0:
        raise ValueError("Radius and vacuum must be positive")
    repeats = max(2, int(np.ceil(2 * radius / lattice)) + 2)
    atoms = bulk("Al", "fcc", a=lattice, cubic=True).repeat((repeats,) * 3)
    center = atoms.cell.lengths() / 2.0
    atoms = atoms[np.linalg.norm(atoms.positions - center, axis=1) <= radius]
    if not len(atoms):
        raise ValueError("No atoms inside sphere")
    atoms.pbc = False
    atoms.center(vacuum=vacuum)
    return atoms


def cavity(radius, lattice=4.05, ligament=12.0):
    if min(radius, ligament) <= 0:
        raise ValueError("Radius and ligament must be positive")
    repeats = max(2, int(np.ceil((2 * radius + 2 * ligament) / lattice)))
    atoms = bulk("Al", "fcc", a=lattice, cubic=True).repeat((repeats,) * 3)
    center = atoms.cell.lengths() / 2.0
    atoms = atoms[np.linalg.norm(atoms.positions - center, axis=1) >= radius]
    atoms.pbc = True
    return atoms


def parents(tier, lattice, vacuum):
    if tier == "smoke":
        yield "bulk-smoke", "bulk", bulk("Al", "fcc", a=lattice, cubic=True)
        yield "slab-smoke", "slab", fcc100("Al", (1, 1, 2), a=lattice, vacuum=vacuum)
        from ase import Atoms

        dimer = Atoms("Al2", positions=[[0, 0, 0], [0, 0, lattice / np.sqrt(2)]])
        dimer.center(vacuum=vacuum)
        yield "finite-smoke", "finite", dimer
        return
    for n in [1, 2, 3]:
        yield (
            f"bulk-{n}",
            "bulk",
            bulk("Al", "fcc", a=lattice, cubic=True).repeat((n,) * 3),
        )
    for name, builder in [("100", fcc100), ("110", fcc110), ("111", fcc111)]:
        for layers in [4, 6, 8]:
            yield (
                f"slab-{name}-{layers}",
                "slab",
                builder(
                    "Al", (2, 2, layers), a=lattice, vacuum=vacuum, orthogonal=True
                ),
            )
    for radius in [3.0, 4.0, 5.0]:
        yield f"sphere-{radius:g}", "particle", sphere(radius, lattice, vacuum)
    for shells in [2, 3]:
        atoms = Icosahedron("Al", shells, latticeconstant=lattice)
        atoms.center(vacuum=vacuum)
        yield f"ico-{shells}", "particle", atoms
        atoms = Decahedron("Al", shells, 2, 0, latticeconstant=lattice)
        atoms.center(vacuum=vacuum)
        yield f"deca-{shells}", "particle", atoms
    atoms = FaceCenteredCubic(
        "Al", [(1, 0, 0), (1, 1, 1)], [2, 2], latticeconstant=lattice
    )
    atoms.center(vacuum=vacuum)
    yield "faceted-2", "particle", atoms
    for radius in [2.5, 3.0, 3.5]:
        slab = fcc100("Al", (4, 4, 6), a=lattice, vacuum=vacuum)
        center = np.r_[slab.cell.lengths()[:2] / 2.0, slab.positions[:, 2].max()]
        slab = slab[np.linalg.norm(slab.positions - center, axis=1) >= radius]
        yield f"pit-{radius:g}", "pit", slab
    for repeats, radius in [(3, 2.5), (3, 3.5), (4, 3.0)]:
        atoms = bulk("Al", "fcc", a=lattice, cubic=True).repeat((repeats,) * 3)
        center = atoms.cell.lengths() / 2.0
        atoms = atoms[np.linalg.norm(atoms.positions - center, axis=1) >= radius]
        yield f"void-{repeats}-{radius:g}", "cavity", atoms


def generate(
    output,
    tier="pilot",
    variants=2,
    seed=0,
    lattice=4.05,
    vacuum=12.0,
    rattle=0.03,
    strain=0.015,
):
    if variants < 1:
        raise ValueError("At least one variant is required")
    output = external_directory(output)
    path = output / "structures.extxyz"
    if path.exists():
        raise FileExistsError(path)
    rng = np.random.default_rng(seed)
    summary = []
    for group, family, parent in parents(tier, lattice, vacuum):
        for variant in range(variants):
            atoms = parent.copy()
            if variant:
                deformation = np.eye(3) + rng.normal(0, strain / 3.0, (3, 3))
                # Preserve orthorhombic cells for the initial inference implementation.
                deformation = np.diag(np.diag(deformation))
                atoms.set_cell(atoms.cell @ deformation, scale_atoms=True)
                atoms.positions += rng.normal(0, rattle, atoms.positions.shape)
                atoms.wrap()
            sample = (group + f"-v{variant}").replace(".", "p")
            atoms.info.update(
                sample_id=sample,
                group_id=group.replace(".", "p"),
                family=family,
                generation_seed=seed,
                relaxed=False,
                purpose="software_smoke_only"
                if tier == "smoke"
                else "candidate_reference",
            )
            atoms.info.pop("adsorbate_info", None)
            if family == "bulk" and tier != "smoke":
                atoms.info["group_id"] = "bulk-fcc-equivalent-supercells"
            write(path, atoms, format="extxyz", append=True)
            summary.append(
                {
                    "sample_id": sample,
                    "group_id": atoms.info["group_id"],
                    "family": family,
                    "natoms": len(atoms),
                    "pbc": atoms.pbc.tolist(),
                }
            )
    save_json(
        output / "structures.json",
        {
            "seed": seed,
            "tier": tier,
            "structures": summary,
            "relaxed": False,
            "rattle_is_not_a_temperature": True,
        },
    )
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tier", choices=["smoke", "pilot"], default="pilot")
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lattice", type=float, default=4.05)
    parser.add_argument("--vacuum", type=float, default=12.0)
    parser.add_argument("--shape", choices=["sphere", "cavity"])
    parser.add_argument(
        "--radius", type=float, help="Radius in angstrom for one inference geometry"
    )
    parser.add_argument("--ligament", type=float, default=12.0)
    args = parser.parse_args()
    if args.shape:
        if args.radius is None:
            parser.error("--shape needs --radius in angstrom")
        output = external_directory(args.output)
        path = output / "structure.extxyz"
        if path.exists():
            raise FileExistsError(path)
        atoms = (
            sphere(args.radius, args.lattice, args.vacuum)
            if args.shape == "sphere"
            else cavity(args.radius, args.lattice, args.ligament)
        )
        write(path, atoms)
        print(
            f"Generated {len(atoms)} atoms; no electronic-structure calculation was run"
        )
    else:
        print(
            generate(
                args.output,
                args.tier,
                args.variants,
                args.seed,
                args.lattice,
                args.vacuum,
            )
        )


if __name__ == "__main__":
    main()
