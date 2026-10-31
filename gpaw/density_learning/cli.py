"""Command line interface for reference densities, PaiNN training and inference."""

import argparse
import json
from pathlib import Path
import tomllib

from ase.io import read

from gpaw.density_learning.reference import DensityCalculator, ReferenceSettings


def settings_from_file(path):
    if path is None:
        return ReferenceSettings()
    with Path(path).open("rb") as stream:
        values = tomllib.load(stream)
    return ReferenceSettings(**values.get("reference", values))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ref = commands.add_parser(
        "reference", help="Collectively calculate and export reference cubes"
    )
    ref.add_argument("--input", required=True)
    ref.add_argument("--output", required=True)
    ref.add_argument("--settings")
    ref.add_argument("--gpu", action="store_true",
                     help="Use the single-rank CUDA PW reference profile")
    ref.add_argument("--max-atoms", type=int, default=256)
    ref.add_argument("--max-density-voxels", type=int, default=100_000_000)
    ref.add_argument(
        "--max-wavefunction-gib", type=float,
        help="Reject FD jobs above estimated total wavefunction storage; not a RAM guarantee",
    )
    ref.add_argument("--start", type=int, default=0)
    ref.add_argument("--stop", type=int)
    ref.add_argument("--relax-steps", type=int, default=0)
    ref.add_argument("--fmax", type=float, default=0.02)
    ref.add_argument("--resume-completed", action="store_true")
    ref.add_argument("--plan-only", action="store_true")
    split = commands.add_parser(
        "split", help="Group-disjoint train/validation/test split"
    )
    split.add_argument("--references", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=0)
    training = commands.add_parser("train", help="Train the PaiNN density adapter")
    training.add_argument("--splits", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--steps", type=int, default=10000)
    training.add_argument("--probes", type=int, default=1024)
    training.add_argument("--cutoff", type=float, default=5.0)
    training.add_argument("--interactions", type=int, default=3)
    training.add_argument("--hidden", type=int, default=64)
    training.add_argument("--learning-rate", type=float, default=1e-3)
    training.add_argument("--validate-every", type=int, default=100)
    training.add_argument("--surface-fraction", type=float, default=0.5)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--resume")
    evaluation = commands.add_parser("evaluate", help="Full-grid held-out validation")
    evaluation.add_argument("--splits", required=True)
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument(
        "--partition", choices=["validation", "test"], default="test"
    )
    evaluation.add_argument("--batch-size", type=int, default=4096)
    prediction = commands.add_parser(
        "predict", help="Resource plan or halo-tiled cube prediction"
    )
    prediction.add_argument("--input", required=True)
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--output", required=True)
    prediction.add_argument("--name", default="prediction")
    prediction.add_argument("--spacing", type=float, default=0.2)
    prediction.add_argument("--tile", type=int, default=24)
    prediction.add_argument("--probe-batch", type=int, default=2048)
    prediction.add_argument("--max-voxels", type=int, default=100_000_000)
    prediction.add_argument("--max-halo-atoms", type=int, default=50_000)
    prediction.add_argument("--plan-only", action="store_true")
    convergence = commands.add_parser(
        "converge", help="Fixed-geometry density convergence sweep"
    )
    convergence.add_argument("--input", required=True)
    convergence.add_argument("--index", type=int, default=0)
    convergence.add_argument("--settings")
    convergence.add_argument("--output", required=True)
    convergence.add_argument(
        "--parameter",
        required=True,
        choices=["grid_spacing", "cutoff_ev", "kpoint_density", "smearing_ev"],
    )
    convergence.add_argument("--values", required=True, type=float, nargs="+")
    for command in [training, evaluation, prediction]:
        command.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        command.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "reference":
        from dataclasses import asdict
        from gpaw.mpi import world, broadcast
        from gpaw.density_learning.io import geometry_hash, file_hash

        if not args.plan_only:
            from gpaw.density_learning.runtime import runtime_snapshot

            for rank in range(world.size):
                snapshot = broadcast(
                    runtime_snapshot() if world.rank == rank else None,
                    root=rank, comm=world,
                )
                if world.rank == 0:
                    print(json.dumps({"event": "runtime", "rank": rank,
                                      "mpi_ranks": world.size, **snapshot}), flush=True)
        settings = settings_from_file(args.settings)
        calculator = DensityCalculator(
            settings,
            max_atoms=args.max_atoms,
            max_density_voxels=args.max_density_voxels,
            max_wavefunction_gib=args.max_wavefunction_gib,
            gpu=args.gpu,
        )
        structures = read(args.input, index=":")
        if args.start < 0 or (args.stop is not None and args.stop < args.start):
            raise ValueError("Reference index range must be nonnegative and ordered")
        for index, atoms in enumerate(structures[args.start : args.stop], args.start):
            sample_id = atoms.info.get("sample_id", f"sample-{index:06d}")
            if args.plan_only:
                if world.rank == 0:
                    print(
                        json.dumps({"sample_id": sample_id, **calculator.plan(atoms)}),
                        flush=True,
                    )
                continue
            cube = Path(args.output) / (sample_id + ".cube")
            if args.resume_completed:
                complete, error = False, None
                if world.rank == 0:
                    try:
                        if cube.exists() and cube.with_suffix(".json").exists():
                            record = json.loads(cube.with_suffix(".json").read_text())
                            if record["settings"] != asdict(settings):
                                raise ValueError(
                                    "Completed reference uses different settings"
                                )
                            if record.get("input_geometry_sha256") != geometry_hash(
                                atoms
                            ):
                                raise ValueError(
                                    "Completed reference uses different input geometry"
                                )
                            if record["cube_sha256"] != file_hash(cube):
                                raise ValueError(
                                    "Completed reference cube checksum mismatch"
                                )
                            relaxation = record["relaxation"]
                            if relaxation["requested_steps"] != args.relax_steps or (
                                args.relax_steps and relaxation["fmax"] != args.fmax
                            ):
                                raise ValueError(
                                    "Completed reference uses a different relaxation request"
                                )
                            complete = True
                    except Exception as exception:
                        error = str(exception)
                complete, error = broadcast(
                    (complete, error) if world.rank == 0 else None, comm=world
                )
                if error:
                    raise ValueError(error)
                if complete:
                    continue
            metadata = calculator.calculate(
                atoms,
                args.output,
                sample_id,
                atoms.info.get("group_id", sample_id),
                atoms.info.get("family", "unspecified"),
                args.relax_steps,
                args.fmax,
                progress=lambda event: print(json.dumps(event), flush=True),
            )
            if world.rank == 0:
                print(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "integral": metadata["integral_electrons"],
                        }
                    ),
                    flush=True,
                )
    elif args.command == "split":
        from gpaw.density_learning.data import split_dataset

        split_dataset(args.references, args.output, args.seed)
    elif args.command == "train":
        from gpaw.density_learning.training import train

        train(
            args.splits,
            args.output,
            device=args.device,
            steps=args.steps,
            probes=args.probes,
            threads=args.threads,
            seed=args.seed,
            learning_rate=args.learning_rate,
            cutoff=args.cutoff,
            interactions=args.interactions,
            hidden=args.hidden,
            validate_every=args.validate_every,
            surface_fraction=args.surface_fraction,
            resume=args.resume,
        )
    elif args.command == "evaluate":
        from gpaw.density_learning.training import evaluate

        result = evaluate(
            args.splits,
            args.checkpoint,
            args.output,
            partition=args.partition,
            device=args.device,
            batch_size=args.batch_size,
            threads=args.threads,
        )
        print(
            json.dumps(
                {
                    "metrics_file": str(Path(args.output) / "metrics.json"),
                    "samples": [
                        {
                            key: r[key]
                            for key in [
                                "sample_id",
                                "mae_e_per_angstrom3",
                                "grid_electron_error",
                            ]
                        }
                        for r in result["samples"]
                    ],
                }
            )
        )
    elif args.command == "predict":
        from gpaw.density_learning.inference import predict

        print(
            json.dumps(
                predict(
                    read(args.input),
                    args.checkpoint,
                    args.output,
                    name=args.name,
                    spacing=args.spacing,
                    tile=args.tile,
                    probe_batch=args.probe_batch,
                    device=args.device,
                    threads=args.threads,
                    max_voxels=args.max_voxels,
                    max_halo_atoms=args.max_halo_atoms,
                    plan_only=args.plan_only,
                ),
                indent=2,
            )
        )
    else:
        from gpaw.density_learning.convergence import converge

        converge(
            read(args.input, index=args.index),
            settings_from_file(args.settings),
            args.output,
            args.parameter,
            args.values,
        )


if __name__ == "__main__":
    main()
