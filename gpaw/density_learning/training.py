"""Reproducible PaiNN training and held-out density validation.

Small reference structures are cached on disk; evaluation integrates the full
grid in probe batches and never substitutes a sampled charge for an integral.
"""

import json
from pathlib import Path
import random
import tempfile

import numpy as np

from gpaw.density_learning.data import (
    grid_points,
    read_sample,
    sample_indices,
    signature,
    surface_atoms,
)
from gpaw.density_learning.io import external_directory, save_json
from gpaw.density_learning.painn import (
    PaiNNDensity,
    ProbeGraph,
    load_checkpoint,
    optional_no_grad,
    require_torch,
    torch,
    implementation_fingerprints,
)


def device_check(device):
    require_torch()
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable; use a GPU allocation or --device cpu"
            )
        if (
            torch.cuda.get_device_capability()[0:2] == (7, 0)
            and "sm_70" not in torch.cuda.get_arch_list()
        ):
            raise RuntimeError("The installed PyTorch wheel does not support this V100")


def set_seed(seed, threads):
    if threads < 1:
        raise ValueError("CPU thread count must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def load_splits(path):
    split = json.loads(Path(path).read_text())
    if split.get("schema") != "gpaw-density-splits-v1":
        raise ValueError("Unsupported split schema")
    seen = set()
    checksums = set()
    for partition in ["train", "validation", "test"]:
        groups = {r["group_id"] for r in split["partitions"][partition]}
        if seen & groups:
            raise ValueError("Parent-group leakage between partitions")
        seen.update(groups)
        for record in split["partitions"][partition]:
            if record["cube_sha256"] in checksums:
                raise ValueError("Duplicate reference in partitions")
            checksums.add(record["cube_sha256"])
            if signature(record) != split["reference_signature"]:
                raise ValueError("Reference signature mismatch")
    return split


def prepare(record, cache, cutoff):
    sample = read_sample(record, cache)
    if np.min(sample["density"]) < -1e-7:
        raise ValueError(
            "Reference has significant negative electron density; check grid/export"
        )
    sample["graph"] = ProbeGraph(sample["atoms"], cutoff)
    return sample


def save_checkpoint(path, checkpoint):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        torch.save(checkpoint, stream)
    temporary.replace(path)


def train(
    split_path,
    output,
    *,
    device="cpu",
    steps=10000,
    probes=1024,
    threads=4,
    seed=0,
    learning_rate=1e-3,
    cutoff=5.0,
    interactions=3,
    hidden=64,
    validate_every=100,
    surface_fraction=0.5,
    resume=None,
):
    if min(steps, probes, validate_every) < 1 or not 0 <= surface_fraction <= 1:
        raise ValueError("Invalid training budget or sampling fraction")
    device_check(device)
    set_seed(seed, threads)
    output = external_directory(output)
    if (output / "best.pt").exists() and resume is None:
        raise FileExistsError(
            "A model already exists; use resume or a new output directory"
        )
    split = load_splits(split_path)
    training_config = dict(
        seed=seed,
        probes=probes,
        learning_rate=learning_rate,
        surface_fraction=surface_fraction,
        validate_every=validate_every,
    )
    if resume:
        model, previous = load_checkpoint(resume, device)
        if previous["reference_signature"] != split["reference_signature"]:
            raise ValueError("Resume checkpoint uses a different density protocol")
        if previous["split"] != split:
            raise ValueError("Resume checkpoint uses a different dataset/split")
        if previous["training_config"] != training_config:
            raise ValueError(
                "Resume must preserve seed, sampling, learning rate and validation interval"
            )
        if (cutoff, interactions, hidden) != (
            model.config["cutoff"],
            model.config["interactions"],
            model.config["hidden"],
        ):
            raise ValueError("Resume must preserve the model architecture")
    else:
        model = PaiNNDensity(cutoff, interactions, hidden).to(device)
        previous = None
    cutoff = model.config["cutoff"]
    samples = {
        part: [prepare(r, output / "cache", cutoff) for r in split["partitions"][part]]
        for part in ["train", "validation"]
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if previous:
        optimizer.load_state_dict(previous["optimizer"])
        torch.set_rng_state(previous["torch_rng"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in previous["cuda_rng"]]
            )
    rng = np.random.default_rng(seed)
    if previous:
        rng.bit_generator.state = json.loads(previous["numpy_rng"])
    validation = []
    validation_rng = np.random.default_rng(seed + 100)
    for sample in samples["validation"]:
        indices = validation_rng.integers(sample["density"].size, size=probes)
        points = grid_points(
            indices, sample["density"].shape, sample["atoms"].cell, sample["origin"]
        )
        validation.append(
            (
                sample["graph"].batch(points, device),
                torch.as_tensor(
                    np.asarray(sample["density"]).reshape(-1)[indices].copy(),
                    dtype=torch.float32,
                    device=device,
                )[None, :],
            )
        )
    best = previous["best_validation_mse"] if previous else float("inf")
    start = previous["step"] + 1 if previous else 0
    if start >= steps:
        raise ValueError("steps must exceed the checkpoint step")
    metadata = {
        "schema": "gpaw-painn-training-v1",
        "device": device,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name() if device == "cuda" else "CPU",
        "deepdft_source_sha256": implementation_fingerprints(),
        "seed": seed,
        "threads": threads,
        "surface_fraction": surface_fraction,
        "model_config": model.config,
        "reference_signature": split["reference_signature"],
        "effective_radius_angstrom": model.receptive_radius,
        "validation_is_sampled": True,
        "split": split,
    }
    save_json(output / "training.json", metadata)
    with (output / "history.jsonl").open("a") as history:
        for step in range(start, steps):
            model.train()
            sample = samples["train"][int(rng.integers(len(samples["train"])))]
            indices = sample_indices(sample, probes, rng, surface_fraction)
            points = grid_points(
                indices, sample["density"].shape, sample["atoms"].cell, sample["origin"]
            )
            batch = sample["graph"].batch(points, device)
            target = torch.tensor(
                np.asarray(sample["density"]).reshape(-1)[indices].copy(),
                dtype=torch.float32,
                device=device,
            )[None, :]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(batch) - target) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise RuntimeError("Missing or nonfinite gradients")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=10.0, error_if_nonfinite=True
            )
            optimizer.step()
            if step % validate_every == 0 or step + 1 == steps:
                model.eval()
                with torch.no_grad():
                    score = float(
                        torch.stack(
                            [torch.mean((model(b) - t) ** 2) for b, t in validation]
                        )
                        .mean()
                        .cpu()
                    )
                if not np.isfinite(score):
                    raise RuntimeError("Nonfinite validation metric")
                entry = {
                    "step": step,
                    "train_mse": float(loss.detach().cpu()),
                    "validation_mse": score,
                }
                history.write(json.dumps(entry) + "\n")
                history.flush()
                checkpoint = {
                    "schema": "gpaw-painn-density-v1",
                    "model_config": model.config,
                    "deepdft_source_sha256": metadata["deepdft_source_sha256"],
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "reference_signature": split["reference_signature"],
                    "split": split,
                    "training_config": training_config,
                    "species_numbers": sorted(
                        {
                            int(z)
                            for sample in samples["train"]
                            for z in sample["atoms"].numbers
                        }
                    ),
                    "step": step,
                    "best_validation_mse": min(best, score),
                    "numpy_rng": json.dumps(rng.bit_generator.state),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()
                    if device == "cuda"
                    else [],
                }
                save_checkpoint(output / "last.pt", checkpoint)
                if score < best:
                    best = score
                    save_checkpoint(output / "best.pt", checkpoint)
                print(json.dumps(entry), flush=True)
    return metadata


@optional_no_grad
def evaluate(
    split_path,
    checkpoint_path,
    output,
    *,
    partition="test",
    device="cpu",
    batch_size=4096,
    threads=4,
):
    if batch_size < 1:
        raise ValueError("Validation batch_size must be positive")
    device_check(device)
    set_seed(0, threads)
    output = external_directory(output)
    split = load_splits(split_path)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if (
        checkpoint["reference_signature"] != split["reference_signature"]
        or checkpoint["split"] != split
    ):
        raise ValueError("Checkpoint and evaluation dataset do not match")
    results = []
    for record in split["partitions"][partition]:
        sample = prepare(record, output / "cache", model.config["cutoff"])
        density = sample["density"]
        graph = sample["graph"]
        encoded = model.encode({k: v.to(device) for k, v in graph.base.items()})
        total_abs = total_sq = total_pred = total_ref_abs = 0.0
        plane_ref = np.zeros(density.shape[2])
        plane_pred = np.zeros_like(plane_ref)
        plane_count = np.zeros_like(plane_ref)
        region = {
            name: [0.0, 0] for name in ["near_atoms", "low_density", "far_from_atoms"]
        }
        region["near_undercoordinated_atoms"] = [0.0, 0]
        undercoordinated = surface_atoms(sample["atoms"])
        center = (
            sample["atoms"].get_center_of_mass()
            if not sample["atoms"].pbc.any()
            else sample["atoms"].cell.array.sum(axis=0) / 2.0
        )
        corners = (
            np.array([[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]])
            @ sample["atoms"].cell
        )
        radial_count = np.zeros(
            int(np.ceil(np.linalg.norm(corners - center, axis=1).max() / 0.25)) + 1
        )
        radial_ref = np.zeros_like(radial_count)
        radial_pred = np.zeros_like(radial_count)
        for start in range(0, density.size, batch_size):
            indices = np.arange(start, min(start + batch_size, density.size))
            points = grid_points(
                indices, density.shape, sample["atoms"].cell, sample["origin"]
            )
            predicted = (
                model(graph.batch(points, device), encoded).cpu().numpy().ravel()
            )
            if not np.isfinite(predicted).all():
                raise RuntimeError("Nonfinite validation prediction")
            reference = np.asarray(density).reshape(-1)[indices]
            error = predicted - reference
            total_abs += float(np.abs(error).sum(dtype=np.float64))
            total_sq += float(np.square(error, dtype=np.float64).sum())
            total_pred += float(predicted.sum(dtype=np.float64))
            total_ref_abs += float(np.abs(reference).sum(dtype=np.float64))
            z = np.unravel_index(indices, density.shape)[2]
            np.add.at(plane_ref, z, reference)
            np.add.at(plane_pred, z, predicted)
            np.add.at(plane_count, z, 1)
            nearest, images = graph.tree.query(points)
            masks = {
                "near_atoms": nearest < 2.0,
                "low_density": reference < 0.01,
                "far_from_atoms": nearest >= 3.0,
                "near_undercoordinated_atoms": (nearest < 3.0)
                & undercoordinated[graph.source[images]],
            }
            for name, mask in masks.items():
                region[name][0] += float(np.abs(error[mask]).sum(dtype=np.float64))
                region[name][1] += int(mask.sum())
            bins = np.floor(np.linalg.norm(points - center, axis=1) / 0.25).astype(int)
            np.add.at(radial_count, bins, 1)
            np.add.at(radial_ref, bins, reference)
            np.add.at(radial_pred, bins, predicted)
        dv = sample["atoms"].get_volume() / density.size
        ref_count = record["integral_electrons"]
        results.append(
            {
                "sample_id": record["sample_id"],
                "group_id": record["group_id"],
                "family": record["family"],
                "mae_e_per_angstrom3": total_abs / density.size,
                "rmse_e_per_angstrom3": (total_sq / density.size) ** 0.5,
                "normalized_l1": total_abs / max(total_ref_abs, 1e-20),
                "reference_grid_electrons": ref_count,
                "predicted_grid_electrons": total_pred * dv,
                "grid_electron_error": total_pred * dv - ref_count,
                "regions": {
                    name: {"mae": value / count if count else None, "count": count}
                    for name, (value, count) in region.items()
                },
                "plane_reference_e_per_angstrom3": (plane_ref / plane_count).tolist(),
                "plane_prediction_e_per_angstrom3": (plane_pred / plane_count).tolist(),
                "profile_fractional_z": (
                    np.arange(len(plane_ref)) / len(plane_ref)
                ).tolist(),
            }
        )
        results[-1].update(
            radial_center_angstrom=center.tolist(),
            radial_radius_angstrom=(
                (np.arange(len(radial_count)) + 0.5) * 0.25
            ).tolist(),
            radial_grid_counts=radial_count.astype(int).tolist(),
            radial_reference_e_per_angstrom3=[
                float(v / c) if c else None for v, c in zip(radial_ref, radial_count)
            ],
            radial_prediction_e_per_angstrom3=[
                float(v / c) if c else None for v, c in zip(radial_pred, radial_count)
            ],
        )
    result = {
        "schema": "gpaw-density-evaluation-v1",
        "partition": partition,
        "full_grid": True,
        "samples": results,
    }
    save_json(output / "metrics.json", result)
    return result
