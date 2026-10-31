"""Fixed-cell energy and density convergence comparisons."""

from dataclasses import replace

import numpy as np
from scipy.ndimage import map_coordinates

from gpaw.density_learning.data import grid_points
from gpaw.density_learning.io import external_directory, load_reference, save_json
from gpaw.density_learning.reference import DensityCalculator


def compare(reference, candidate):
    if not np.allclose(reference["atoms"].cell, candidate["atoms"].cell):
        raise ValueError(
            "Compare densities in the same cell; vacuum sweeps need a common physical ROI"
        )
    if not np.array_equal(
        reference["atoms"].numbers, candidate["atoms"].numbers
    ) or not np.allclose(reference["atoms"].positions, candidate["atoms"].positions):
        raise ValueError("Convergence comparison requires identical ionic positions")
    padded = candidate["density"]
    for axis, periodic in enumerate(candidate["atoms"].pbc):
        widths = [(0, 0)] * 3
        widths[axis] = (1, 1)
        padded = np.pad(padded, widths, mode="wrap" if periodic else "constant")
    shape = reference["density"].shape
    total = squared = 0.0
    reference_abs = 0.0
    for start in range(0, reference["density"].size, 100000):
        flat = np.arange(start, min(start + 100000, reference["density"].size))
        points = grid_points(flat, shape, reference["atoms"].cell, reference["origin"])
        coords = (
            (points - candidate["origin"])
            @ np.linalg.inv(candidate["atoms"].cell)
            * candidate["density"].shape
        ).T
        # Padding independently respects periodic and open axes at boundaries.
        other = map_coordinates(
            padded, coords + 1.0, order=1, mode="constant", cval=0.0
        )
        ref = reference["density"].reshape(-1)[flat]
        delta = other - ref
        total += float(np.abs(delta).sum())
        squared += float(np.square(delta).sum())
        reference_abs += float(np.abs(ref).sum())
    return {
        "mae": total / reference["density"].size,
        "rmse": (squared / reference["density"].size) ** 0.5,
        "normalized_l1": total / max(reference_abs, 1e-20),
        "electron_integral_difference": candidate["metadata"]["integral_electrons"]
        - reference["metadata"]["integral_electrons"],
    }


def converge(atoms, settings, output, parameter, values, device="auto", experimental_fd_gpu=False):
    from gpaw.mpi import world, broadcast

    if len(values) < 2:
        raise ValueError(
            "Supply at least two values, ending with the chosen reference value"
        )
    output = external_directory(output)
    for index, value in enumerate(values):
        current = replace(settings, **{parameter: value})
        DensityCalculator(current, device=device,
                          experimental_fd_gpu=experimental_fd_gpu).calculate(
            atoms, output, f"convergence-{index:03d}", "convergence", "convergence"
        )
    error = None
    if world.rank == 0:
        try:
            reference = load_reference(
                output / f"convergence-{len(values) - 1:03d}.cube"
            )
            result = []
            for index, value in enumerate(values):
                candidate = load_reference(output / f"convergence-{index:03d}.cube")
                result.append(
                    {
                        "value": value,
                        **compare(reference, candidate),
                        "energy_difference_ev_per_atom": (
                            candidate["metadata"]["energy_ev"]
                            - reference["metadata"]["energy_ev"]
                        )
                        / len(atoms),
                    }
                )
            save_json(
                output / "convergence.json",
                {
                    "parameter": parameter,
                    "reference": values[-1],
                    "results": result,
                    "accepted_automatically": False,
                },
            )
        except Exception as exception:
            error = str(exception)
    (error,) = broadcast((error,) if world.rank == 0 else None, comm=world)
    if error:
        raise RuntimeError(error)
