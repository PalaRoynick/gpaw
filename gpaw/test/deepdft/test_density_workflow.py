"""Units, leakage, symmetry and tiled-inference regression checks."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.io.cube import read_cube
from ase.units import Bohr

from gpaw.density_learning.data import split_dataset
from gpaw.density_learning.io import CubeWriter, file_hash, load_reference, save_json
from gpaw.density_learning.reference import ReferenceSettings


def make_reference(directory, name, group, value=0.2):
    directory.mkdir(exist_ok=True)
    atoms = Atoms("Al2", positions=[[2, 2, 2], [2, 2, 4.8]], cell=[7, 7, 8])
    density = np.full((7, 7, 8), value)
    path = directory / (name + ".cube")
    with CubeWriter(path, atoms, density.shape) as writer:
        writer.write(density)
    save_json(
        path.with_suffix(".json"),
        {
            "schema": "gpaw-density-v1",
            "sample_id": name,
            "group_id": group,
            "family": "finite",
            "density_kind": "pseudo",
            "pseudo_compensation_charges": False,
            "cell_angstrom": atoms.cell.tolist(),
            "pbc": atoms.pbc.tolist(),
            "shape": list(density.shape),
            "origin_angstrom": [0.0, 0.0, 0.0],
            "numbers": atoms.numbers.tolist(),
            "positions_angstrom": atoms.positions.tolist(),
            "integral_electrons": value * atoms.get_volume(),
            "valence_electrons": 6.0,
            "settings": asdict(ReferenceSettings()),
            "gpaw_version": "synthetic-test-only",
            "setups": [
                {"symbol": "Al", "valence": 3.0, "fingerprint": "synthetic-test-only"}
            ],
            "cube_sha256": file_hash(path),
        },
    )
    return path


def test_cube_units_positions_and_pbc(tmp_path):
    path = make_reference(tmp_path, "reference", "parent")
    with path.open() as stream:
        independent = read_cube(stream)
    np.testing.assert_allclose(independent["data"], 0.2 * Bohr**3, rtol=1e-9)
    reference = load_reference(path, tmp_path / "cache")
    np.testing.assert_allclose(reference["density"], 0.2, rtol=1e-7)
    assert not reference["atoms"].pbc.any()
    np.testing.assert_allclose(
        reference["atoms"].positions, [[2, 2, 2], [2, 2, 4.8]], atol=1e-8
    )
    # No artificial renormalization to 3 electrons/Al atom.
    assert (
        abs(
            reference["density"].sum()
            * reference["atoms"].get_volume()
            / reference["density"].size
            - 6.0
        )
        > 1.0
    )
    cached = load_reference(path, tmp_path / "cache")
    np.testing.assert_array_equal(reference["density"], cached["density"])


def test_incomplete_cube_is_not_published(tmp_path):
    with pytest.raises(ValueError, match="Incomplete"):
        with CubeWriter(
            tmp_path / "bad.cube", Atoms("Al", cell=[4] * 3), (2, 2, 2)
        ) as writer:
            writer.write([0.1])
    assert not (tmp_path / "bad.cube").exists()


def test_reference_tools_import_without_torch():
    program = """
import importlib.abc
import sys
class WithoutTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch':
            raise ModuleNotFoundError('Torch deliberately absent', name='torch')
sys.meta_path.insert(0, WithoutTorch())
import gpaw.density_learning.reference
import gpaw.density_learning.painn
import gpaw.density_learning.training
import gpaw.density_learning.inference
assert 'torch' not in sys.modules
try:
    gpaw.density_learning.painn.PaiNNDensity()
except ImportError as error:
    assert 'optional' in str(error)
else:
    raise AssertionError('PaiNN must require its optional dependencies')
"""
    subprocess.run([sys.executable, "-c", program], check=True)


def test_concurrent_cube_destination_is_not_overwritten(tmp_path):
    path = tmp_path / "shared.cube"
    atoms = Atoms("Al", cell=[4] * 3)
    with pytest.raises(FileExistsError):
        with CubeWriter(path, atoms, (1, 1, 1)) as first:
            with CubeWriter(path, atoms, (1, 1, 1)) as second:
                second.write([0.2])
            first.write([0.9])
    with path.open() as stream:
        result = read_cube(stream)
    np.testing.assert_allclose(result["data"], 0.2 * Bohr**3)


def test_group_split_keeps_variants_together(tmp_path):
    for group in range(6):
        for variant in range(2):
            make_reference(
                tmp_path / "data",
                f"parent-{group}-v{variant}",
                f"parent-{group}",
                0.1 + group * 0.01 + variant * 0.001,
            )
    result = split_dataset(tmp_path / "data", tmp_path / "split")
    partitions = result["partitions"]
    groups = [{r["group_id"] for r in rows} for rows in partitions.values()]
    assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i))
    assert sum(map(len, partitions.values())) == 12
    assert all(len(rows) % 2 == 0 for rows in partitions.values())


@pytest.fixture
def model():
    torch = pytest.importorskip("torch")
    from gpaw.density_learning.painn import PaiNNDensity

    torch.set_num_threads(1)
    torch.manual_seed(3)
    return PaiNNDensity(cutoff=3.0, interactions=2, hidden=8).eval()


@pytest.mark.parametrize("periodic", [False, True])
def test_scalar_density_so3_and_translation(model, periodic):
    import torch
    from gpaw.density_learning.painn import ProbeGraph

    atoms = Atoms("Al3", positions=[[2, 2, 2], [2, 2, 4.8], [4.5, 2, 2]], cell=[9] * 3)
    if periodic:
        atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    points = np.array([[2.5, 3.0, 2.5], [4.0, 1.0, 3.0]])
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    cross = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    angle = 0.73
    rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
    transformed = atoms.copy()
    shift = np.array([1.2, -2.0, 0.5])
    transformed.positions = atoms.positions @ rotation.T + shift
    transformed.cell = atoms.cell @ rotation.T
    with torch.no_grad():
        original = model(ProbeGraph(atoms, 3.0).batch(points)).numpy()
        rotated = model(
            ProbeGraph(transformed, 3.0).batch(points @ rotation.T + shift)
        ).numpy()
    np.testing.assert_allclose(rotated, original, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("periodic", ["finite", "bulk", "slab"])
def test_halo_tiling_matches_full_model(model, periodic):
    import torch
    from gpaw.density_learning.inference import HaloIndex, tile_values
    from gpaw.density_learning.painn import ProbeGraph

    atoms = (
        bulk("Al", "fcc", a=4.05, cubic=True)
        if periodic == "bulk"
        else Atoms(
            "Al20",
            positions=[[2, 2, 2.86 * i + 1.0] for i in range(20)],
            cell=[8, 8, 65],
        )
    )
    if periodic == "slab":
        from ase.build import fcc100

        atoms = fcc100("Al", (2, 2, 3), a=4.05, vacuum=3.0)
    points = np.array([[0.5, 0.5, 1.0], [1.0, 1.0, 2.0]])
    with torch.no_grad():
        full = model(ProbeGraph(atoms, 3.0).batch(points)).numpy()[0]
    tiled = tile_values(model, HaloIndex(atoms), points, "cpu", 1, 5000)
    np.testing.assert_allclose(tiled, full, rtol=2e-5, atol=2e-6)


def test_painn_vacuum_is_zero_and_nonnegative(model):
    import torch
    from gpaw.density_learning.painn import ProbeGraph

    atoms = Atoms("Al2", positions=[[2, 2, 2], [2, 2, 4.8]], cell=[20] * 3)
    with torch.no_grad():
        prediction = model(
            ProbeGraph(atoms, 3.0).batch([[2, 2, 2], [15, 15, 15]])
        ).numpy()[0]
    assert prediction[0] > 0
    assert prediction[1] == 0


def test_training_resume_and_full_grid_validation(tmp_path):
    pytest.importorskip("torch")
    from gpaw.density_learning.training import train, evaluate

    for group in range(3):
        make_reference(
            tmp_path / "data", f"case-{group}", f"group-{group}", 0.1 + group * 0.01
        )
    split_dataset(tmp_path / "data", tmp_path / "split")
    splits = tmp_path / "split/splits.json"
    train(
        splits,
        tmp_path / "model",
        steps=2,
        probes=16,
        interactions=1,
        hidden=8,
        cutoff=3.0,
        threads=1,
        validate_every=1,
    )
    train(
        splits,
        tmp_path / "model",
        steps=3,
        probes=16,
        interactions=1,
        hidden=8,
        cutoff=3.0,
        threads=1,
        validate_every=1,
        resume=tmp_path / "model/last.pt",
    )
    result = evaluate(
        splits,
        tmp_path / "model/best.pt",
        tmp_path / "evaluation",
        batch_size=100,
        threads=1,
    )
    assert result["full_grid"]
    assert len(result["samples"]) == 1
    assert np.isfinite(result["samples"][0]["normalized_l1"])
    history = [
        json.loads(line)
        for line in (tmp_path / "model/history.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in history] == [0, 1, 2]


@pytest.mark.slow
def test_aluminum_reference_export(in_tmp_dir, comm):
    from gpaw.density_learning.reference import DensityCalculator

    settings = ReferenceSettings(
        grid_spacing=0.30,
        kpoint_density=1.0,
        smearing_ev=0.10,
        energy_tolerance=1e-5,
        density_tolerance=1e-4,
        eigenstate_tolerance=1e-6,
    )
    events = []
    result = DensityCalculator(settings, communicator=comm).calculate(
        bulk("Al", "fcc", a=4.05, cubic=True),
        Path.cwd() / "references",
        "al-bulk",
        "bulk-parent",
        "bulk",
        progress=events.append,
    )
    if comm.rank == 0:
        assert events[0]["event"] == "initialization_started"
        assert events[-1]["event"] == "reference_completed"
        assert any(e["event"] == "scf_step" for e in events)
        times = [e["elapsed_seconds"] for e in events]
        assert times == sorted(times)
    else:
        assert events == []
    sample = load_reference(Path.cwd() / "references/al-bulk.cube")
    assert result["mpi_ranks"] == comm.size
    assert result["electrostatic_source_ready"] is False
    assert result["final_magnetic_moment"] == pytest.approx(0.0)
    assert result["pseudo_compensation_charges"] is False
    assert result["valence_electrons"] == 12.0
    assert all(s["valence"] == 3 for s in result["setups"])
    assert np.min(sample["density"]) >= 0
    integral = (
        sample["density"].sum() * sample["atoms"].get_volume() / sample["density"].size
    )
    assert integral == pytest.approx(result["integral_electrons"], abs=1e-7)


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf")])
def test_invalid_wavefunction_budget(budget):
    from gpaw.density_learning.reference import DensityCalculator

    with pytest.raises(ValueError, match="positive and finite"):
        DensityCalculator(max_wavefunction_gib=budget)


def test_reference_budget_stops_before_calculator(tmp_path, monkeypatch):
    import gpaw
    from gpaw.density_learning.reference import DensityCalculator

    def forbidden(*args, **kwargs):
        pytest.fail("GPAW must not be initialized for an over-budget job")

    monkeypatch.setattr(gpaw, "GPAW", forbidden)
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    settings = ReferenceSettings(expected_valence={"Al": 3})
    calculator = DensityCalculator(settings, max_wavefunction_gib=0.01)
    assert calculator.plan(atoms)["wavefunction_budget_satisfied"] is False
    with pytest.raises(MemoryError, match="before SCF"):
        calculator.calculate(atoms, tmp_path / "unused", "bulk", "bulk")
    assert not (tmp_path / "unused").exists()
    calculator = DensityCalculator(settings, max_density_voxels=1)
    with pytest.raises(MemoryError, match="voxel budget before SCF"):
        calculator.calculate(atoms, tmp_path / "unused", "bulk", "bulk")


def test_pw_plan_does_not_claim_fd_storage():
    from gpaw.density_learning.reference import DensityCalculator

    calculator = DensityCalculator(
        ReferenceSettings(mode="pw", expected_valence={"Al": 3})
    )
    assert calculator.plan(bulk("Al"))["wavefunction_storage_gib_total"] is None


def test_cuda_reference_rejects_fd():
    from gpaw.density_learning.reference import DensityCalculator

    with pytest.raises(ValueError, match="experimental_fd_gpu"):
        DensityCalculator(ReferenceSettings(mode="fd"), gpu=True)
    plan = DensityCalculator(ReferenceSettings(mode="pw"), gpu=True).plan(bulk("Al"))
    assert plan["execution_device_requested"] == "cuda"


@pytest.mark.parametrize('shadowed', [False, True])
def test_cuda_environment_rejects_source_shadowing(tmp_path, monkeypatch, shadowed):
    from gpaw.density_learning import runtime

    prefix = tmp_path / 'venv'
    monkeypatch.setattr(sys, 'prefix', str(prefix))
    monkeypatch.setenv('GPAW_CUDA_STATE', str(tmp_path / 'build'))
    paths = {'gpaw_file': str(prefix / 'lib/gpaw/__init__.py'),
             'extension_file': str(prefix / 'lib/_gpaw.so')}
    if shadowed:
        paths['extension_file'] = str(tmp_path / 'checkout/_gpaw.so')
    monkeypatch.setattr(runtime, 'loaded_module_paths', lambda: paths)
    if shadowed:
        with pytest.raises(RuntimeError, match='Leave all GPAW source checkouts'):
            runtime.validate_cuda_environment()
    else:
        runtime.validate_cuda_environment()


@pytest.mark.parametrize('available', [False, True])
def test_reference_auto_device_policy(monkeypatch, available):
    from gpaw.density_learning import devices

    monkeypatch.setattr(devices, 'cuda_status', lambda: available)
    assert devices.select_reference_device('auto', 'pw', 1)[0] == ('cuda' if available else 'cpu')
    assert devices.select_reference_device('auto', 'fd', 1)[0] == 'cpu'
    assert devices.select_reference_device('auto', 'fd', 1, True)[0] == ('cuda' if available else 'cpu')
    assert devices.select_reference_device('auto', 'pw', 2)[0] == 'cpu'
    assert devices.select_reference_device('cpu', 'pw', 1)[0] == 'cpu'
    with pytest.raises(RuntimeError, match='one MPI rank'):
        devices.select_reference_device('cuda', 'pw', 2)
    if not available:
        with pytest.raises(RuntimeError, match='unavailable'):
            devices.select_reference_device('cuda', 'pw', 1)


@pytest.mark.parametrize('slab', [False, True])
def test_host_staged_poisson_preserves_multipole_solution(slab):
    from gpaw.core import UGDesc
    from gpaw.density_learning.poisson import HostStagedPoisson
    from gpaw.poisson import PoissonSolver

    pbc = (True, True, False) if slab else (False, False, False)
    grid = UGDesc(cell=[10, 10, 10], size=[16, 16, 16], pbc=pbc,
                  zerobc=tuple(not p for p in pbc))
    charge = grid.zeros()
    xyz = grid.xyz()
    center1, center2 = ([5, 5, 4], [5, 5, 6]) if slab else ([4, 5, 5], [6, 5, 5])
    charge.data[:] = np.exp(-np.sum((xyz - center1)**2, axis=-1))
    charge.data[:] -= np.exp(-np.sum((xyz - center2)**2, axis=-1))
    params = ({'dipolelayer': 'xy'} if slab else
              {'name': 'MomentCorrectionPoissonSolver',
               'poissonsolver': {'name': 'fast'}, 'moment_corrections': 4})
    direct = PoissonSolver(**params).build(grid, np)
    staged = HostStagedPoisson(params).build(grid, np)
    a, b = grid.zeros(), grid.zeros()
    direct.solve(a, charge)
    staged.solve(b, charge)
    np.testing.assert_allclose(a.data, b.data, rtol=0, atol=1e-12)
    # Exercise transfers using array emulation only, without CUDA kernels.
    from gpaw.gpu import cpupy
    gpu_potential = grid.zeros(xp=cpupy)
    staged.solve(gpu_potential, charge.to_xp(cpupy))
    np.testing.assert_allclose(a.data, cpupy.asnumpy(gpu_potential.data), rtol=0, atol=1e-12)


@pytest.mark.parametrize('available', [False, True])
def test_ml_auto_device(monkeypatch, available):
    torch = pytest.importorskip('torch')
    from gpaw.density_learning.training import device_check

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: available)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda: (7, 0))
    monkeypatch.setattr(torch.cuda, 'get_arch_list', lambda: ['sm_70'])
    assert device_check('auto') == ('cuda' if available else 'cpu')
    assert device_check('cpu') == 'cpu'



def test_experimental_fd_relaxation_uses_bfgs_with_emulated_arrays(tmp_path, monkeypatch):
    from gpaw.gpu import cupy_is_fake
    from gpaw.density_learning import devices
    from gpaw.density_learning.reference import DensityCalculator

    # Exercise device-array operations on CPU-only CI, without claiming hardware validation.
    if cupy_is_fake:
        import gpaw.new.builder
        monkeypatch.setattr(gpaw.new.builder, 'GPAW_CPUPY', True)
    monkeypatch.setattr(devices, 'cuda_status', lambda: True)
    settings = ReferenceSettings(grid_spacing=0.3, kpoint_density=1.0,
                                 smearing_ev=0.1, energy_tolerance=1e-5,
                                 density_tolerance=1e-4, eigenstate_tolerance=1e-6)
    calculator = DensityCalculator(settings, device='cuda', experimental_fd_gpu=True)
    atoms = Atoms('Al2', positions=[[0, 0, 0], [0, 0, 2.4]])
    atoms.center(vacuum=4)
    events = []
    record = calculator.calculate(atoms, tmp_path / 'references', 'al', 'al',
                                  relax_steps=20, fmax=0.02, progress=events.append)
    assert record['relaxation']['requested_steps'] == 20
    assert record['relaxation']['converged']
    assert any(event['event'] == 'experimental_fd_relaxation' for event in events)
    assert record['experimental_fd_gpu'] is True
    assert not np.allclose(record['positions_angstrom'], atoms.positions)


def test_recalculated_reference_rejects_old_split(tmp_path):
    from gpaw.density_learning.data import read_sample

    cube = make_reference(tmp_path, 'al', 'parent', value=0.2)
    snapshot = {'cube': str(cube), **json.loads(cube.with_suffix('.json').read_text())}
    cube.unlink()
    cube.with_suffix('.json').unlink()
    make_reference(tmp_path, 'al', 'parent', value=0.3)
    with pytest.raises(ValueError, match='Reference changed after the split') as error:
        read_sample(snapshot)
    assert str(cube) in str(error.value)
    assert snapshot['cube_sha256'] in str(error.value)
