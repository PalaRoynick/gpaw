"""Optional GPAW-to-DeepDFT data and pretrained CPU inference checks.

These checks validate interoperability, not the accuracy of QM9 models on
GPAW densities. The pretrained models were trained against VASP data.
"""
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest
from ase.build import molecule
from ase.io import write
from ase.units import Bohr

from gpaw import GPAW


pytestmark = pytest.mark.serial
DEEPDFT = Path(__file__).resolve().parents[3] / 'third_party' / 'DeepDFT'


@pytest.fixture
def deepdft(monkeypatch):
    if not (DEEPDFT / 'densitymodel.py').is_file():
        pytest.skip('Initialize the DeepDFT submodule first')
    for dependency in ['torch', 'asap3', 'lz4.frame']:
        pytest.importorskip(dependency)
    # DeepDFT is a source checkout with absolute imports, not a Python package.
    # Do not create untracked bytecode caches inside the external submodule.
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    monkeypatch.syspath_prepend(str(DEEPDFT))
    return importlib.import_module('dataset'), importlib.import_module(
        'predict_with_model')


@pytest.mark.parametrize('model_name', ['qm9_schnet', 'qm9_painn'])
def test_gpaw_density_and_inference(deepdft, tmp_path, model_name):
    import torch

    dataset, prediction = deepdft
    atoms = molecule('H2')
    atoms.center(vacuum=2.0)
    atoms.pbc = True
    atoms.calc = GPAW(mode='fd', gpts=(16, 16, 16), xc='LDA', txt=None)
    assert np.isfinite(atoms.get_potential_energy())
    # GPAW's public API returns the spin-summed all-electron density in e/Å³.
    density = atoms.calc.get_all_electron_density(gridrefinement=2)
    dv = atoms.get_volume() / density.size
    assert density.sum() * dv == pytest.approx(2.0, abs=1e-5)

    cube = tmp_path / 'h2.cube'
    # Gaussian cube density convention is e/Bohr³; ASE does not convert data.
    write(cube, atoms, data=density * Bohr**3)
    loaded, loaded_atoms, origin = dataset._read_cube(cube.read_bytes())
    # Cube output stores six significant digits, hence a 5e-6 relative bound.
    np.testing.assert_allclose(loaded, density, rtol=5e-6, atol=1e-12)
    assert loaded.sum() * dv == pytest.approx(2.0, abs=1e-5)
    np.testing.assert_allclose(loaded_atoms.positions, atoms.positions,
                               atol=1e-6, rtol=0)
    # QM9 checkpoints describe nonperiodic molecules, while the reference DFT
    # calculation uses a periodic box. This is an interface check, not an MAE
    # benchmark between those distinct physical reference conventions.
    loaded_atoms.pbc = False
    grid = dataset._calculate_grid_pos(loaded, origin, loaded_atoms.cell)
    rng_state = np.random.get_state()
    try:
        np.random.seed(0)
        graph = dataset.atoms_and_probe_sample_to_graph_dict(
            loaded, loaded_atoms, grid, cutoff=4.0, num_probes=32)
    finally:
        np.random.set_state(rng_state)
    batch = dataset.collate_list_of_dicts([graph])
    model, cutoff = prediction.load_model(
        str(DEEPDFT / 'pretrained_models' / model_name), 'cpu')
    assert cutoff == 4.0
    model.eval()
    with torch.no_grad():
        result = model(batch)
    assert result.shape == (1, 32)
    assert torch.isfinite(result).all()
    with torch.no_grad():
        repeat = model(batch)
    torch.testing.assert_close(repeat, result, rtol=0, atol=0)
