"""Check actual DeepDFT forward/backward/Adam steps on a GPAW density.

This small check tests execution and finite gradients, not prediction accuracy
or model convergence. It does not modify external sources or save checkpoints.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from ase.build import molecule

from gpaw import GPAW


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--require-v100', action='store_true')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    if args.require_v100 and args.device != 'cuda':
        parser.error('--require-v100 needs --device cuda')

    report = {'torch': torch.__version__, 'device': args.device,
              'cuda_runtime': torch.version.cuda, 'models': {}}
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; check driver and the CUDA PyTorch wheel')
        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        arches = torch.cuda.get_arch_list()
        report.update(gpu=name, capability=capability, compiled_arches=arches)
        if args.require_v100:
            if 'V100' not in name or capability != (7, 0):
                raise RuntimeError(f'Expected a V100 (sm_70); found {name}, {capability}')
            if 'sm_70' not in arches:
                raise RuntimeError(f'This PyTorch wheel lacks sm_70: {arches}')
        # Exercise the runtime before calculating the reference density.
        x = torch.ones((32, 32), device='cuda')
        assert torch.isfinite(x @ x).all()
        torch.cuda.synchronize()

    dependency = Path(__file__).resolve().parents[2] / 'third_party/DeepDFT'
    if not (dependency / 'densitymodel.py').is_file():
        raise RuntimeError('Run git submodule update --init --recursive')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(dependency))
    import dataset
    import densitymodel

    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    atoms = molecule('H2')
    atoms.center(vacuum=2.0)
    atoms.pbc = True
    atoms.calc = GPAW(mode='fd', gpts=(16, 16, 16), xc='LDA', txt=None)
    energy = atoms.get_potential_energy()
    density = atoms.calc.get_all_electron_density(gridrefinement=2)
    electrons = density.sum() * atoms.get_volume() / density.size
    assert np.isfinite(energy) and abs(electrons - 2.0) < 1e-5
    # Density: spin-summed all-electron e/Å³; positions and cutoff: Å.
    # Nonperiodic model graphs are used only for this execution smoke check.
    atoms.pbc = False
    grid = dataset._calculate_grid_pos(density, np.zeros(3), atoms.cell)
    graph = dataset.atoms_and_probe_sample_to_graph_dict(
        density, atoms, grid, cutoff=4.0, num_probes=64)
    batch = {key: value.to(args.device) for key, value in
             dataset.collate_list_of_dicts([graph]).items()}
    target = batch['probe_target']
    report['reference_electrons'] = float(electrons)
    for cls in [densitymodel.DensityModel, densitymodel.PainnDensityModel]:
        model = cls(num_interactions=2, hidden_state_size=32, cutoff=4.0)
        model.to(args.device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = [p.detach().clone() for p in model.parameters()]
        losses = []
        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            assert prediction.shape == target.shape
            loss = torch.mean((prediction - target)**2)
            assert torch.isfinite(loss), 'Nonfinite training loss'
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            assert any(torch.count_nonzero(g) > 0 for g in gradients)
            optimizer.step()
            assert all(torch.isfinite(p).all() for p in model.parameters())
            losses.append(float(loss.detach().cpu()))
        assert any(not torch.equal(p.detach(), b)
                   for p, b in zip(model.parameters(), before)), 'Parameters unchanged'
        report['models'][cls.__name__] = {'steps': args.steps, 'losses': losses}
    if args.device == 'cuda':
        torch.cuda.synchronize()
    output = json.dumps(report, indent=2)
    print(output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + '\n')


if __name__ == '__main__':
    main()
