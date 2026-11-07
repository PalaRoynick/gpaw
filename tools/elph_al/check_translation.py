"""Check rigid-translation covariance of the complete PAW operator.

Run on a serial, fully diagonalized PW GPW file with Gamma as its first point.
This tests unnormalized Cartesian derivatives, not enforced acoustic g=0.
"""
import numpy as np
from gpaw.old.calculator import GPAW
from gpaw.utilities import unpack_hermitian
from gpaw.elph.gmatrix import ElectronPhononMatrix
from ase.units import Bohr, Hartree
import argparse
import json
from pathlib import Path
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('gpw', type=Path)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
c = GPAW(str(args.gpw), txt=None)
c.initialize_positions(c.atoms)
w = c.wfs
k = w.kpt_u[0]
P = k.P_ani[0]
dP = -w.pt.derivative(k.psit_nG[:], q=k.q)[0]
E = c.get_eigenvalues(kpt=0) / Hartree
D = unpack_hermitian(c.hamiltonian.dH_asp[0][0])
Q = w.setups[0].dO_ii
nabla = w.setups[0].nabla_iiv
psi = np.array([c.get_pseudo_wave_function(
    band=n, kpt=0, periodic=True) for n in range(len(E))])
N = np.array(psi.shape[-3:])
cell = c.atoms.cell.array
dv = abs(np.linalg.det(cell)) / np.prod(N)
freq = np.array(np.meshgrid(
    *[np.fft.fftfreq(n) * n for n in N], indexing='ij')).reshape(3, -1).T
G = (freq @ (2 * np.pi * np.linalg.inv(cell / Bohr).T)).reshape(*N, 3)
V = c.hamiltonian.vt_sG[0]
Vfft = np.fft.fftn(V)
pfft = np.fft.fftn(psi, axes=(-3, -2, -1))
kernels = {key: np.zeros((3, len(E), len(E)), complex)
           for key in ['V', 'D', 'P', 'Q', 'Qright', 'R']}
pAE = []
for v in range(3):
    dV = -np.fft.ifftn(1j * G[..., v] * Vfft).real
    kernels['V'][v] = dv * \
        np.einsum('mxyz,xyz,nxyz->mn', psi.conj(), dV, psi, optimize=True)
    kernels['P'][v] = dP[:, :, v].conj() @ D @ P.T + \
        P.conj() @ D @ dP[:, :, v].T
    kernels['Q'][v] = dP[:, :, v].conj() @ Q @ P.T
    kernels['Qright'][v] = P.conj() @ Q @ dP[:, :, v].T
    kernels['R'][v] = -P.conj() @ nabla[:, :, v] @ P.T
    grad = np.fft.ifftn(1j * G[..., v] * pfft, axes=(-3, -2, -1))
    pAE.append(dv * np.einsum('mxyz,nxyz->mn', psi.conj(), grad,
               optimize=True) + P.conj() @ nabla[:, :, v] @ P.T)
parts = ElectronPhononMatrix._assemble_paw(kernels, E, E)
expected = (E[None, :, None] - E[None, None, :]) * np.array(pAE)
report = dict(
    pseudo_plus_augmentation_norm_error=float(abs(
        dv * np.einsum('mxyz,nxyz->mn', psi.conj(), psi)
        + P.conj() @ Q @ P.T - np.eye(len(E))).max()),
    translation_residual_Hartree_per_Bohr=float(
        abs(parts['total'] - expected).max()),
    expected_scale_Hartree_per_Bohr=float(abs(expected).max()),
    relative_translation_residual=float(
        np.linalg.norm(parts['total'] - expected) / np.linalg.norm(expected)),
    definition='g_uniform = (epsilon_final-epsilon_initial) * '
               '<psi_final|grad_r|psi_initial>_AE')
print(json.dumps(report, indent=2))
if args.output:
    args.output.write_text(json.dumps(report, indent=2) + '\n')
assert report['pseudo_plus_augmentation_norm_error'] < 1e-7
assert report['relative_translation_residual'] < 1e-7
