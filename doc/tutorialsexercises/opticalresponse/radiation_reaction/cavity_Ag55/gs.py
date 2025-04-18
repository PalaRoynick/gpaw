from ase.io import read
from ase.parallel import parprint
from gpaw import GPAW

# Read the structure from the xyz file
atoms = read('Ag55CO.xyz')

# Add dipole moment corrections to center of metallic nanoparticle
metal_i = [sym in ['Ag'] for sym in atoms.get_chemical_symbols()]
center = atoms.get_positions()[metal_i].mean(axis=0)
parprint(f'Center of NP is at {center}', flush=True)
moment_corrections = [dict(moms=list(range(1 + 3)), center=center)]

# Increase the accuracy of density for ground state
convergence = {'density': 1e-12}

# Use occupation smearing, weak mixer and GLLB weight smearing
# to facilitate convergence
occupations = {'name': 'fermi-dirac', 'width': 0.025}
mixer = {'beta': 0.1, 'nmaxold': 5, 'weight': 50.0}
xc = 'GLLBSC:width=0.002'

# Parallelzation settings
parallel = {'sl_auto': True, 'domain': 2, 'augment_grids': True}

# Apply multipole corrections for monopole and dipoles
poissonsolver = {'name': 'MomentCorrectionPoissonSolver',
                 'moment_corrections': moment_corrections,
                 'poissonsolver': 'fast'}

# Ground-state calculation
calc = GPAW(mode='lcao', xc=xc, h=0.2, nbands=360,
            setups={'Ag': '11'},
            basis={'Ag': 'pvalence.dz', 'default': 'dzp'},
            convergence=convergence, poissonsolver=poissonsolver,
            occupations=occupations, mixer=mixer, parallel=parallel,
            maxiter=1000,
            txt='gs.out',
            symmetry={'point_group': False})
atoms.calc = calc
atoms.get_potential_energy()
calc.write('gs.gpw', mode='all')
