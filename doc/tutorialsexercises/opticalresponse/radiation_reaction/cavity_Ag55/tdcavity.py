import numpy as np
from ase.parallel import paropen
from gpaw.analyse.wignerseitz import wignerseitz
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.qed import RRemission
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter

# Parallelzation settings
parallel = {'sl_auto': True, 'domain': 2, 'augment_grids': True}

# Time propagation
env = {'environment': 'cavity',
       'resonance': 4.2,
       'loss': 0.01 * 4.2,
       'quantization_plane': 120,
       'cavity_polarization': [0, 0, 1]}
td_calc = LCAOTDDFT('gs.gpw', parallel=parallel, txt='tdcavity.out',
                    rremission=RRemission(env))
a_g = wignerseitz(td_calc.density.finegd, td_calc.atoms)
mol_atoms = [i for i, sym in enumerate(td_calc.atoms.get_chemical_symbols())
             if sym in ['C', 'O']]
mask_g = np.isin(a_g, mol_atoms)

with paropen('cavcharge.dat', 'w') as fd:
    line = '# Integrated charge on CO molecule\n'
    line += '# {:15} {:15}\n'.format('time', 'charge')
    fd.write(line)


def write_charge():
    gd = td_calc.density.finegd
    rho_g = td_calc.density.rhot_g
    charge = gd.integrate(rho_g * mask_g)
    line = ''
    if td_calc.action == 'kick':
        kx, ky, kz = td_calc.kick_strength
        line += (f'# Kick = [{kx:22.12e}, {ky:22.12e}, {kz:22.12e}]; '
                 f'Time = {td_calc.time:.8f}\n')
    line += '{:17.5f} {:15.8f}\n'.format(td_calc.time, charge)
    with paropen('cavcharge.dat', 'a') as fd:
        fd.write(line)


td_calc.attach(write_charge)
DipoleMomentWriter(td_calc, 'dmcavity.dat')
td_calc.absorption_kick([0, 0, 1e-5])
td_calc.propagate(10, 3000)
