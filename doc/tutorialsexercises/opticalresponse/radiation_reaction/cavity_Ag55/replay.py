import numpy as np
from ase.parallel import paropen
from gpaw.analyse.wignerseitz import wignerseitz
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter

# Parallelzation settings
parallel = {'sl_auto': True, 'domain': 2, 'augment_grids': True}

with paropen('charge.dat', 'w') as fd:
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
    with paropen('charge.dat', 'a') as fd:
        fd.write(line)


# Time propagation
td_calc = LCAOTDDFT('gs.gpw', parallel=parallel, txt='td.out')
a_g = wignerseitz(td_calc.density.finegd, td_calc.atoms)
mol_atoms = [i for i, sym in enumerate(td_calc.atoms.get_chemical_symbols())
             if sym in ['C', 'O']]
mask_g = np.isin(a_g, mol_atoms)

td_calc.attach(write_charge)
td_calc.replay(name='wfs.ulm', update='all')
