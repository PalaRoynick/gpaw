import numpy as np
from ase.geometry import cell_to_cellpar
from ase.units import Bohr

from gpaw.cgpaw import get_num_threads


def print_cell(gd, pbc_c, log):
    log("""Unit cell:
           periodic     x           y           z      points  spacing""")
    h_c = gd.get_grid_spacings()
    for c in range(3):
        log('  %d. axis:    %s  %10.6f  %10.6f  %10.6f   %3d   %8.4f'
            % ((c + 1, ['no ', 'yes'][int(pbc_c[c])]) +
               tuple(Bohr * gd.cell_cv[c]) +
               (gd.N_c[c], Bohr * h_c[c])))

    par = cell_to_cellpar(gd.cell_cv * Bohr)
    log('\n  Lengths: {:10.6f} {:10.6f} {:10.6f}'.format(*par[:3]))
    log('  Angles:  {:10.6f} {:10.6f} {:10.6f}\n'.format(*par[3:]))

    h_eff = gd.dv**(1.0 / 3.0) * Bohr
    log(f'Effective grid spacing dv^(1/3) = {h_eff:.4f}')
    log()


def print_positions(atoms, log, magmom_av):
    from gpaw.io.plot_atoms import plot_atoms
    log(plot_atoms(atoms))
    log('\nAtomic positions and initial magnetic moments')
    log('\nPositions:')
    symbols = atoms.get_chemical_symbols()
    for a, pos_v in enumerate(atoms.get_positions()):
        symbol = symbols[a]
        log('{0:>4} {1:3} {2[0]:11.6f} {2[1]:11.6f} {2[2]:11.6f}'
            '    ({3[0]:7.4f}, {3[1]:7.4f}, {3[2]:7.4f})'
            .format(a, symbol, pos_v, magmom_av[a]))
    log()


def print_parallelization_details(wfs, ham, log):
    log('Total number of cores used:', wfs.world.size)
    if wfs.kd.comm.size > 1:
        log('Parallelization over k-points:', wfs.kd.comm.size)

    # Domain decomposition settings:
    coarsesize = tuple(wfs.gd.parsize_c)
    finesize = tuple(ham.finegd.parsize_c)

    try:  # Only planewave density
        xc_gd = ham.xc_gd
    except AttributeError:
        xc_gd = ham.finegd
    xcsize = tuple(xc_gd.parsize_c)

    if any(np.prod(size) != 1 for size in [coarsesize, finesize, xcsize]):
        title = 'Domain decomposition:'
        template = '%d x %d x %d'
        log(title, template % coarsesize)
        if coarsesize != finesize:
            log(' ' * len(title), template % finesize, '(fine grid)')
        if xcsize != finesize:
            log(' ' * len(title), template % xcsize, '(xc only)')

    if wfs.bd.comm.size > 1:  # band parallelization
        log('Parallelization over states: %d' % wfs.bd.comm.size)

    if get_num_threads() > 1:  # OpenMP threading
        log(f'OpenMP threads: {get_num_threads()}')
    log()
