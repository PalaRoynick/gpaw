import pickle
from pathlib import Path

from ase.build import mx2
from ase.units import Ha
from gpaw.band_structure import band_structure
from gpaw.mpi import world
from gpaw.new.ase_interface import GPAW


def mos2():
    """Do HSE06 calculation for MoS2 layer."""
    atoms = mx2(formula='MoS2',
                kind='2H',
                a=3.184,
                thickness=3.13,
                size=(1, 1, 1))
    atoms.center(vacuum=3.5, axis=2)
    k = 6
    atoms.calc = GPAW(mode={'name': 'pw',
                            'ecut': 400},
                      kpts=(k, k, 1),
                      xc='HSE06',
                      txt='hse06_sc.txt')
    atoms.get_potential_energy()
    atoms.calc.write('hse06_sc.gpw', mode='all')
    return atoms


def bandstructure(gs_calc, bp):
    """Calculate HSE06 bandstructure."""
    vacuum_level = gs_calc.dft.vacuum_level()
    kpts = band_structure(gs_calc.dft, bp)
    hse_skn, _ = kpts.get_all_eigs_and_occs()
    return hse_skn[0] * Ha - vacuum_level


def run():
    if 1:
        atoms = mos2()
    else:
        atoms = GPAW('hse06_sc.gpw').get_atoms()
    bp = atoms.cell.bandpath('GMKG', npoints=50)
    # bp = atoms.cell.bandpath('GM', npoints=3)
    hse_kn = bandstructure(atoms.calc, bp)
    if world.rank == 0:
        Path('bs_sc.pckl').write_bytes(
            pickle.dumps((bp, hse_kn)))


if __name__ == '__main__':
    run()
