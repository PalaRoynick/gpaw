# web-page: MoS2-lcao-gw-soc.png
"""G0W0+SOC LCAO band-structure interpolation for monolayer MoS2."""

from ase.build import mx2

from gpaw import FermiDirac, GPAW, mpi
from gpaw.dft import DFT
from gpaw.response.g0w0 import G0W0
from gpaw.utilities.bandstructure import (create_soc_bandstructure,
                                          plot_soc_bandstructures)


# A production calculation: converge kpts, ecut and vacuum for your system.
atoms = mx2('MoS2', kind='2H', a=3.184, thickness=3.127, vacuum=10.0)
calc = GPAW(mode='lcao',
            basis='dzp',
            xc='PBE',
            nbands='nao',
            setups={'Mo': '6'},
            kpts={'size': (24, 24, 1), 'gamma': True},
            occupations=FermiDirac(0.01),
            parallel={'domain': 1},
            txt='MoS2-lcao.txt')
atoms.calc = calc
atoms.get_potential_energy()
calc.write('MoS2-lcao.gpw', mode='all')
nao = calc.get_number_of_bands()

# Response calculations use a PW representation of the LCAO wave functions.
# Keep the original LCAO calculation for the interpolation below.
dft = DFT.from_gpw_file('MoS2-lcao.gpw')
dft.change_mode('pw')
dft.write_gpw_file('MoS2-lcao-pw.gpw', include_wfs=True)

# All LCAO bands are needed to Fourier-interpolate the GW correction.
gw = G0W0('MoS2-lcao-pw.gpw',
          filename='MoS2-g0w0',
          bands=(0, nao),
          nbands=nao,
          ecut=200,
          ecut_extrapolation=False,
          truncation='2D',
          integrate_gamma='1BZ2D',
          q0_correction=True,
          frequencies={'type': 'nonlinear',
                       'domega0': 0.025,
                       'omega2': 10.0},
          eta=0.1,
          nblocksmax=True)
results = gw.calculate()
delta_skn = results['qp'] - results['eps']

if mpi.world.rank == 0:
    # Calculator-backed interpolation is deliberately serial and rank-zero
    # only.  A calculator loaded by one rank must use the serial communicator.
    calc = GPAW('MoS2-lcao.gpw', txt=None, communicator=mpi.serial_comm)
    path = atoms.cell.bandpath('GMKG', npoints=300)
    pbe_soc, pbe_sz_kn = create_soc_bandstructure(
        calc, path, world=mpi.world
    )
    gw_soc, gw_sz_kn = create_soc_bandstructure(
        calc, path, shift_skn=delta_skn, world=mpi.world
    )

    # A spin-paired scalar calculation has twice as many SOC spinor bands.
    noccupied_soc = int(round(calc.get_number_of_electrons()))
    vbm = gw_soc.energies[0, :, :noccupied_soc].max()
    plot_soc_bandstructures(
        [('PBE+SOC', pbe_soc, pbe_sz_kn),
         ('G0W0+SOC', gw_soc, gw_sz_kn)],
        'MoS2-lcao-gw-soc.png',
        reference=vbm,
        emin=-4,
        emax=4,
        world=mpi.world,
    )
