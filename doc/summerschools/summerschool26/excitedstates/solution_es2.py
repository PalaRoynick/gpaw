# snippet-pw-groundstate-start
from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw import PW

atoms = bulk('Si', 'diamond', a=5.4) # student:
print(len(atoms))
calc = GPAW(mode=PW(300),  # student: # energy cutoff for plane wave basis (in eV)
            kpts={'size': (3, 3, 3), 'gamma': True}, # student: kpts={'size': (?, ?, ?), 'gamma': True},
            xc='LDA',
            occupations=FermiDirac(0.001),
            parallel={'domain': 1},
            txt='Si_groundstate.txt') # student: txt='???_groundstate.txt'

atoms.calc = calc
atoms.get_potential_energy()

calc.diagonalize_full_hamiltonian()  # determine all bands
calc.write('Si_groundstate.gpw', 'all')  # student: '???_groundstate.gpw', 'all' # write out wavefunctions
# snippet-pw-groundstate-end

# snippet-pw-g0w0-start
from gpaw.response.g0w0 import G0W0

gw = G0W0(calc='Si_groundstate.gpw', # student: calc='???_groundstate.gpw',
          nbands=30,  # student: nbands=???,  # number of bands for calculation of self-energy
          bands=(3, 5), # student: bands=(?, ?), # VB and CB
          ecut=20.0,  # student: ecut=???, # plane-wave cutoff for self-energy (20-200)
          integrate_gamma='WS',  # Use supercell Wigner-Seitz truncation for W.
          filename='Si-g0w0') # student: filename='???-g0w0'
gw.calculate()
# snippet-pw-g0w0-end

# snippet-pw-direct-gap-start
import pickle

results = pickle.load(open('Si-g0w0_results_GW.pckl', 'rb')) # student: open('???-g0w0_results_GW.pckl', 'rb')
direct_gap = results['qp'][0, 0, -1] - results['qp'][0, 0, -2]

print('Direct bandgap of Si:', direct_gap) # student: 'Direct bandgap of ???:', direct_gap
# snippet-pw-direct-gap-end


# snippet-lcao-groundstate-start
calc = GPAW(mode="lcao",  # student mode lcao
            basis="dzp",  # basis set dzp
            nbands="nao", # nbands nao
            kpts={'size': (3, 3, 3), 'gamma': True}, # student: kpts={'size': (?, ?, ?), 'gamma': True},
            xc='LDA',
            occupations=FermiDirac(0.001),
            parallel={'domain': 1},
            txt='Si_lcao_groundstate.txt') # student: txt='???lcao_groundstate.txt'

atoms.calc = calc
atoms.get_potential_energy()

calc.write('Si_lcao_groundstate.gpw', 'all')  # student: '???_lcao_groundstate.gpw', 'all' # write out wavefunctions
# snippet-lcao-groundstate-end

# snippet-lcao-to-pw-start
from gpaw.new.calculation import DFTCalculation
dft = DFTCalculation.from_gpw_file('Si_lcao_groundstate.gpw')
dft.change_mode('pw')
dft.write_gpw_file('Si_pw_from_lcao_groundstate.gpw', include_wfs=True) # student: calc='???_pw_from_lcao_groundstate.gpw'
# snippet-lcao-to-pw-end


# snippet-lcao-g0w0-start
gw = G0W0(calc='Si_pw_from_lcao_groundstate.gpw', # student: calc='???_pw_lcao_groundstate.gpw',
          nbands=26,  # student: nbands=???,  # number of bands for calculation of self-energy
          bands=(3, 5), # student: bands=(?, ?), # VB and CB
          ecut=20.0, # student: ecut=???, # plane-wave cutoff for self-energy (20-200)
          integrate_gamma='WS',  # Use supercell Wigner-Seitz truncation for W.
          filename='Si-from-lcao-g0w0') # student: filename='???-from-lcao-g0w0'
gw.calculate()
# snippet-lcao-g0w0-end

# snippet-lcao-direct-gap-start
results = pickle.load(open('Si-from-lcao-g0w0_results_GW.pckl', 'rb')) # student: open('???-from-lcao-g0w0_results_GW.pckl', 'rb')
direct_gap = results['qp'][0, 0, -1] - results['qp'][0, 0, -2]

print('Direct bandgap of Si:', direct_gap) # student: 'Direct bandgap of ???:', direct_gap
# snippet-lcao-direct-gap-end
