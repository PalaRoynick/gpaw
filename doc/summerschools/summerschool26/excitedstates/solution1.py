# snippet-basic-imports-start
# basic imports
import numpy as np
from ase import Atoms  # Import atoms
from ase.build import bulk
# snippet-basic-imports-end

Si = bulk('Si', 'diamond', a=5.4)  # student:
Ge = bulk('Ge', 'diamond', a=5.7)  # student:
C = bulk('C', 'diamond', a=3.6)  # student:
CdTe = bulk('CdTe', 'zincblende', 6.5)  # student:
GaAs = bulk('GaAs', 'zincblende', a=5.6)  # student:
BN = Atoms('BN', pbc=[True, True, False], positions=[[0., 0., 7.],[0., 2.5/np.sqrt(3), 7.]], cell=[[2.50, 0., 0.],[-2.5/2., 2.5 * np.sqrt(3.)/2., 0.],[0., 0., 14.]])  # student:

atoms = Si  # student: atoms = ???
label = 'Si'  # student: label = '???'

# view(atoms)  # check your initial structure # student: view(atoms)

# snippet-calc-start-student
from gpaw import GPAW, PW, FermiDirac
xc = ...
occupations = FermiDirac(...)
mode = ...
kpts = ...
# snippet-calc-end-student


xc = 'PBE'
occupations = FermiDirac(0.01)
mode = PW(600)
kpts = {'size': (6,6,6)}

# snippet-calculator-start
# We create the calculator object
calc = GPAW(xc = xc,
            txt = label + '_relax.txt',
            occupations = occupations, # smearing the occupation
            mode = mode, # plane wave basis
            kpts = kpts
)

atoms.calc = calc
# snippet-calculator-end


# snippet-opt-start-student
from ase.filters import UnitCellFilter
from ase.optimize import BFGS
filt = ...
op = ...
fmax = ...
# snippet-opt-end-student

filt = UnitCellFilter(atoms)
op = BFGS(filt)

filt = filt
op = op

fmax = 0.05


# snippet-run-optimization-start
# Run the optimization. This will take some time, do not get nervous.
# Only if it takes longer than 4-5 minutes or if it does not print anything
# contact us :)
op.run(fmax=fmax)

# save the results in a file
calc.write(label + '_gs.gpw')
# snippet-run-optimization-end
