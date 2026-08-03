kpts_grid = (12, 12, 4)
# snippet-rpa-groundstate-start
from gpaw import GPAW
from gpaw.occupations import FermiDirac


# Load and initialize ground state gpw file from previous exercise
calc_old = GPAW('Si_gs_LDA.gpw', txt=None)

# Extract number of valence bands:
nval = int(calc_old.wfs.nvalence)

# Do new ground state calculations with more k-points.
# This is because in general RPA calculations require more k-points
# to be converged.

calc = GPAW('Si_gs_LDA.gpw').fixed_density(
    kpts=kpts_grid,
    nbands=8 * nval,  # number of bands to include in calculation
    convergence={'bands': 6 * nval},  # number of bands to convergence
    txt='es.txt',
    occupations=FermiDirac(width=1e-4))


calc.get_potential_energy()
# Now save the .gpw file. 'all' means we save all the wave functions to
# the gpw file. This is required for the rpa calculations
gpw_file_name = f'Si_{kpts_grid[0]}x{kpts_grid[1]}x{kpts_grid[2]}.gpw'
calc.write(gpw_file_name, 'all')
# snippet-rpa-groundstate-end

# snippet-rpa-polarizability-start
# Define parameters for rpa calculations. You should change the name of
# the output file so it corresponds with your calculation:
from gpaw import GPAW
from gpaw.occupations import FermiDirac
from gpaw.response.df import DielectricFunction

calc_old = GPAW('Si_gs_LDA.gpw', txt=None)
nval = int(calc_old.wfs.nvalence)

# Note: For a 2D material (here BN) we use an additional keyword in
# the parameters below: 'truncation': '2D'
# This truncates the Coulomb interaction in the lateral direction to
# avoid non-physical periodic interactions.

# snippet-rpa-parameters-start-student
kwargs = {
    'frequencies': {
        'type': 'nonlinear',
        'domega0': ...},  # Define spacing of frequency grid
    'eta': ...,           # Broadening parameter
    'intraband': False,   # Here we do not include intraband transitions
                          # for calculating the absorption spectrum
    'nblocks': 8,         # Number of blocks used for parallelization
    'ecut': ...,          # Plane wave cutoff in eV
    'nbands': ...,        # Number of bands included in rpa calculation
}
# snippet-rpa-parameters-end-student

# snippet-rpa-parameters-solution-start
kwargs = {
    'frequencies': {
        'type': 'nonlinear',
        'domega0': 0.01},
    'eta': 0.05,
    'intraband': False,
    'nblocks': 8,
    'ecut': 50,
    'nbands': 3 * nval}
# snippet-rpa-parameters-solution-end
# Calculate dielectric function. Takes ground state calculation
# and defined parameters in "kwargs" as input:
df = DielectricFunction(gpw_file_name, **kwargs)


# Finally we calculate the polarizability in the x, y, and z direction.
# The output is a .csv file (one for each direction) which can be plotted.
# Consider the symmetries of your material and figure out if calculating
# the absorption spectrum in all 3 directions
# is needed or two or more of them will be the same. In that case only
# do the directions you will need (this is for saving
# time and memory). Also remember to change the filename so it fits
# with your calculation.
# snippet-rpa-directions-start-student
directions = [...]
# snippet-rpa-directions-end-student

# snippet-rpa-directions-solution-start
directions = ['x', 'y', 'z']
# snippet-rpa-directions-solution-end
for direction in directions:
    df.get_polarizability(
        xc='RPA',
        q_c=[0, 0, 0],
        direction=direction,
        filename=f'Si_rpa_{direction}.csv',
    )
# snippet-rpa-polarizability-end
