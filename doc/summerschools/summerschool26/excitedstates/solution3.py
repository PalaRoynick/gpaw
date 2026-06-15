# snippet-rpa-groundstate-start
from gpaw import GPAW
from gpaw.occupations import FermiDirac


# Load and initialize ground state gpw file from previous exercise
calc_old = GPAW('Si_gs_LDA.gpw', txt=None)  # student: calc_old = GPAW('.gpw', txt=None)

# Extract number of valence bands:
nval = int(calc_old.wfs.nvalence)

# Do new ground state calculations with more k-points.
# This is because in general RPA calculations requires more k-poins to be converged.

calc = GPAW('Si_gs_LDA.gpw').fixed_density(  # student: calc = GPAW('???.gpw').fixed_density(
    kpts=(12, 12, 4),  # student: kpts=???,
    nbands=8 * nval,  # number of bands to include in calculation
    convergence={'bands': 6 * nval},  # number of bands to convergence
    txt='es.txt',  # student: txt = '???',
    occupations=FermiDirac(width=1e-4))


calc.get_potential_energy()
# Now save the .gpw file. 'all' means we save all the wave functions to the gpw file. This is required for the rpa calculations
calc.write('Si_12x12x4.gpw', 'all')  # student: calc.write('', 'all')
# snippet-rpa-groundstate-end

# snippet-rpa-polarizability-start
# Define parameters for rpa calculations. You should change the name of the output file so it corresponds with your calculation:
from gpaw import GPAW
from gpaw.occupations import FermiDirac
from gpaw.response.df import DielectricFunction

# Insert the name of your structure and the k-point grid you are using instead of "???".

calc_old = GPAW('Si_gs_LDA.gpw', txt=None)  # student: calc_old = GPAW('.gpw', txt=None)
nval = int(calc_old.wfs.nvalence)

# Note: For a 2D material (here BN) we use an additional keyword in the parameters below: 'truncation': '2D'
# This truncates the Coulomb interaction in the lateral direction to avoid non-physical periodic interactions.

kwargs = {
    'frequencies': {
        'type': 'nonlinear',
        'domega0': 0.01},  # Define spacing of frequency grid
    'eta': 0.05,           # Broadening parameter
    'intraband': False,    # Here we do not include intraband transitions for calculating the absorption spectrum
    'nblocks': 8,          # Number of blocks used for parallelization
    'ecut': 50,            # Plane wave cutoff in eV
    'nbands': 3 * nval}    # Number of bands included in rpa calculation

# Calculate dielectric function. Takes ground state calculation and defined parameters in "kwargs" as input:
df = DielectricFunction('Si_12x12x4.gpw', **kwargs)  # student: df = DielectricFunction('.gpw', **kwargs)


# Finally we calculate he polarizability in the x, y, and z direction. The output is a .csv file (one for each direction) which can be plotted.
# Consider the symmetries of your material and figure out if calculating the absorption spectrum in all 3 directions
# is needed or two or more of them will be the same. In that case only do the directions you will need (this is for saving
# time and memory). Also remember to change the filename so it fits with your calculation.

df.get_polarizability(xc='RPA',                         # We want to calculate the absorption spectrum within RPA
                      q_c=[0, 0, 0],                    # We consider the zero momentum wave vector
                      direction='x',                    # Define real space direction
                      filename='Si_rpa_x.csv')  # student: filename='=???_rpa_x.csv'       #Name of output file

df.get_polarizability(xc='RPA',
                      q_c=[0, 0, 0],
                      direction='y',
                      filename='Si_rpa_y.csv')  # student: filename='=???_rpa_y.csv'
df.get_polarizability(xc='RPA',
                      q_c=[0, 0, 0],
                      direction='z',
                      filename='Si_rpa_z.csv')  # student: filename='=???_rpa_z.csv'
# snippet-rpa-polarizability-end
