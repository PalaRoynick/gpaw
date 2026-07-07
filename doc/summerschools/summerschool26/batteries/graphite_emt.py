# creates: ccdists.json
# This script will calculate the energy of graphite for a series of
# C-C distances.
import json
from pathlib import Path
from ase.calculators.emt import EMT
from ase.lattice.hexagonal import Graphite
# -- start -- energy-volume
import numpy as np

# Provide some initial guesses on the distances
ccdist = 1.3
layerdist = 3.7

# We will explore distances around the initial guess
dists = np.linspace(ccdist * .8, ccdist * 1.2, 10)
# The resulting energies are put in the following list
energies = []
for cc in dists:
    # Calculate the lattice parameters a and c from the C-C distance
    # and interlayer distance, respectively
    a = cc * np.sqrt(3)
    c = 2 * layerdist

    gra = Graphite('C', latticeconstant={'a': a, 'c': c})

    # Set the calculator and attach it to the Atoms object with
    # the graphite structure
    calc = EMT()
    gra.calc = calc

    energy = gra.get_potential_energy()
    # Append results to the list
    energies.append(energy)
# -- end -- energy-volume

poly = np.polynomial.Polynomial.fit(dists, energies, 3)
ccdists = next(d for d in poly.deriv().roots() if poly.deriv(2)(d) > 0)
Path('ccdists.json').write_text(json.dumps({'distance': ccdists}))
