# creates: ccdists.json
# This script will calculate the energy of graphite for a series of
# C-C distances.
import json
from pathlib import Path
# -- start -- energy-volume
import numpy as np
from ase.calculators.emt import EMT
from ase.lattice.hexagonal import Graphite

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

# -- start -- fit-poly
# Fit a polynomial:
poly = np.polynomial.Polynomial.fit(dists, energies, 3)
# and plot it:
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
ax.plot(dists, energies, '*r')
x = np.linspace(dists.min()-0.1, dists.max()+0.1, 100)
ax.plot(x, poly(x))
ax.set_xlabel('C-C distance [Ang]')
ax.set_ylabel('energy [eV]')
# show plot
if 0:
    plt.show()
# -- end -- fit-poly

# -- start -- opt-layer
poly1 = poly.deriv()
poly1.roots()  # two extrema
# Find the minimum:
emin, ldist = min((poly(d), d) for d in poly1.roots())
print(f'Layer dist: {ldist:.3f} Å')
# alternatively:
poly2 = poly1.deriv()
for ldist in poly1.roots():
    if poly2(ldist) > 0:
        break
print(f'Layer dist: {ldist:.3f} Å')
# -- end -- opt-layer

poly = np.polynomial.Polynomial.fit(dists, energies, 3)
ccdists = next(d for d in poly.deriv().roots() if poly.deriv(2)(d) > 0)
Path('ccdists.json').write_text(json.dumps({'distance': ccdists}))
